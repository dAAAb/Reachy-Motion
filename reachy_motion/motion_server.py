"""``reachy-motion-server``: run the flow-matching generator (and Binh Pham's fine-tuned planner) on a Mac for Reachy.

The robot's Raspberry Pi CM4 is too slow to run the generator while it talks (~0.2 s per gesture on an M-series
Mac, seconds on the CM4), and the planner needs Apple Silicon (MLX). This server runs both on the Mac, and
advertises ``_reachy-motion._tcp.local.`` over mDNS so the robot finds it when its motion engine is ``flow`` or
``local`` (see ``motion/backends.py``).

    reachy-motion-server                   # generator + 0.8B planner on port 18110
    reachy-motion-server --planner 4b      # the larger planner (~9 GB download, better motions)
    reachy-motion-server --no-planner      # generator only (for the "flow" engine)

    reachy-motion-server --standby         # stay up but load nothing until the app page presses Start

API (JSON): ``GET /health`` (``state``: off | loading | on | error) · ``POST /control {action: start | stop}``
(load / unload the models; the app's settings page uses it) · ``POST /render {recipe, seed?}`` -> ``{poses: (T, 9), fps}`` ·
``POST /plan {prompt, retries?}`` -> ``{idea, recipe}`` · ``POST /generate-dense {prompt, seed?}`` -> the
reference server's response (``moves`` in the SDK's recorded-move format), so Binh's visualizer can use it too.
It only computes motions — no keys, no audio, nothing private — so it has no authentication; run it on your LAN.
"""

from __future__ import annotations

import argparse
import logging
import socket
import threading
import time

import numpy as np

from reachy_motion.motion.backends import DEFAULT_PORT, LOCAL_PLANNERS, SERVICE_TYPE, LocalPlanner, can_run_mlx
from reachy_motion.motion.flow import FPS, FlowRenderer, _rpy_to_mat
from reachy_motion.motion.recipe import RecipeError

logger = logging.getLogger("reachy_motion.server")


def to_move(A: np.ndarray, description: str = "") -> dict:
    """(T, 9) poses -> the SDK's recorded-move dict (as ``common/motion.py`` ``to_move``)."""
    H = np.tile(np.eye(4), (len(A), 1, 1))
    H[:, :3, :3] = _rpy_to_mat(A[:, 3:6])
    H[:, :3, 3] = A[:, :3]
    return dict(description=description, time=(np.arange(len(A)) / FPS).round(4).tolist(),
                set_target_data=[dict(head=np.round(H[i], 5).tolist(), antennas=np.round(A[i, 6:8], 5).tolist(),
                                      body_yaw=round(float(A[i, 8]), 5), check_collision=False) for i in range(len(A))])


class Models:
    """The generator (+ planner) with an on/off switch: ``off`` holds no weights, so a standby server costs ~nothing."""

    def __init__(self, planner_size: str | None) -> None:
        self.gen = FlowRenderer()
        self.planner_size = planner_size
        self.planner: LocalPlanner | None = None
        self.state, self.detail = "off", ""
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self.state in ("on", "loading"):
                return
            self.state, self.detail = "loading", "loading the models (the first run downloads them)…"
        threading.Thread(target=self._load, name="models-load", daemon=True).start()

    def _load(self) -> None:
        t0 = time.perf_counter()
        planner = None
        if self.planner_size and can_run_mlx():
            planner = LocalPlanner(self.planner_size)
        threads = [threading.Thread(target=self.gen.load)] + ([threading.Thread(target=planner.load)] if planner else [])
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if not self.gen.ready:
            self.state, self.detail = "error", f"generator failed to load: {self.gen.error}"
            logger.error(self.detail)
            return
        self.gen.render("go .4 e=10 p=-4 | hold .5")  # warm-up
        if planner is not None and not planner.ready:
            logger.warning("planner failed to load (%s); serving the generator only", planner.error)
            planner = None
        self.planner = planner
        self.state = "on"
        self.detail = f"ready in {time.perf_counter() - t0:.1f} s"
        logger.info("models on: generator + %s (%s)", planner.repo if planner else "no planner", self.detail)

    def stop(self) -> None:
        with self._lock:
            if self.state == "loading":
                raise RuntimeError("still loading; try again in a few seconds")
            planner, self.planner = self.planner, None
            self.gen.unload()
            if planner is not None:
                planner.unload()
            self.state, self.detail = "off", "models unloaded"
        logger.info("models off")

    def health(self) -> dict:
        try:
            import psutil

            rss = round(psutil.Process().memory_info().rss / 1e6)
        except Exception:  # noqa: BLE001
            rss = None
        p = self.planner
        return {"state": self.state, "detail": self.detail, "generator": self.gen.ready,
                "planner": (p.name.split(":", 1)[1] if p else None), "planner_ready": bool(p and p.ready),
                "planner_size": self.planner_size, "host": socket.gethostname().split(".")[0], "rss_mb": rss, "v": 2}


def make_app(models: Models):
    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(title="reachy-motion-server", docs_url=None, redoc_url=None)

    def render(recipe: str, seed) -> np.ndarray:
        if models.state != "on" or not models.gen.ready:
            raise HTTPException(503, f"models are {models.state}")
        try:
            return models.gen.render(recipe, None if seed is None else int(seed))
        except RecipeError as e:
            raise HTTPException(422, str(e)) from None

    def plan(prompt: str, retries: int) -> tuple[str, str]:
        if models.state != "on":
            raise HTTPException(503, f"models are {models.state}")
        if models.planner is None:
            raise HTTPException(404, "this server has no planner")
        try:
            return models.planner.plan_prompt(prompt[:500], max(0, min(int(retries), 8)))
        except ValueError as e:
            raise HTTPException(422, str(e)) from None

    @app.get("/health")
    def health() -> dict:
        return models.health()

    @app.post("/control")
    def control(body: dict = Body(...)) -> dict:  # noqa: B008
        action = body.get("action")
        try:
            if action == "start":
                models.start()
            elif action == "stop":
                models.stop()
            else:
                raise HTTPException(400, "action must be start or stop")
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from None
        return models.health()

    @app.post("/render")
    def render_ep(body: dict = Body(...)) -> dict:  # noqa: B008
        t0 = time.perf_counter()
        A = render(str(body.get("recipe", "")), body.get("seed"))
        return {"poses": np.round(A, 5).tolist(), "fps": FPS, "ms": round((time.perf_counter() - t0) * 1000)}

    @app.post("/plan")
    def plan_ep(body: dict = Body(...)) -> dict:  # noqa: B008
        t0 = time.perf_counter()
        idea, recipe = plan(str(body.get("prompt", "")), body.get("retries", 2))
        return {"idea": idea, "recipe": recipe, "ms": round((time.perf_counter() - t0) * 1000)}

    @app.post("/generate-dense")
    def dense(body: dict = Body(...)) -> dict:  # noqa: B008
        prompt, seed = str(body.get("prompt", "")), int(body.get("seed", 0))
        t0 = time.perf_counter()
        idea, recipe = plan(prompt, body.get("retries", 2))
        t1 = time.perf_counter()
        A = render(recipe, seed)
        t2 = time.perf_counter()
        return {"prompt": prompt, "idea": idea, "recipe": recipe, "moves": [to_move(A, prompt)],
                "durations_s": [round(len(A) / FPS, 2)],
                "timing_ms": {"planner": round((t1 - t0) * 1000), "generator": round((t2 - t1) * 1000),
                              "total": round((t2 - t0) * 1000)}}

    return app


def _advertise(port: int):
    from zeroconf import ServiceInfo, Zeroconf

    from reachy_motion.taigi_node import lan_ip

    host = socket.gethostname().split(".")[0]
    info = ServiceInfo(SERVICE_TYPE, f"Reachy Motion on {host}.{SERVICE_TYPE}", addresses=[socket.inet_aton(lan_ip())],
                       port=port, properties={"v": "2"}, server=f"{host}.local.")
    zc = Zeroconf()
    zc.register_service(info, allow_name_change=True)  # a second Mac (or a stale record) gets "… (2)"
    return zc, info


LABEL = "com.reachy-motion.server"


def login_item(install: bool, port: int) -> None:
    """Install / remove a per-user LaunchAgent that keeps ``reachy-motion-server --standby`` running."""
    import os
    import plistlib
    import subprocess
    import sys
    from pathlib import Path

    plist = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", f"{domain}/{LABEL}"], capture_output=True, check=False)  # ok if not loaded
    if not install:
        plist.unlink(missing_ok=True)
        print(f"  removed {plist}")
        return
    log = Path.home() / "Library" / "Logs" / "reachy-motion-server.log"
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_bytes(plistlib.dumps({
        "Label": LABEL,
        "ProgramArguments": [sys.executable, "-m", "reachy_motion.motion_server", "--standby", "--port", str(port)],
        "RunAtLoad": True, "KeepAlive": True, "ProcessType": "Background",
        "StandardOutPath": str(log), "StandardErrorPath": str(log),
    }))
    subprocess.run(["launchctl", "bootstrap", domain, str(plist)], check=True)
    print(f"  installed {plist} (standby; logs: {log}). Remove with --uninstall-login-item.")


def main() -> None:
    p = argparse.ArgumentParser(prog="reachy-motion-server", description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--planner", default="0.8b", help=f"fine-tuned planner: {' | '.join(LOCAL_PLANNERS)} or a HF repo id")
    p.add_argument("--no-planner", action="store_true", help="serve the generator only")
    p.add_argument("--no-mdns", action="store_true")
    p.add_argument("--standby", action="store_true", help="start with no models loaded (Start from the app page)")
    p.add_argument("--install-login-item", action="store_true",
                   help="macOS: start this server in standby at every login (a LaunchAgent), then exit")
    p.add_argument("--uninstall-login-item", action="store_true", help="macOS: remove that LaunchAgent, then exit")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.install_login_item or a.uninstall_login_item:
        login_item(install=a.install_login_item, port=a.port)
        return

    import httpx

    try:  # one per machine: if it is already running, say so instead of failing on the port / mDNS name
        h = httpx.get(f"http://127.0.0.1:{a.port}/health", timeout=1).json()
        print(f"  reachy-motion-server is already running on port {a.port} (planner: {h.get('planner') or '—'}). "
              "Nothing to do.")
        return
    except (httpx.HTTPError, ValueError):
        pass

    if not a.no_planner and not can_run_mlx():
        print("  ⚠ the planner needs Apple Silicon with mlx-lm (pip install 'reachy-motion[local]'); "
              "serving the generator only")
    models = Models(None if a.no_planner else a.planner)
    if a.standby:
        print("  standby: nothing loaded until Start is pressed on the Reachy-Motion app page")
    else:
        print("  loading the generator" + ("" if a.no_planner else f" and the {a.planner} planner")
              + " (first run downloads them)…")
        models.start()

    zc = info = None
    if not a.no_mdns:
        zc, info = _advertise(a.port)
    import uvicorn

    print(f"\n  Reachy motion server on http://{a.host}:{a.port}  (mDNS {SERVICE_TYPE})")
    print("  Ctrl+C to stop.\n")
    try:
        uvicorn.run(make_app(models), host=a.host, port=a.port, log_level="warning")
    finally:
        if zc is not None:
            zc.unregister_service(info)
            zc.close()


if __name__ == "__main__":
    main()
