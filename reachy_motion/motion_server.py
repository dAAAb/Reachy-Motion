"""``reachy-motion-server``: run the flow-matching generator (and Binh Pham's fine-tuned planner) on a Mac for Reachy.

The robot's Raspberry Pi CM4 is too slow to run the generator while it talks (~0.2 s per gesture on an M-series
Mac, seconds on the CM4), and the planner needs Apple Silicon (MLX). This server runs both on the Mac, and
advertises ``_reachy-motion._tcp.local.`` over mDNS so the robot finds it when its motion engine is ``flow`` or
``local`` (see ``motion/backends.py``).

    reachy-motion-server                   # generator + 0.8B planner on port 18110
    reachy-motion-server --planner 4b      # the larger planner (~9 GB download, better motions)
    reachy-motion-server --no-planner      # generator only (for the "flow" engine)

API (JSON): ``GET /health`` · ``POST /render {recipe, seed?}`` -> ``{poses: (T, 9), fps}`` ·
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


def make_app(gen: FlowRenderer, planner: LocalPlanner | None):
    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(title="reachy-motion-server", docs_url=None, redoc_url=None)

    def render(recipe: str, seed) -> np.ndarray:
        if not gen.load():
            raise HTTPException(503, gen.error or "generator unavailable")
        try:
            return gen.render(recipe, None if seed is None else int(seed))
        except RecipeError as e:
            raise HTTPException(422, str(e)) from None

    def plan(prompt: str, retries: int) -> tuple[str, str]:
        if planner is None:
            raise HTTPException(404, "this server has no planner (started with --no-planner)")
        if not planner.load():
            raise HTTPException(503, planner.error or "planner unavailable")
        try:
            return planner.plan_prompt(prompt[:500], max(0, min(int(retries), 8)))
        except ValueError as e:
            raise HTTPException(422, str(e)) from None

    @app.get("/health")
    def health() -> dict:
        return {"generator": gen.ready, "planner": planner.name.split(":", 1)[1] if planner else None,
                "planner_ready": bool(planner and planner.ready), "v": 1}

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


def _advertise(port: int, planner: str | None):
    from zeroconf import ServiceInfo, Zeroconf

    from reachy_motion.taigi_node import lan_ip

    host = socket.gethostname().split(".")[0]
    info = ServiceInfo(SERVICE_TYPE, f"Reachy Motion on {host}.{SERVICE_TYPE}", addresses=[socket.inet_aton(lan_ip())],
                       port=port, properties={"planner": planner or "", "v": "1"}, server=f"{host}.local.")
    zc = Zeroconf()
    zc.register_service(info, allow_name_change=True)  # a second Mac (or a stale record) gets "… (2)"
    return zc, info


def main() -> None:
    p = argparse.ArgumentParser(prog="reachy-motion-server", description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--planner", default="0.8b", help=f"fine-tuned planner: {' | '.join(LOCAL_PLANNERS)} or a HF repo id")
    p.add_argument("--no-planner", action="store_true", help="serve the generator only")
    p.add_argument("--no-mdns", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import httpx

    try:  # one per machine: if it is already running, say so instead of failing on the port / mDNS name
        h = httpx.get(f"http://127.0.0.1:{a.port}/health", timeout=1).json()
        print(f"  reachy-motion-server is already running on port {a.port} (planner: {h.get('planner') or '—'}). "
              "Nothing to do.")
        return
    except (httpx.HTTPError, ValueError):
        pass

    gen = FlowRenderer()
    planner = None
    if not a.no_planner:
        if can_run_mlx():
            planner = LocalPlanner(a.planner)
        else:
            print("  ⚠ the planner needs Apple Silicon with mlx-lm (pip install 'reachy-motion[local]'); "
                  "serving the generator only")
    print("  loading the generator" + (f" and the {a.planner} planner" if planner else "") + " (first run downloads them)…")
    threads = [threading.Thread(target=gen.load)] + ([threading.Thread(target=planner.load)] if planner else [])
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if not gen.ready:
        raise SystemExit(f"generator failed to load: {gen.error}")
    t0 = time.perf_counter()
    gen.render("go .4 e=10 p=-4 | hold .5")  # warm-up
    print(f"  generator ready ({(time.perf_counter() - t0) * 1000:.0f} ms per short gesture)")
    if planner is not None and not planner.ready:
        print(f"  ⚠ planner failed to load ({planner.error}); serving the generator only")
        planner = None

    zc = info = None
    if not a.no_mdns:
        zc, info = _advertise(a.port, planner.name.split(":", 1)[1] if planner else None)
    import uvicorn

    print(f"\n  Reachy motion server on http://{a.host}:{a.port}  (mDNS {SERVICE_TYPE})")
    print(f"    generator: flow-matching (numpy)   planner: {planner.repo if planner else '—'}")
    print("  Ctrl+C to stop.\n")
    try:
        uvicorn.run(make_app(gen, planner), host=a.host, port=a.port, log_level="warning")
    finally:
        if zc is not None:
            zc.unregister_service(info)
            zc.close()


if __name__ == "__main__":
    main()
