"""Reachy-Motion: talk with Reachy Mini; it moves with what it says.

Two ways to run it:

* as a **Reachy Mini app** (installed on the robot, started from the dashboard) — ``ReachyMotion.run``;
* from a computer with the **CLI** — ``reachy-motion --mode taigi`` (robot over the network, or ``--no-robot`` to
  watch the motion in the browser preview only).

The settings page (``http://<host>:8042``) switches modes and shows the transcript and every gesture.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
from pathlib import Path

from reachy_mini import ReachyMini, ReachyMiniApp

from reachy_motion.config import Settings
from reachy_motion.engine import MODES, Engine

logger = logging.getLogger("reachy_motion")
STATIC = Path(__file__).parent / "static"


def add_routes(app, engine: Engine) -> None:
    """JSON API used by static/main.js (also mounted on the daemon-provided settings app)."""
    from fastapi import Body

    @app.get("/api/state")
    def state(after: int = 0) -> dict:
        return {
            "mode": engine.running_mode,
            "modes": list(MODES),
            "settings": engine.settings.public(),
            "pose": engine.current_pose(),
            "events": engine.events_since(after),
        }

    @app.post("/api/start")
    def start(body: dict = Body(default={})) -> dict:  # noqa: B008
        engine.start(body.get("mode") or engine.settings.mode)
        return {"ok": True, "mode": engine.running_mode}

    @app.post("/api/stop")
    def stop() -> dict:
        engine.stop()
        return {"ok": True}


class ReachyMotion(ReachyMiniApp):
    """Dashboard-managed app (runs on the robot; uses the robot's mic and speaker)."""

    custom_app_url: str | None = "http://0.0.0.0:8042"

    def __init__(self, *a, **kw) -> None:
        from reachy_motion.config import _load_dotenv

        _load_dotenv()  # before reading REACHY_MOTION_AUDIO below
        if os.environ.get("REACHY_MOTION_AUDIO", "auto") == "local":
            self.request_media_backend = "no_media"  # free the robot's audio devices, use this machine's
        super().__init__(*a, **kw)

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        settings = Settings.from_env()
        engine = Engine(settings, robot=reachy_mini)
        if self.settings_app is not None:
            add_routes(self.settings_app, engine)
        engine.start_animation()
        engine.start(settings.mode)
        try:
            stop_event.wait()
        finally:
            engine.close()


def cli() -> None:
    p = argparse.ArgumentParser(prog="reachy-motion", description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=MODES, help="voice mode (default: $REACHY_MOTION_MODE or gpt-live)")
    p.add_argument("--audio", choices=("auto", "robot", "local"), help="whose mic/speaker to use")
    p.add_argument("--half-duplex", action=argparse.BooleanOptionalAction, default=None,
                   help="mute the mic while the robot talks (default: on for local audio)")
    p.add_argument("--no-robot", action="store_true", help="don't connect to a robot (browser preview only)")
    p.add_argument("--host", default=None, help="robot hostname (default: auto-detect)")
    p.add_argument("--sim", action="store_true", help="connect to a local simulated daemon (reachy-mini-daemon --sim)")
    p.add_argument("--port", type=int, default=8042, help="settings page port")
    p.add_argument("--no-reflexes", action="store_true", help="disable instant cue-word gestures")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "websockets", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    audio = a.audio or ("local" if (a.no_robot or a.sim) else None)
    settings = Settings.from_env(mode=a.mode, audio=audio, half_duplex=a.half_duplex)
    if a.no_reflexes:
        settings.reflexes = False

    robot = None
    if not a.no_robot:
        kw: dict = {"media_backend": "no_media" if settings.audio == "local" else "default"}
        if a.sim:
            kw["connection_mode"] = "localhost_only"
        elif a.host:
            kw["host"] = a.host
        robot = ReachyMini(**kw)

    engine = Engine(settings, robot=robot)
    server = _serve(engine, a.port)
    engine.start_animation()
    engine.start(settings.mode)
    print(f"\n  Reachy-Motion · mode={settings.mode} · open http://127.0.0.1:{a.port}  (Ctrl+C to quit)\n")
    # Our own handlers: an imported library (uvicorn / gi / zeroconf) swallows the default SIGINT -> KeyboardInterrupt.
    quit_evt = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: quit_evt.set())
    try:
        while not quit_evt.wait(0.5):
            pass
    finally:
        engine.close()
        engine.rest_robot()
        server.should_exit = True
        if robot is not None:
            robot.__exit__(None, None, None)


def _serve(engine: Engine, port: int):
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Reachy-Motion")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    add_routes(app, engine)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, name="web", daemon=True).start()
    return server


if __name__ == "__main__":  # the daemon runs apps with `python -m reachy_motion.main`
    app = ReachyMotion()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
