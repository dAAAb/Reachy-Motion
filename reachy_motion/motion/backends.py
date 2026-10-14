"""The three motion engines, and where each part runs.

=========  ==========================================  ===========================================================
engine     who writes the recipe                       who turns it into motion
=========  ==========================================  ===========================================================
recipe     cloud LLM (``GesturePlanner``)              ``recipe.posture_to_poses``: eased keys + band-limited noise
flow       cloud LLM (``GesturePlanner``)              Binh Pham's flow-matching generator (``flow.py``)
local      Binh Pham's fine-tuned planner (MLX)        Binh Pham's flow-matching generator
=========  ==========================================  ===========================================================

The generator is small (numpy, ~0.2 s per gesture on an M-series Mac) but too slow for the robot's Raspberry Pi
CM4 to keep up with speech, and the fine-tuned planner needs Apple Silicon (MLX). So both can also run on a Mac on
the LAN: ``reachy-motion-server`` serves them and advertises ``_reachy-motion._tcp`` over mDNS; the robot uses it
when it finds one (``REACHY_MOTION_MOTION_SERVER`` = ``auto`` | ``http://host:port`` | ``off``).
"""

from __future__ import annotations

import logging
import platform
import sys
import threading
import time

import numpy as np

from reachy_motion.motion.planner import TALK_FALLBACK, Gesture, reflex
from reachy_motion.motion.recipe import check

logger = logging.getLogger(__name__)

ENGINES = ("recipe", "flow", "local")
SERVICE_TYPE = "_reachy-motion._tcp.local."
DEFAULT_PORT = 18110
LOCAL_PLANNERS = {  # Binh Pham's fine-tuned planners (Apache-2.0); the 0.8B is ~1.7 GB, the 4B ~9 GB
    "0.8b": "binhpham/reachy-mini-motion-planner-0.8b",
    "4b": "binhpham/reachy-mini-motion-planner-4b",
}


def fast_machine() -> bool:
    """Can this machine run the generator in-process fast enough for live speech? (Not the robot's CM4.)"""
    return sys.platform == "darwin" or platform.machine().lower() in ("x86_64", "amd64")


def can_run_mlx() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True


def planner_prompt(line: str | None, heard: str | None = None, tone: str | None = None) -> str:
    """The fine-tuned planners were trained on ``"word. one sentence of context."`` prompts."""
    if line is None:  # anticipatory reaction while the person talks
        return f"listening. Someone says to you: {heard}"
    return f"{tone}. {line}" if tone else line


# -- local fine-tuned planner (MLX) ----------------------------------------------------------------------------------
class LocalPlanner:
    """Binh Pham's fine-tuned planner on Apple Silicon. Same interface as ``GesturePlanner`` (plan / react)."""

    def __init__(self, size: str = "0.8b") -> None:
        self.repo = LOCAL_PLANNERS.get(size, size)
        self.name = f"local:{size}"
        self._model = self._tok = None
        self._loading = threading.Lock()
        self._lock = threading.Lock()
        self.error: str | None = None

    def start_loading(self) -> None:
        if self._model is None and self.error is None and not self._loading.locked():
            threading.Thread(target=self.load, name="planner-load", daemon=True).start()

    def load(self) -> bool:
        with self._loading:
            if self._model is None and self.error is None:
                try:
                    from huggingface_hub import snapshot_download
                    from mlx_lm import load

                    t0 = time.perf_counter()
                    root = snapshot_download(self.repo, allow_patterns=["planner/*"])  # once: ~1.7 GB for the 0.8B
                    self._model, self._tok = load(f"{root}/planner")
                    self._warm()
                    logger.info("local planner %s loaded in %.1f s", self.repo, time.perf_counter() - t0)
                except Exception as e:  # noqa: BLE001
                    self.error = f"{type(e).__name__}: {e}"
                    logger.warning("local planner unavailable: %s", self.error)
        return self._model is not None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _warm(self) -> None:
        """One throw-away generation: the first one compiles MLX kernels (~1 s slower than the rest)."""
        from mlx_lm import generate

        from reachy_motion.motion.prompt import STUDENT_SYSTEM

        msgs = [{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": "warm-up. You stretch."}]
        text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        generate(self._model, self._tok, text, max_tokens=8)

    def write(self, prompt: str, temperature: float = 0.0, max_tokens: int = 300) -> tuple[str, str | None]:
        """prompt -> (idea, valid recipe or None)."""
        import json
        import re

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        from reachy_motion.motion.prompt import STUDENT_SYSTEM

        if self._model is None:  # never block the speech path on a download
            self.start_loading()
            raise RuntimeError(self.error or "local planner still loading")
        msgs = [{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": prompt}]
        text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        with self._lock:  # one MLX generation at a time
            out = generate(self._model, self._tok, text, max_tokens=max_tokens, sampler=make_sampler(temp=temperature))
        m = re.search(r"\{.*\}", out.split("</think>")[-1], re.DOTALL)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            d = {}
        rec = d.get("recipe")
        return d.get("idea", ""), (rec if rec and check(rec) is None else None)

    def plan_prompt(self, prompt: str, retries: int = 2) -> tuple[str, str]:
        """Greedy first, then samples at temperature 0.7 (as the reference server). Raises ValueError if all fail."""
        for k in range(retries + 1):
            idea, rec = self.write(prompt, 0.0 if k == 0 else 0.7)
            if rec:
                return idea, rec
        raise ValueError("planner did not produce a valid recipe")

    def plan(self, line: str, heard: str | None = None, retries: int = 2, tone: str | None = None) -> Gesture:
        t0 = time.perf_counter()
        try:
            idea, rec = self.plan_prompt(planner_prompt(line, heard, tone), retries)
            return Gesture(rec, idea, "planner", line, (time.perf_counter() - t0) * 1000, {"planner": self.name})
        except Exception as e:  # noqa: BLE001
            logger.warning("local planner failed: %s", e)
        fb = reflex(line) or Gesture(TALK_FALLBACK, "talking", "fallback", line)
        fb.source, fb.latency_ms = "fallback", (time.perf_counter() - t0) * 1000
        return fb

    def react(self, heard: str) -> Gesture | None:
        t0 = time.perf_counter()
        try:
            idea, rec = self.plan_prompt(planner_prompt(None, heard), retries=0)
        except Exception:  # noqa: BLE001
            return None
        return Gesture(rec, idea, "reaction", heard, (time.perf_counter() - t0) * 1000, {"planner": self.name})


# -- a motion server on the LAN --------------------------------------------------------------------------------------
def discover(timeout_s: float = 2.5) -> str | None:
    """Base URL of a ``reachy-motion-server`` advertised over mDNS, or None."""
    try:
        import socket

        from zeroconf import ServiceBrowser, Zeroconf
    except ImportError:
        return None
    found: list[str] = []

    class Listener:
        def add_service(self, zc, type_, name):
            info = zc.get_service_info(type_, name, timeout=1500)
            if info and info.addresses:
                found.append(f"http://{socket.inet_ntoa(info.addresses[0])}:{info.port}")

        def update_service(self, *a):
            pass

        def remove_service(self, *a):
            pass

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, SERVICE_TYPE, Listener())
        deadline = time.monotonic() + timeout_s
        while not found and time.monotonic() < deadline:
            time.sleep(0.1)
    finally:
        zc.close()
    return found[0] if found else None


class RemoteMotion:
    """Client for ``reachy-motion-server``: ``render`` (recipe -> poses) and, if it has a planner, plan / react."""

    def __init__(self, base: str, timeout_s: float = 6.0) -> None:
        import httpx

        self.base = base.rstrip("/")
        self.http = httpx.Client(timeout=timeout_s)
        self.info = self.http.get(self.base + "/health", timeout=3).raise_for_status().json()
        self.name = f"server:{self.info.get('planner') or 'generator'}@{self.base.split('//')[-1]}"

    @property
    def has_planner(self) -> bool:
        return bool(self.info.get("planner"))

    def render(self, recipe: str, seed: int | None = None) -> np.ndarray:
        r = self.http.post(self.base + "/render", json={"recipe": recipe, "seed": seed})
        r.raise_for_status()
        return np.asarray(r.json()["poses"], dtype=np.float64)

    def _plan(self, prompt: str, retries: int) -> tuple[str, str]:
        r = self.http.post(self.base + "/plan", json={"prompt": prompt, "retries": retries})
        r.raise_for_status()
        d = r.json()
        return d["idea"], d["recipe"]

    def plan(self, line: str, heard: str | None = None, retries: int = 2, tone: str | None = None) -> Gesture:
        t0 = time.perf_counter()
        try:
            idea, rec = self._plan(planner_prompt(line, heard, tone), retries)
            return Gesture(rec, idea, "planner", line, (time.perf_counter() - t0) * 1000, {"planner": self.name})
        except Exception as e:  # noqa: BLE001
            logger.warning("remote planner failed: %s", e)
        fb = reflex(line) or Gesture(TALK_FALLBACK, "talking", "fallback", line)
        fb.source, fb.latency_ms = "fallback", (time.perf_counter() - t0) * 1000
        return fb

    def react(self, heard: str) -> Gesture | None:
        t0 = time.perf_counter()
        try:
            idea, rec = self._plan(planner_prompt(None, heard), 0)
        except Exception:  # noqa: BLE001
            return None
        return Gesture(rec, idea, "reaction", heard, (time.perf_counter() - t0) * 1000, {"planner": self.name})


# -- the renderer handed to the director -----------------------------------------------------------------------------
class Renderer:
    """Recipe -> poses through the generator (in-process or on the server), with a cache of pre-rendered variants
    for the fixed reflex recipes, so reflexes stay instant.

    If this machine renders too slowly to keep up with speech (the robot's CM4 without a motion server), planned
    gestures fall back to the built-in rendering and only the pre-rendered reflexes use the generator."""

    VARIANTS = 3
    MAX_S_PER_MOTION_S = 0.25  # render time per second of motion that still lands with the speech

    def __init__(self, backend, name: str) -> None:
        self.backend, self.name = backend, name  # backend has .render(recipe, seed) -> (T, 9)
        self._cache: dict[str, list[np.ndarray]] = {}
        self._turn = 0
        self.live = True
        self.note = ""

    def render(self, recipe: str) -> np.ndarray:
        if not self.live:
            raise RuntimeError("generator too slow on this machine for live gestures")
        return self.backend.render(recipe, None)

    def instant(self, recipe: str) -> np.ndarray | None:
        """A pre-rendered variant of a fixed recipe, or None if it isn't cached (yet)."""
        v = self._cache.get(recipe)
        if not v:
            return None
        self._turn += 1
        return v[self._turn % len(v)]

    def warm(self, recipes: list[str]) -> None:
        """Background: load the generator (if in-process), time it, then pre-render each fixed recipe a few times."""
        def run() -> None:
            load = getattr(self.backend, "load", None)
            if load is not None and not load():
                self.note = "generator unavailable: " + str(getattr(self.backend, "error", ""))
                return
            for i, rec in enumerate(recipes):
                try:
                    if i == 0:
                        self.backend.render(rec, 999)  # warm-up: first call pays one-off costs
                    t0 = time.perf_counter()
                    first = self.backend.render(rec, 1000)
                    per = (time.perf_counter() - t0) / max(0.5, len(first) / 25)
                    if i == 0 and per > self.MAX_S_PER_MOTION_S:
                        self.live = False
                        self.note = (f"generator too slow here ({per:.2f} s per motion second): learned motion for "
                                     "reflexes only — run reachy-motion-server on a Mac for every gesture")
                        logger.warning(self.note)
                    self._cache[rec] = [first] + [self.backend.render(rec, 1001 + k) for k in range(self.VARIANTS - 1)]
                except Exception as e:  # noqa: BLE001
                    logger.warning("could not pre-render a reflex (%s)", e)
                    return

        threading.Thread(target=run, name="reflex-warm", daemon=True).start()


def build(engine: str, cloud_planner, server: str = "auto", local_size: str = "0.8b"):
    """-> (planner, renderer | None, description). Falls back gracefully when a part is unavailable."""
    from reachy_motion.motion.flow import FlowRenderer

    if engine not in ENGINES:
        engine = "recipe"
    if engine == "recipe":
        return cloud_planner, None, "recipe (cloud planner + noise detail)"
    # Import what the loader threads use here, on the calling thread: openai checks sys.modules for a legacy httpx,
    # and a half-imported one (mid-import in a loader thread) crashes every OpenAI() created meanwhile.
    import httpx  # noqa: F401
    import huggingface_hub  # noqa: F401

    remote = None
    # in-process when this machine is fast enough (a Mac), else / when configured: a reachy-motion-server
    explicit = server.startswith("http")
    need = not fast_machine() or (engine == "local" and not can_run_mlx())
    if server != "off" and (explicit or need):
        url = server if server.startswith("http") else discover()
        if url:
            try:
                remote = RemoteMotion(url)
            except Exception as e:  # noqa: BLE001
                logger.warning("motion server %s unreachable: %s", url, e)
    gen = Renderer(remote, remote.name) if remote is not None else Renderer(FlowRenderer(), "generator:in-process")
    from reachy_motion.motion.director import LISTENING
    from reachy_motion.motion.planner import REFLEXES

    gen.warm([r for _, _, r in REFLEXES] + [LISTENING, TALK_FALLBACK])

    if engine == "flow":
        return cloud_planner, gen, f"flow ({'cloud planner' if cloud_planner else 'reflexes only'} + {gen.name})"
    if remote is not None and remote.has_planner:
        return remote, gen, f"local ({remote.name})"
    if can_run_mlx():
        lp = LocalPlanner(local_size)
        lp.start_loading()
        return lp, gen, f"local ({lp.name} + {gen.name})"
    return cloud_planner, gen, ("flow — no local planner here and no reachy-motion-server with one on the LAN "
                                f"({gen.name})")
