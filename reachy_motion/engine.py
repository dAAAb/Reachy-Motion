"""Wires one voice mode, the speaker clock, the director and the animator together; switchable at runtime."""

from __future__ import annotations

import collections
import itertools
import logging
import threading
import time

from reachy_motion.audio_io import AudioIO
from reachy_motion.config import Settings
from reachy_motion.motion.director import Director
from reachy_motion.motion.planner import GesturePlanner

logger = logging.getLogger(__name__)

MODES = ("gpt-live", "elevenlabs", "taigi")


def mode_class(name: str):
    if name == "gpt-live":
        from reachy_motion.voices.gpt_live import GptLiveMode

        return GptLiveMode
    if name == "elevenlabs":
        from reachy_motion.voices.elevenlabs import ElevenLabsMode

        return ElevenLabsMode
    if name == "taigi":
        from reachy_motion.voices.taigi import TaigiMode

        return TaigiMode
    raise ValueError(f"unknown mode {name!r}; choose from {MODES}")


class Engine:
    def __init__(self, settings: Settings, robot=None, audio_factory=None) -> None:
        """``audio_factory(mode_cls, on_chunk) -> AudioIO`` overrides mic/speaker creation (tests, recordings)."""
        from reachy_animation import Animator, to_target

        self.settings = settings
        self.robot = robot
        self.audio_factory = audio_factory
        self.animator = Animator(fps=60, speech_latency_s=0.08)
        self.pose = None
        self._pose_lock = threading.Lock()

        def on_pose(p) -> None:
            with self._pose_lock:
                self.pose = p
            if robot is not None:
                head, antennas, body_yaw = to_target(p)
                robot.set_target(head=head, antennas=antennas, body_yaw=body_yaw)

        self.animator.on_pose(on_pose)
        self.events: collections.deque[dict] = collections.deque(maxlen=300)
        self._ids = itertools.count(1)
        self.planner = GesturePlanner(settings.planner_model, settings.openai_api_key or None) if (
            settings.openai_api_key
        ) else None
        self._session: threading.Thread | None = None
        self._stop = threading.Event()
        self._lifecycle = threading.Lock()  # serialises start/stop (double-clicks in the UI hit two threads)
        self.running_mode: str | None = None

    # -- events (for the web UI / logs) ----------------------------------------------------------------------------
    def emit(self, ev: dict) -> None:
        ev = {"id": next(self._ids), "t": round(time.time(), 2), **ev}
        self.events.append(ev)
        if ev["type"] in ("gesture", "user", "robot", "status", "timing"):
            logger.info("%s", {k: v for k, v in ev.items() if k not in ("id", "t")})

    def events_since(self, after: int) -> list[dict]:
        return [e for e in list(self.events) if e["id"] > after]  # copy first: voice threads append concurrently

    def current_pose(self) -> list[float] | None:
        with self._pose_lock:
            return None if self.pose is None else [round(float(v), 4) for v in self.pose]

    # -- lifecycle -------------------------------------------------------------------------------------------------
    def start_animation(self) -> None:
        self.animator.start()

    def start(self, mode: str | None = None) -> None:
        with self._lifecycle:
            self._stop_locked()
            self._start_locked(mode or self.settings.mode)

    def _start_locked(self, mode: str) -> None:
        cls = mode_class(mode)
        audio_kind = self.settings.audio
        if audio_kind == "auto":
            audio_kind = "robot" if self.robot is not None and _robot_has_audio(self.robot) else "local"
        if self.audio_factory is not None:
            audio_kind = "custom"
            audio = self.audio_factory(cls, self.animator.feed_speech)
        else:
            half = self.settings.half_duplex if self.settings.half_duplex is not None else cls.default_half_duplex
            audio = AudioIO.create(
                audio_kind, self.robot, out_rate=cls.output_rate, half_duplex=half, on_chunk=self.animator.feed_speech,
            )
        audio.speaker.on_flush = self.animator.interrupt_speech
        director = Director(
            play=self.animator.play,
            planner=self.planner,
            time_of=audio.speaker.time_of,
            position=lambda: audio.speaker.position,
            on_event=self.emit,
            is_busy=lambda: self.animator.playing is not None,
            use_reflex=self.settings.reflexes,
            late_grace_s=cls.late_grace_s,
        )
        if self.planner is None:
            self.emit({"type": "status", "text": "no OPENAI_API_KEY: gesture planner off, reflex gestures only"})
        voice = cls(self.settings, audio, director, self.emit)
        self._stop = threading.Event()
        stop = self._stop

        def run() -> None:
            try:
                audio.speaker.start()
                voice.run(stop)
            except Exception as e:
                logger.exception("voice mode crashed")
                self.emit({"type": "status", "mode": mode, "text": f"crashed: {e}"})
            finally:
                director.close()
                try:
                    audio.speaker.stop()
                except Exception:
                    logger.debug("speaker stop failed", exc_info=True)
                if self._session is threading.current_thread():  # don't clobber a newer session's state
                    self.running_mode = None
                self.emit({"type": "status", "mode": mode, "text": "stopped"})

        self.running_mode = mode
        self.emit({"type": "status", "mode": mode, "text": f"starting ({audio_kind} audio, "
                   f"{'half' if audio.half_duplex else 'full'} duplex)"})
        self._session = threading.Thread(target=run, name=f"voice-{mode}", daemon=True)
        self._session.start()

    def stop(self) -> None:
        with self._lifecycle:
            self._stop_locked()

    def _stop_locked(self) -> None:
        """Stop the running session and wait for it to release the mic/speaker (a new one must not overlap it)."""
        if self._session is not None and self._session.is_alive():
            self._stop.set()
            self._session.join(timeout=20)
            if self._session.is_alive():
                logger.error("voice session did not stop within 20 s; starting anyway")
        self._session = None
        self.running_mode = None

    def close(self) -> None:
        self.stop()
        self.animator.close()


def _robot_has_audio(robot) -> bool:
    try:
        return robot.media is not None and robot.media.get_output_audio_samplerate() > 0
    except Exception:  # noqa: BLE001 - no_media backend / remote robot without WebRTC audio
        return False
