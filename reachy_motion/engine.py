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
        self.voice = None  # the running VoiceMode
        self.audio: AudioIO | None = None
        self.gaze_on = settings.gaze
        self._gaze_w: float | None = None
        self._gaze_stop = threading.Event()
        self._switching = threading.Lock()
        self._recent: collections.deque[tuple[str, str]] = collections.deque(maxlen=6)
        self.body_agent = None
        if settings.openai_api_key:
            from reachy_motion.intent import BodyAgent

            self.body_agent = BodyAgent(settings.planner_model, settings.openai_api_key)
        self.web = None
        if settings.openai_api_key:
            from reachy_motion.web import WebLookup

            self.web = WebLookup(settings.vision_model, settings.openai_api_key, location=settings.location)
        self.eyes = None
        if settings.openai_api_key and robot is not None and getattr(robot, "media", None) is not None:
            from reachy_motion.vision import Eyes

            self.eyes = Eyes(robot, settings.vision_model, settings.openai_api_key)

    # -- events (for the web UI / logs) ----------------------------------------------------------------------------
    def emit(self, ev: dict) -> None:
        ev = {"id": next(self._ids), "t": round(time.time(), 2), **ev}
        self.events.append(ev)
        if ev["type"] in ("user", "robot"):
            self._recent.append(("person" if ev["type"] == "user" else "robot", (ev.get("text") or "")[:200]))
        if ev["type"] == "user" and ev.get("text", "").strip():
            threading.Thread(target=self._decide, args=(ev["text"],), name="body-agent", daemon=True).start()
        if ev["type"] in ("gesture", "user", "robot", "status", "timing"):
            logger.info("%s", {k: v for k, v in ev.items() if k not in ("id", "t")})

    def events_since(self, after: int) -> list[dict]:
        return [e for e in list(self.events) if e["id"] > after]  # copy first: voice threads append concurrently

    def current_pose(self) -> list[float] | None:
        with self._pose_lock:
            return None if self.pose is None else [round(float(v), 4) for v in self.pose]

    # -- lifecycle -------------------------------------------------------------------------------------------------
    def start_animation(self) -> None:
        self.prepare_robot()
        self.restore_volume()
        self.animator.start()
        if self.robot is not None:
            threading.Thread(target=self._gaze_loop, name="gaze", daemon=True).start()

    # -- gaze: daemon-side face tracking, blended under our gestures ------------------------------------------------
    def _gaze_loop(self) -> None:
        """Look at the person: strong while listening/idle, lighter while a gesture plays so it shows through."""
        while not self._gaze_stop.wait(0.2):
            if not self.gaze_on:
                target = 0.0
            else:
                target = (self.settings.gaze_weight_idle if self.animator.playing is None
                          else self.settings.gaze_weight_gesture)
            cur = self._gaze_w
            if cur is not None and abs(cur - target) < 0.02:
                continue
            nxt = target if cur is None else cur + max(-0.15, min(0.15, target - cur))  # ramp, no snaps
            try:
                if nxt <= 0.01:
                    self.robot.stop_head_tracking()
                else:
                    self.robot.start_head_tracking(weight=round(nxt, 2))
                self._gaze_w = nxt
            except Exception as e:  # noqa: BLE001 - older daemons have no tracking: give up quietly
                logger.info("head tracking unavailable: %s", e)
                return

    # -- spoken commands ----------------------------------------------------------------------------------------
    def _decide(self, utterance: str) -> None:
        """Ask the body agent (LLM + tools) what the utterance wants; regex rules only when there is no API key."""
        from reachy_motion.commands import parse_command

        cmds = []
        if self.body_agent is not None:
            state = {"voice_mode": self.running_mode, "looking_at_person": self.gaze_on}
            try:
                cmds = self.body_agent.decide(utterance, state, list(self._recent)[:-1])
            except Exception as e:  # noqa: BLE001 - network hiccup: fall back to the rules for this utterance
                logger.warning("body agent failed (%s); using keyword rules", e)
                cmds = [c for c in [parse_command(utterance)] if c]
        else:
            cmds = [c for c in [parse_command(utterance)] if c]
        # NOTICE: the small intent model keeps reading "聽不懂" (don't understand) as "can't hear" (louder).
        # Guard: no volume change when the person says they don't understand and never mentions loudness.
        import re as _re

        if _re.search(r"聽不懂|聽不太懂|听不懂|听不太懂|不懂你", utterance) and not _re.search(
                r"聽不清|听不清|小聲|小声|大聲|大声|音量|louder|volume", utterance):
            cmds = [c for c in cmds if c.kind != "volume"]
        for cmd in cmds:
            self.handle_command(cmd)

    def handle_command(self, cmd) -> None:
        self.emit({"type": "command", "kind": cmd.kind, "arg": cmd.arg, "text": cmd.text})
        try:
            if cmd.kind == "volume":
                self.set_volume(cmd.arg)
            elif cmd.kind == "gaze":
                self.gaze_on = bool(cmd.arg)
                self.emit({"type": "status", "text": "looking at you" if self.gaze_on else "stopped looking"})
            elif cmd.kind == "mode":
                self.switch_mode(str(cmd.arg))
            elif cmd.kind == "look":
                self.look_and_answer(str(cmd.arg))
            elif cmd.kind == "web":
                self.web_and_answer(str(cmd.arg))
        except Exception as e:  # noqa: BLE001
            logger.exception("command failed")
            self.emit({"type": "status", "text": f"command failed: {e}"})

    def look_and_answer(self, question: str) -> None:
        if self.eyes is None or self.voice is None:
            return
        self.gaze_on = True  # look at the person while looking
        seen = self.eyes.describe(question)
        if not seen:
            self.voice.say("我的攝影機現在拿不到畫面，暫時看不到。")
            return
        self.emit({"type": "seen", "text": seen, "question": question})
        self.voice.answer_from_sight(question, seen)

    def web_and_answer(self, query: str) -> None:
        if self.web is None or self.voice is None:
            return
        # a question split across two transcript fragments triggers two lookups: answer it once
        now = time.monotonic()
        key = set("".join(ch for ch in query if ch.isalnum()))
        last_key, last_t = getattr(self, "_last_web", (set(), 0.0))
        if now - last_t < 20 and key and len(key & last_key) / len(key | last_key) > 0.5:
            return
        self._last_web = (key, now)
        found = self.web.answer(query)
        if not found:
            self.voice.say("我剛剛想上網查，但是現在查不到，等一下再試試看。")
            return
        self.emit({"type": "found", "text": found, "question": query})
        self.voice.answer_from_web(query, found)

    def _daemon_url(self) -> str | None:
        return getattr(self.robot, "_daemon_http_url", None) if self.robot is not None else None

    def set_volume(self, arg) -> None:
        import httpx

        base = self._daemon_url()
        if base is None:  # local speaker: software gain
            if self.audio is not None:
                g = self.audio.speaker.gain
                g = {"+": g * 1.4, "-": g / 1.4}.get(arg, g) if isinstance(arg, str) else arg / 70.0
                self.audio.speaker.gain = max(0.1, min(3.0, g))
                self.emit({"type": "status", "text": f"speaker gain {self.audio.speaker.gain:.2f}"})
            return
        cur = httpx.get(base + "/api/volume/current", timeout=5).json().get("volume", 60)
        vol = {"+": cur + 15, "-": cur - 15}.get(arg, arg) if isinstance(arg, str) else arg
        vol = int(max(10, min(100, vol)))  # never fully mute by voice: you couldn't hear the answer
        httpx.post(base + "/api/volume/set", json={"volume": vol}, timeout=5).raise_for_status()
        self.emit({"type": "status", "text": f"volume {cur} → {vol}"})
        _save_state({"volume": vol})  # the daemon resets volume on restart: remember the person's choice

    def restore_volume(self) -> None:
        base, vol = self._daemon_url(), _load_state().get("volume")
        if base is None or vol is None:
            return
        try:
            import httpx

            httpx.post(base + "/api/volume/set", json={"volume": int(vol)}, timeout=5).raise_for_status()
            self.emit({"type": "status", "text": f"volume restored to {vol}"})
        except Exception as e:  # noqa: BLE001
            logger.info("could not restore volume: %s", e)

    def switch_mode(self, target: str) -> None:
        """Switch voice mode on request; for Taiwanese, first make sure the services are reachable."""
        if not self._switching.acquire(blocking=False):
            return
        try:
            current = self.running_mode
            if target == "default" and current != "taigi":
                return  # "back to Mandarin" only means something while speaking Taiwanese
            if target == "default":
                target = self.settings.mode if self.settings.mode != "taigi" else "gpt-live"
                if current == target:
                    target = "gpt-live" if current != "gpt-live" else "elevenlabs"
            if target == current:
                return
            if target == "taigi":
                from reachy_motion.discovery import probe_taigi

                s = self.settings
                self.emit({"type": "status", "text": "looking for Taiwanese speech services on the LAN…"})
                p = probe_taigi(s.taigi_asr_url, s.taigi_llm_url, s.taigi_tts_url, s.taigi_llm_model)
                if not p.ok:
                    names = {"asr": "台語語音辨識", "llm": "台語語言模型", "tts": "台語語音合成"}
                    lost = "、".join(names[m] for m in p.missing)
                    self.emit({"type": "status", "text": f"Taiwanese unavailable, missing: {', '.join(p.missing)}"})
                    if self.voice is not None:
                        self.voice.say(f"現在區網上找不到{lost}的服務，所以暫時沒辦法切換成台語，先維持目前的模式。")
                    return
                s.taigi_asr_url, s.taigi_llm_url, s.taigi_tts_url = p.asr_url, p.llm_url, p.tts_url
                if p.llm_model:
                    s.taigi_llm_model = p.llm_model
                self.emit({"type": "status", "text": f"Taiwanese services found ({p.source})"})
            # let the current voice finish its "OK, switching" sentence before hanging up
            t0 = time.monotonic()
            while self.audio is not None and self.audio.speaker.busy(0.3) and time.monotonic() - t0 < 8:
                time.sleep(0.2)
            announce = {
                "taigi": "我轉做台語矣，你欲佮我講啥物？",
                "gpt-live": "已經切換到 GPT-Live 模式了，用一句話跟使用者打招呼。",
                "elevenlabs": None,  # the agent greets with its own first message
            }.get(target)
            self.start(target, announce=announce)
        finally:
            self._switching.release()

    def prepare_robot(self) -> None:
        """Wake the robot before streaming poses.

        A Wireless unit boots asleep (motors disabled). ``enable_motors()`` pins targets to the present pose, and
        streaming ``set_target`` immediately after it can crash the daemon (reachy_mini#1430), so ease to the
        neutral pose with ``goto_target`` first and only then start the 60 Hz animator.
        """
        if self.robot is None:
            return
        try:
            import numpy as np

            self.robot.enable_motors()
            self.robot.goto_target(head=np.eye(4), antennas=[0.0, 0.0], body_yaw=0.0, duration=1.2)
            self.emit({"type": "status", "text": "robot awake (motors enabled)"})
        except Exception as e:  # noqa: BLE001 - keep running: the animator will still try set_target
            logger.warning("could not wake the robot: %s", e)
            self.emit({"type": "status", "text": f"could not wake the robot: {e}"})

    def rest_robot(self) -> None:
        """Put the robot back to its sleep pose (CLI exit; the daemon handles this for dashboard apps)."""
        if self.robot is None:
            return
        try:
            self.robot.goto_sleep()
        except Exception as e:  # noqa: BLE001
            logger.warning("goto_sleep failed: %s", e)

    def start(self, mode: str | None = None, announce: str | None = None) -> None:
        with self._lifecycle:
            self._stop_locked()
            self._start_locked(mode or self.settings.mode, announce)

    def _start_locked(self, mode: str, announce: str | None = None) -> None:
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
        voice.announce = announce
        self.voice, self.audio = voice, audio
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
            self._session.join(timeout=12)
            if self._session.is_alive():
                logger.error("voice session did not stop within 12 s; continuing anyway")
        self._session = None
        self.running_mode = None

    def close(self) -> None:
        self._gaze_stop.set()
        if self.robot is not None and self._gaze_w:
            try:
                self.robot.stop_head_tracking()
            except Exception:  # noqa: BLE001
                pass
        self.stop()
        self.animator.close()


_STATE = __import__("pathlib").Path.home() / ".config" / "reachy_motion" / "state.json"


def _load_state() -> dict:
    import json

    try:
        return json.loads(_STATE.read_text())
    except (OSError, ValueError):
        return {}


def _save_state(update: dict) -> None:
    import json

    try:
        _STATE.parent.mkdir(parents=True, exist_ok=True)
        _STATE.write_text(json.dumps({**_load_state(), **update}))
    except OSError as e:
        logger.info("could not save state: %s", e)


def _robot_has_audio(robot) -> bool:
    try:
        return robot.media is not None and robot.media.get_output_audio_samplerate() > 0
    except Exception:  # noqa: BLE001 - no_media backend / remote robot without WebRTC audio
        return False
