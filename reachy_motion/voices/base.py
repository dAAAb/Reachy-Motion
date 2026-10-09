"""Common shape of a voice mode: mic in, speaker out, speech text to the director."""

from __future__ import annotations

import abc
import logging
import threading
from collections.abc import Callable

from reachy_motion.audio_io import AudioIO
from reachy_motion.config import Settings
from reachy_motion.motion.director import Director

logger = logging.getLogger(__name__)


class VoiceMode(abc.ABC):
    """One conversation backend. ``run`` blocks until ``stop`` is set or the session ends."""

    name: str = "base"
    #: sample rate this mode produces audio at (the local speaker is opened at this rate)
    output_rate: int = 24000
    #: a planned gesture may start this long after its clause ended (text that arrives late needs more)
    late_grace_s: float = 0.4
    #: None = audio backend default (robot: full duplex, local: half duplex)
    default_half_duplex: bool | None = None

    def __init__(
        self, settings: Settings, audio: AudioIO, director: Director, on_event: Callable[[dict], None]
    ) -> None:
        self.settings, self.audio, self.director, self.on_event = settings, audio, director, on_event

    @abc.abstractmethod
    def run(self, stop: threading.Event) -> None: ...

    #: spoken once the session is live (e.g. "switched to Taiwanese"); set by the engine before run()
    announce: str | None = None

    def say(self, text: str) -> None:
        """Make the robot tell the person something (a system notice) in this mode's voice. Best effort."""
        self.status(f"(notice) {text}")

    def answer_from_sight(self, question: str, seen: str) -> None:
        """The camera saw ``seen``: let this voice answer ``question`` with it (default: relay as a notice)."""
        self.say(f"你剛剛用攝影機看了一下，看到的是：{seen}。請用這個直接回答使用者剛才的問題「{question}」，"
                 "像是你親眼看到一樣。")

    # helpers ---------------------------------------------------------------------------------------------------------
    def barge_in(self) -> None:
        """The person talked over the robot: drop queued audio and pending gestures."""
        self.audio.speaker.flush()
        self.director.interrupt()

    def status(self, text: str) -> None:
        logger.info("[%s] %s", self.name, text)
        self.on_event({"type": "status", "mode": self.name, "text": text})
