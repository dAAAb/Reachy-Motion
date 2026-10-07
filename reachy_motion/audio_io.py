"""Audio in/out with a playback clock.

Two backends with one interface:

* ``local`` — this computer's microphone and speaker (sounddevice). Use it when Reachy-Motion runs on a Mac next to
  the robot, e.g. the Taiwanese pipeline whose models live on the Mac.
* ``robot`` — Reachy Mini's own microphone array and speaker through ``reachy_mini.media`` (on the robot itself).

The speaker keeps a *playback clock*: every chunk written gets a stream position (seconds of audio queued since the
stream started), and ``time_of(pos)`` converts a position into the wall-clock (``time.monotonic``) moment it will
be heard. Gestures are scheduled against that clock, and every chunk is handed to ``on_chunk`` exactly when it is
queued so the animator's speech sway follows the same timeline.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

MicCallback = Callable[[np.ndarray, int], None]  # (mono float32 chunk, sample_rate)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """One-shot resampling of a whole signal (use ``StreamResampler`` for chunked streams)."""
    if sr_from == sr_to or len(x) == 0:
        return x.astype(np.float32, copy=False)
    g = np.gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


class StreamResampler:
    """Stateful resampler for a chunked stream: no clicks at chunk edges, no length drift (soxr)."""

    def __init__(self, sr_to: int) -> None:
        self.sr_to = sr_to
        self._sr_from: int | None = None
        self._rs = None

    def __call__(self, x: np.ndarray, sr_from: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if sr_from == self.sr_to:
            return x
        if sr_from != self._sr_from:
            import soxr

            self._sr_from = sr_from
            self._rs = soxr.ResampleStream(sr_from, self.sr_to, 1, dtype="float32", quality="HQ")
        return self._rs.resample_chunk(x)


def pcm16_to_float(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class Speaker:
    """Non-blocking speaker with a playback clock and a short, flushable queue."""

    def __init__(self, sample_rate: int, on_chunk: Callable[[np.ndarray, int], None] | None = None) -> None:
        self.sample_rate = sample_rate
        self.on_chunk = on_chunk
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._lock = threading.Lock()
        self._queued_until = 0.0  # monotonic time when everything queued so far will have been heard
        self._voice_until = 0.0  # ... when the last NON-SILENT queued audio will have been heard (echo gating)
        self._pos = 0.0  # stream position (s) of the end of the queue
        self._epoch = 0  # bumped on flush, so stale chunks are dropped
        self._resampler = StreamResampler(sample_rate)
        self.on_flush: Callable[[], None] | None = None

    # -- clock ---------------------------------------------------------------------------------------------------
    @property
    def position(self) -> float:
        """Stream position (s) where the next written chunk will start."""
        with self._lock:
            return self._pos

    def time_of(self, pos: float) -> float:
        """Monotonic wall-clock time when stream position ``pos`` is (or was) heard.

        Positions beyond what is queued (text that arrives before its audio) are counted from the moment new
        audio would start playing, not from a stale end-of-queue time.
        """
        with self._lock:
            if pos >= self._pos:
                return max(self._queued_until, time.monotonic() + self.latency_s()) + (pos - self._pos)
            return self._queued_until - (self._pos - pos)

    def busy(self, tail_s: float = 0.0) -> bool:
        """True while queued *voiced* audio is still playing (plus ``tail_s`` of echo tail).

        Silence does not count: GPT-Live streams silent audio continuously while it listens.
        """
        with self._lock:
            return time.monotonic() < self._voice_until + tail_s

    # -- writing -------------------------------------------------------------------------------------------------
    def write(self, audio: np.ndarray, sample_rate: int) -> float:
        """Queue mono float32 audio; returns its start position on the stream clock."""
        x = self._resampler(audio, sample_rate)
        if len(x) == 0:
            return self.position
        with self._lock:
            now = time.monotonic()
            start = self._pos
            dur = len(x) / self.sample_rate
            self._queued_until = max(self._queued_until, now + self.latency_s()) + dur
            if len(x) and float(np.max(np.abs(x))) > 0.01:  # roughly -40 dBFS peak
                self._voice_until = self._queued_until
            self._pos += dur
            epoch = self._epoch
        if self.on_chunk is not None:
            self.on_chunk(x, self.sample_rate)
        self._q.put((epoch, x))
        return start

    def flush(self) -> None:
        """Barge-in: drop everything not yet played."""
        with self._lock:
            self._epoch += 1
            now = time.monotonic()
            # the stream clock jumps forward: whatever was queued is now "already heard"
            self._queued_until = self._voice_until = now
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        if self.on_flush:
            self.on_flush()

    def latency_s(self) -> float:
        return 0.0

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class LocalSpeaker(Speaker):
    """sounddevice output: a writer thread performs blocking writes, which paces the queue in real time."""

    def __init__(self, sample_rate: int = 24000, device: int | str | None = None, **kw) -> None:
        super().__init__(sample_rate, **kw)
        self.device = device
        self._stream = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    def latency_s(self) -> float:
        return float(self._stream.latency) if self._stream is not None else 0.05

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32", device=self.device, latency="low"
        )
        self._stream.start()
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="speaker", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        block = int(self.sample_rate * 0.02)
        while self._running.is_set():
            try:
                epoch, x = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            for i in range(0, len(x), block):
                if epoch != self._epoch or not self._running.is_set():
                    break
                self._stream.write(x[i : i + block].reshape(-1, 1))

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=1)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class RobotSpeaker(Speaker):
    """Reachy Mini speaker: ``push_audio_sample`` is non-blocking, so we pace pushes ourselves (~120 ms ahead)."""

    AHEAD_S = 0.12

    def __init__(self, robot, latency_s: float = 0.15, **kw) -> None:
        super().__init__(robot.media.get_output_audio_samplerate(), **kw)
        self.robot = robot
        self._latency = latency_s  # pacing lead + GStreamer queue/mixer/sink (not measured yet; see Backlog)
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._pushed_until = 0.0

    def latency_s(self) -> float:
        return self._latency

    def flush(self) -> None:
        super().flush()
        self._pushed_until = 0.0
        audio = getattr(self.robot.media, "audio", None)
        clear = getattr(audio, "clear_player", None)
        if callable(clear):  # drop what is already inside the robot's playback pipeline
            try:
                clear()
            except Exception as e:  # noqa: BLE001
                logger.debug("clear_player failed: %s", e)

    def start(self) -> None:
        self.robot.media.start_playing()
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="robot-speaker", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        block = int(self.sample_rate * 0.04)
        channels = self.robot.media.get_output_channels() or 1
        while self._running.is_set():
            try:
                epoch, x = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            for i in range(0, len(x), block):
                if epoch != self._epoch or not self._running.is_set():
                    break
                chunk = x[i : i + block]
                now = time.monotonic()
                wait = self._pushed_until - now - self.AHEAD_S
                if wait > 0:
                    time.sleep(wait)
                frame = np.repeat(chunk[:, None], channels, axis=1) if channels > 1 else chunk[:, None]
                self.robot.media.push_audio_sample(frame.astype(np.float32))
                self._pushed_until = max(self._pushed_until, time.monotonic()) + len(chunk) / self.sample_rate

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=1)
        try:
            self.robot.media.stop_playing()
        except Exception:  # noqa: BLE001
            pass


class Microphone:
    def __init__(self) -> None:
        self.muted = False

    def start(self, callback: MicCallback) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class LocalMicrophone(Microphone):
    def __init__(self, sample_rate: int = 16000, device: int | str | None = None, block_s: float = 0.04) -> None:
        super().__init__()
        self.sample_rate, self.device, self.block = sample_rate, device, int(sample_rate * block_s)
        self._stream = None

    def start(self, callback: MicCallback) -> None:
        import sounddevice as sd

        def cb(indata, frames, t, status):
            x = indata[:, 0].copy()
            callback(np.zeros_like(x) if self.muted else x, self.sample_rate)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32", device=self.device, blocksize=self.block, callback=cb
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class RobotMicrophone(Microphone):
    def __init__(self, robot) -> None:
        super().__init__()
        self.robot = robot
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, callback: MicCallback) -> None:
        self.robot.media.start_recording()
        sr = self.robot.media.get_input_audio_samplerate()
        self._running.set()

        def run() -> None:
            while self._running.is_set():
                frame = self.robot.media.get_audio_sample()
                if frame is None:
                    time.sleep(0.005)
                    continue
                x = np.asarray(frame, dtype=np.float32)
                x = x[:, 0] if x.ndim == 2 else x
                callback(np.zeros_like(x) if self.muted else x, sr)

        self._thread = threading.Thread(target=run, name="robot-mic", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=1)
        try:
            self.robot.media.stop_recording()
        except Exception:  # noqa: BLE001
            pass


class AudioIO:
    """Mic + speaker pair, with optional half-duplex gating (mic muted while the robot speaks)."""

    def __init__(self, mic: Microphone, speaker: Speaker, half_duplex: bool = True, echo_tail_s: float = 0.35) -> None:
        self.mic, self.speaker = mic, speaker
        self.half_duplex, self.echo_tail_s = half_duplex, echo_tail_s

    def gate(self) -> bool:
        """True when the mic should be treated as silent (robot is talking and we run half-duplex)."""
        return self.half_duplex and self.speaker.busy(self.echo_tail_s)

    @classmethod
    def create(cls, kind: str, robot=None, out_rate: int = 24000, half_duplex: bool | None = None, on_chunk=None):
        """``half_duplex=None`` picks the backend default (robot: full duplex, local: half duplex)."""
        if kind == "robot":
            if robot is None:
                raise ValueError("robot audio needs a connected ReachyMini")
            spk = RobotSpeaker(robot, on_chunk=on_chunk)
            mic = RobotMicrophone(robot)
            # Pollen's conversation app streams this mic full-duplex while the speaker plays (it relies on the mic
            # array's echo cancellation). Not yet verified on our unit — pass half_duplex=True if it hears itself.
            return cls(mic, spk, half_duplex=False if half_duplex is None else half_duplex)
        spk = LocalSpeaker(out_rate, on_chunk=on_chunk)
        mic = LocalMicrophone(16000)
        # laptop speaker + mic without AEC would make the model hear itself: half duplex unless told otherwise
        return cls(mic, spk, half_duplex=True if half_duplex is None else half_duplex)
