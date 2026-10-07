"""Record a real voice session headlessly and render it in MuJoCo — no robot, no microphone needed.

A WAV file stands in for the microphone (played in real time), the robot's reply audio is captured instead of played,
and every animator pose is logged. Afterwards the pose trace is rendered with Reachy Mini's MuJoCo model and muxed
with both voices into an MP4, plus a JSON log of transcripts, gestures and timings.

    python scripts/record_session.py --mode elevenlabs --input question.wav --seconds 25 --out recordings/el
    python scripts/record_session.py --mode taigi --input taigi_q.wav --out recordings/taigi

Needs: pip install imageio-ffmpeg pillow  (MuJoCo comes with reachy-mini[mujoco]).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reachy_motion.audio_io import AudioIO, Microphone, Speaker, resample
from reachy_motion.config import Settings
from reachy_motion.engine import Engine

REC_RATE = 24000


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        ch = w.getnchannels()
        return (x.reshape(-1, ch).mean(axis=1) if ch > 1 else x), w.getframerate()


class FileMicrophone(Microphone):
    """Plays a WAV into the session in real time after ``delay_s``, then silence."""

    def __init__(self, audio: np.ndarray, sr: int, t0: float, delay_s: float = 1.5, block_s: float = 0.04):
        super().__init__()
        self.x, self.sr, self.t0, self.delay, self.block = audio, sr, t0, delay_s, int(sr * block_s)
        self._run = threading.Event()
        self.started_at: float | None = None

    def start(self, callback):
        self._run.set()

        def loop():
            i, nxt = 0, time.monotonic()
            while self._run.is_set():
                now = time.monotonic()
                if now - self.t0 >= self.delay and i < len(self.x):
                    if self.started_at is None:
                        self.started_at = now
                    chunk = self.x[i : i + self.block]
                    i += self.block
                else:
                    chunk = np.zeros(self.block, np.float32)
                if len(chunk) < self.block:
                    chunk = np.pad(chunk, (0, self.block - len(chunk)))
                callback(np.zeros_like(chunk) if self.muted else chunk, self.sr)
                nxt += self.block / self.sr
                time.sleep(max(0.0, nxt - time.monotonic()))

        threading.Thread(target=loop, daemon=True).start()

    def stop(self):
        self._run.clear()


class RecordingSpeaker(Speaker):
    """Real-time paced 'speaker' that writes into a timeline buffer instead of a sound card."""

    def __init__(self, t0: float, **kw):
        super().__init__(REC_RATE, **kw)
        self.t0 = t0
        self.track = np.zeros(REC_RATE * 600, np.float32)
        self._running = threading.Event()

    def start(self):
        self._running.set()

        def loop():
            block = int(REC_RATE * 0.02)
            while self._running.is_set():
                try:
                    epoch, x = self._q.get(timeout=0.1)
                except Exception:  # noqa: BLE001
                    continue
                for i in range(0, len(x), block):
                    if epoch != self._epoch or not self._running.is_set():
                        break
                    c = x[i : i + block]
                    at = int((time.monotonic() - self.t0) * REC_RATE)
                    self.track[at : at + len(c)] += c[: max(0, len(self.track) - at)]
                    time.sleep(len(c) / REC_RATE)

        threading.Thread(target=loop, daemon=True).start()

    def stop(self):
        self._running.clear()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("gpt-live", "elevenlabs", "taigi"))
    ap.add_argument("--input", required=True, type=Path, help="WAV played as the person's voice")
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--out", type=Path, required=True, help="output path prefix (writes .mp4 and .json)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--delay", type=float, default=1.5, help="seconds before the input WAV starts")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "websockets", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    user_audio, user_sr = read_wav(a.input)
    t0 = time.monotonic()
    holder: dict = {}

    def factory(cls, on_chunk):
        mic = FileMicrophone(user_audio, user_sr, t0, delay_s=a.delay)
        spk = RecordingSpeaker(t0, on_chunk=on_chunk)
        holder["mic"], holder["spk"] = mic, spk
        return AudioIO(mic, spk, half_duplex=True)

    settings = Settings.from_env(mode=a.mode, audio="local")
    engine = Engine(settings, robot=None, audio_factory=factory)
    trace: list[tuple[float, np.ndarray]] = []
    engine.animator.on_pose(lambda p: trace.append((time.monotonic() - t0, p)))
    engine.start_animation()
    engine.start(a.mode)
    time.sleep(a.seconds)
    engine.close()

    events = list(engine.events)
    for e in events:
        e["session_s"] = round(e["t"] - (time.time() - (time.monotonic() - t0)), 2)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(events, ensure_ascii=False, indent=1))
    print(f"wrote {a.out.with_suffix('.json')} ({len(events)} events, {len(trace)} poses)")
    if a.no_render or not trace:
        return

    from reachy_animation.sim import Sim, _caption, write_video

    times = np.array([t for t, _ in trace])
    poses = np.stack([p for _, p in trace])
    n = int(a.seconds * a.fps)
    grid = np.arange(n) / a.fps
    resampled = np.stack([np.interp(grid, times, poses[:, j]) for j in range(9)], -1)
    sim = Sim(a.fps, 640, 480)
    sim.camera.distance, sim.camera.lookat[2] = 0.60, 0.19  # a bit wider and higher: keep the antennas in frame
    frames = sim.render(resampled)

    # captions: latest user line / robot line / gesture at each frame
    caps = [(e["session_s"], e) for e in events if e["type"] in ("user", "gesture")]
    out_frames = []
    for i, f in enumerate(frames):
        t = i / a.fps
        cur = [e for s, e in caps if s <= t]
        g = next((e for e in reversed(cur) if e["type"] == "gesture"), None)
        txt = f"t={t:4.1f}s  {g['source']}: {g['idea'][:60]}" if g else f"t={t:4.1f}s"
        out_frames.append(_caption(f, txt))

    spk = holder["spk"].track[: int(a.seconds * REC_RATE)]
    mic = np.zeros_like(spk)
    if holder["mic"].started_at is not None:
        u = resample(user_audio, user_sr, REC_RATE)
        at = int((holder["mic"].started_at - t0) * REC_RATE)
        mic[at : at + len(u)] = u[: max(0, len(mic) - at)]
    mix = np.clip(spk + 0.8 * mic, -1, 1)
    write_video(out_frames, a.out.with_suffix(".mp4"), a.fps, (mix * 32767).astype(np.int16), REC_RATE)
    print(f"wrote {a.out.with_suffix('.mp4')}")


if __name__ == "__main__":
    main()
