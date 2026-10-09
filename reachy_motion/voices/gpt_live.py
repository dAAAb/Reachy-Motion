"""Mode 1 — OpenAI GPT-Live-1 (full-duplex voice, ``wss://api.openai.com/v1/live/sessions``).

GPT-Live-1 streams the assistant's speech audio and, separately, its transcript (``session.output_transcript.delta``).
The transcript feeds the director; the audio feeds the speaker (and through it the speech sway). The Live API has no
turn-done / interruption events, so turns are closed by a silence timer and barge-in is detected from user
transcript deltas arriving while the robot is still talking.

We speak the WebSocket protocol directly (JSON events, ``Authorization: Bearer``) instead of ``openai.live``: that
needs openai>=3.12, while the robot's shared apps venv keeps openai 2.x pinned for Pollen's conversation app.
Docs: https://developers.openai.com/api/docs/guides/live (see wiki/voices/GPT-Live-1.md).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time

import numpy as np

from reachy_motion.audio_io import StreamResampler, float_to_pcm16, pcm16_to_float
from reachy_motion.voices.base import VoiceMode

RATE = 24000
TURN_GAP_S = 0.9  # no assistant text for this long = the turn is over
URL = os.environ.get("REACHY_MOTION_GPT_LIVE_URL", "wss://api.openai.com/v1/live/sessions")


class GptLiveMode(VoiceMode):
    name = "gpt-live"
    output_rate = RATE
    late_grace_s = 1.8  # transcript arrives *after* its audio; one reply usually keeps one mood

    def run(self, stop: threading.Event) -> None:
        asyncio.run(self._run(stop))

    def _session(self) -> dict:
        s = self.settings
        return {
            "model": s.gpt_live_model,
            "instructions": s.persona,
            "audio": {"format": {"type": "audio/pcm", "rate": RATE}, "output": {"voice": s.gpt_live_voice}},
            "delegation": {
                "type": "responses",
                "responses": {
                    "model": s.gpt_live_backend_model,
                    "instructions": "Answer the delegated question briefly and factually, in the user's language.",
                    "tools": [{"type": "web_search"}],
                },
            },
        }

    async def _run(self, stop: threading.Event) -> None:
        import websockets

        s = self.settings
        if not s.openai_api_key:
            self.status("OPENAI_API_KEY is not set")
            return
        loop = asyncio.get_running_loop()
        mic_q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=200)
        mic_rs = StreamResampler(RATE)

        def on_mic(x: np.ndarray, sr: int) -> None:
            if self.audio.gate():
                x = np.zeros_like(x)  # half duplex: keep the timeline, but don't let it hear itself
            y = mic_rs(x, sr)
            loop.call_soon_threadsafe(lambda: mic_q.full() or mic_q.put_nowait(y))

        headers = {"Authorization": f"Bearer {s.openai_api_key}"}
        async with websockets.connect(URL, additional_headers=headers, max_size=None) as ws:

            async def send(ev: dict) -> None:
                await ws.send(json.dumps(ev))

            await send({"type": "session.start", "event_id": "start", "session": self._session()})
            self.status("connecting…")
            last_text = 0.0
            stream_base: float | None = None
            t_started = time.monotonic()
            in_turn = False
            user_buf = ""
            user_last = 0.0

            async def pump_mic() -> None:
                while not stop.is_set():
                    y = await mic_q.get()
                    await send({"type": "session.input_audio.append",
                                "audio": base64.b64encode(float_to_pcm16(y)).decode()})

            async def watch() -> None:
                nonlocal in_turn, user_buf
                while not stop.is_set():
                    await asyncio.sleep(0.1)
                    now = time.monotonic()
                    if in_turn and now - last_text > TURN_GAP_S:
                        in_turn = False
                        self.director.end_of_turn()
                    if user_buf and now - user_last > 0.8:
                        self.director.user_text(user_buf)
                        user_buf = ""
                try:
                    await send({"type": "session.close"})
                    await asyncio.wait_for(closed.wait(), 3)
                except Exception:  # noqa: BLE001 - closing anyway
                    pass
                await ws.close()

            closed = asyncio.Event()
            tasks: list[asyncio.Task] = []
            try:
                async for raw in ws:
                    ev = json.loads(raw)
                    t = ev.get("type")
                    if t == "session.started":
                        t_started = time.monotonic()
                        self.status("live — say hi!")
                        self.audio.mic.start(on_mic)
                        tasks += [asyncio.create_task(pump_mic()), asyncio.create_task(watch())]
                    elif t == "session.output_audio.delta":
                        if stream_base is None:  # calibrate: session time 0 = speaker position minus elapsed
                            stream_base = self.audio.speaker.position - (time.monotonic() - t_started)
                        self.audio.speaker.write(pcm16_to_float(base64.b64decode(ev["delta"])), RATE)
                    elif t == "session.output_transcript.delta":
                        last_text, in_turn = time.monotonic(), True
                        # Live audio streams continuously (silence included) from about session start, so the
                        # transcript's session-timeline start_ms maps onto our speaker stream (see stream_base).
                        base = stream_base if stream_base is not None else self.audio.speaker.position
                        self.director.speech_text(ev["delta"], base + ev.get("start_ms", 0) / 1000.0)
                        self.on_event({"type": "robot_delta", "text": ev["delta"], "start_ms": ev.get("start_ms")})
                    elif t == "session.input_transcript.delta":
                        if not user_buf:
                            if self.audio.speaker.busy() and not self.audio.half_duplex:
                                self.barge_in()
                            self.director.listening()
                        user_buf += ev["delta"]
                        user_last = time.monotonic()
                        self.on_event({"type": "user_delta", "text": ev["delta"], "start_ms": ev.get("start_ms")})
                        self.director.anticipate(user_buf)  # GPT-Live answers instantly: plan the reaction now
                    elif t == "error":
                        self.status(f"error: {ev.get('error', ev)}")
                    elif t == "session.closed":
                        closed.set()
                        self.status(f"closed ({ev.get('reason')})")
                        break
            except websockets.ConnectionClosed as e:
                self.status(f"connection closed ({e.code} {e.reason})")
            finally:
                for task in tasks:
                    task.cancel()
                self.audio.mic.stop()
