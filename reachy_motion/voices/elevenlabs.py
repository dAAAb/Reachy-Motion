"""Mode 2 — ElevenLabs Agents (Eleven v4 voices) over the raw Agents WebSocket.

We speak the protocol directly instead of using ``elevenlabs.conversational_ai.Conversation`` because the SDK never
surfaces the negotiated audio formats (``conversation_initiation_metadata``) — an agent set to pcm_24000/44100 would
play at the wrong speed — and we want the per-chunk character alignment to place gestures on the audio timeline.
Protocol: https://elevenlabs.io/docs/agents-platform/api-reference/agents-platform/websocket (see wiki/voices/ElevenLabs Agents.md).
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading

import httpx
import numpy as np

from reachy_motion.audio_io import StreamResampler, float_to_pcm16, pcm16_to_float
from reachy_motion.voices.base import VoiceMode

WS_URL = "wss://api.elevenlabs.io/v1/convai/conversation?agent_id={agent_id}"
SIGNED_URL = "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url"
TURN_GAP_S = 0.8


def _rate(fmt: str | None, default: int = 16000) -> int:
    """'pcm_24000' -> 24000. ulaw_8000 is not supported (we only decode PCM)."""
    if fmt and fmt.startswith("pcm_"):
        return int(fmt.split("_", 1)[1])
    return default


class ElevenLabsMode(VoiceMode):
    name = "elevenlabs"
    output_rate = 16000  # the local speaker resamples whatever the agent sends

    def run(self, stop: threading.Event) -> None:
        asyncio.run(self._run(stop))

    async def _url(self) -> str:
        s = self.settings
        if s.elevenlabs_api_key:  # private agent: exchange the key for a short-lived signed URL
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(
                    SIGNED_URL, params={"agent_id": s.elevenlabs_agent_id}, headers={"xi-api-key": s.elevenlabs_api_key}
                )
                r.raise_for_status()
                return r.json()["signed_url"]
        return WS_URL.format(agent_id=s.elevenlabs_agent_id)

    def _init_message(self) -> dict:
        s = self.settings
        msg: dict = {"type": "conversation_initiation_client_data"}
        if s.elevenlabs_override:
            tts: dict = {}
            if s.elevenlabs_voice_id:
                tts["voice_id"] = s.elevenlabs_voice_id
            if s.elevenlabs_tts_model:
                tts["model_id"] = s.elevenlabs_tts_model
            msg["conversation_config_override"] = {"tts": tts}
        return msg

    async def _run(self, stop: threading.Event) -> None:
        import websockets

        s = self.settings
        if not s.elevenlabs_agent_id:
            self.status("ELEVENLABS_AGENT_ID is not set")
            return
        loop = asyncio.get_running_loop()
        mic_q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=200)
        in_rate, out_rate = 16000, 16000
        last_interrupt = -1
        streaming_text = False  # agent_chat_response_part enabled -> plan from it (it arrives before the audio)
        last_text = 0.0

        mic_rs = StreamResampler(16000)  # replaced once the agent's input format is known

        def on_mic(x: np.ndarray, sr: int) -> None:
            if self.audio.gate():
                x = np.zeros_like(x)
            y = mic_rs(x, sr)
            loop.call_soon_threadsafe(lambda: mic_q.full() or mic_q.put_nowait(y))

        try:
            url = await self._url()
        except httpx.HTTPError as e:
            self.status(f"could not get a signed URL: {e}")
            return

        async with websockets.connect(url, max_size=None, ping_interval=None) as ws:
            await ws.send(json.dumps(self._init_message()))
            self.status("connecting…")
            started = False

            async def pump_mic() -> None:
                buf = np.zeros(0, np.float32)
                chunk = in_rate // 4  # 250 ms chunks, as the docs recommend
                while not stop.is_set():
                    buf = np.concatenate([buf, await mic_q.get()])
                    while len(buf) >= chunk:
                        b64 = base64.b64encode(float_to_pcm16(buf[:chunk])).decode()
                        buf = buf[chunk:]
                        await ws.send(json.dumps({"user_audio_chunk": b64}))

            async def stopper() -> None:
                nonlocal last_text
                while not stop.is_set():
                    await asyncio.sleep(0.1)
                    if last_text and loop.time() - last_text > TURN_GAP_S:  # no new aligned text: turn is over
                        last_text = 0.0
                        self.director.end_of_turn()
                await ws.close()

            tasks = [asyncio.create_task(stopper())]
            try:
                async for raw in ws:
                    m = json.loads(raw)
                    t = m.get("type")
                    if t == "conversation_initiation_metadata":
                        ev = m["conversation_initiation_metadata_event"]
                        in_rate = _rate(ev.get("user_input_audio_format"))
                        out_rate = _rate(ev.get("agent_output_audio_format"))
                        mic_rs = StreamResampler(in_rate)
                        self.status(f"live — agent audio {out_rate} Hz, mic {in_rate} Hz. Say hi!")
                        if not started:
                            started = True
                            self.audio.mic.start(on_mic)
                            tasks.append(asyncio.create_task(pump_mic()))
                    elif t == "audio":
                        ev = m["audio_event"]
                        if int(ev.get("event_id", 0)) <= last_interrupt:
                            continue
                        pos = self.audio.speaker.write(pcm16_to_float(base64.b64decode(ev["audio_base_64"])), out_rate)
                        al = ev.get("alignment") or {}
                        if not streaming_text and al.get("chars"):
                            text = "".join(al["chars"])
                            first_ms = (al.get("char_start_times_ms") or [0])[0]
                            self.director.speech_text(text, pos + first_ms / 1000.0)
                            last_text = loop.time()
                    elif t == "agent_chat_response_part":
                        part = m.get("text_response_part", {})
                        streaming_text = True
                        if part.get("type") == "delta" and part.get("text"):
                            self.director.speech_text(part["text"])
                        elif part.get("type") == "stop":
                            self.director.end_of_turn()
                    elif t == "agent_response":
                        # arrives with the FIRST audio of the turn, not the last: log it, don't close the turn here
                        self.on_event({"type": "robot", "text": m["agent_response_event"]["agent_response"]})
                    elif t == "agent_response_correction":
                        ev = m["agent_response_correction_event"]
                        self.on_event({"type": "robot", "text": ev["corrected_agent_response"] + " ⟂"})
                    elif t == "user_transcript":
                        self.director.user_text(m["user_transcription_event"]["user_transcript"])
                    elif t == "tentative_user_transcript":
                        self.director.listening()
                    elif t == "interruption":
                        last_interrupt = int(m["interruption_event"]["event_id"])
                        self.barge_in()
                    elif t == "ping":
                        ev = m["ping_event"]
                        await ws.send(json.dumps({"type": "pong", "event_id": ev["event_id"]}))
                    elif t == "client_tool_call":
                        call = m["client_tool_call"]
                        await ws.send(json.dumps({
                            "type": "client_tool_result", "tool_call_id": call.get("tool_call_id"),
                            "result": "gestures are generated automatically from speech", "is_error": False,
                        }))
            except websockets.ConnectionClosed as e:
                self.status(f"closed ({e.code} {e.reason})")
            finally:
                for task in tasks:
                    task.cancel()
                self.audio.mic.stop()
