"""Mode 3 — Taiwanese Hokkien (台語), fully local on a Mac.

A cascaded pipeline over the local services of the AIRI NTU-VH2026 speech lab (see wiki/voices/Taigi Local Pipeline.md):

    mic ─VAD─▶ Breeze-ASR-26 (MLX) ─▶ SARC-Taigi-LLM-12b (Ollama, streamed) ─sentence─▶ KaedeTai GPT-SoVITS ─▶ speaker
                                                               └────────────────sentence─▶ director (gesture plan)

Not streaming end-to-end (ASR decodes whole segments, TTS renders whole sentences), so expect ~2-3 s from the end of
your sentence to the first audio. Each sentence is planned *while* it is synthesised, so its gesture still starts
exactly with its audio. The services run on the Mac, so either run Reachy-Motion on the Mac (``--audio local``), or
expose the services on the LAN and point the URLs at the Mac when running on the robot.
"""

from __future__ import annotations

import collections
import io
import json
import logging
import re
import threading
import time
import wave
from concurrent.futures import Future, ThreadPoolExecutor

import httpx
import numpy as np

from reachy_motion.audio_io import StreamResampler, float_to_pcm16, pcm16_to_float
from reachy_motion.voices.base import VoiceMode

logger = logging.getLogger(__name__)

VAD_RATE = 16000
SENTENCE_END = re.compile(r"[。！？!?\n]")
SOFT_END = re.compile(r"[，,、；;]")
TTS_MAX_CHARS = 55  # KaedeTai refuses > 60 Han characters per request


def wav_bytes(x: np.ndarray, sr: int) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(float_to_pcm16(x))
    return bio.getvalue()


def read_wav(b: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(b)) as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        x = pcm16_to_float(w.readframes(n))
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


# KaedeTai (via Taibun) only reads Han characters / POJ: Latin letters and digits are rejected (HTTP 422).
NAME_MAP = {"reachy mini": "瑞奇迷你", "reachy": "瑞奇", "mini": "迷你", "ok": "好", "ai": "人工智慧", "jc": "",
            "dr.": "博士", "dr": "博士"}
_DIGITS = "零一二三四五六七八九"


def han_number(n: int) -> str:
    if n < 10:
        return _DIGITS[n]
    if n >= 100000:
        return "".join(_DIGITS[int(d)] for d in str(n))
    out, units = "", [(10000, "萬"), (1000, "千"), (100, "百"), (10, "十")]
    zero = False
    for u, name in units:
        q, n = divmod(n, u)
        if q:
            if zero and out:
                out += "零"
            out += ("" if (u == 10 and q == 1 and not out) else han_number(q)) + name
            zero = False
        elif out:
            zero = True
    if n:
        out += ("零" if zero else "") + _DIGITS[n]
    return out


def tts_text(text: str) -> str:
    """Make a reply speakable by the Taiwanese TTS: map names, spell numbers, drop leftover Latin."""
    t = text
    for k in sorted(NAME_MAP, key=len, reverse=True):  # whole Latin words only ("drink" is not "dr")
        t = re.sub(r"(?<![A-Za-z])" + re.escape(k) + r"(?![A-Za-z])", NAME_MAP[k], t, flags=re.IGNORECASE)
    t = re.sub(r"\d+", lambda m: han_number(int(m.group())), t)
    t = re.sub(r"[A-Za-z][A-Za-z'.-]*", "", t)
    t = re.sub(r"\s+", "", t)
    return re.sub(r"^[，,、。]+", "", t)


class EnergyVAD:
    """Energy VAD with a minimum-statistics noise floor: good enough for one person talking to a desk robot.

    The floor is the quietest 40 ms block of the last ~3 s, so steady fan / servo noise raises it and never keeps the
    detector stuck "active"; speech pauses keep it low while someone talks.
    """

    def __init__(self, start_db: float = 12.0, end_silence_s: float = 0.7, min_speech_s: float = 0.35,
                 floor_window_s: float = 3.0) -> None:
        self.start_db, self.end_silence_s, self.min_speech_s = start_db, end_silence_s, min_speech_s
        self.floor_window_s = floor_window_s
        self._levels: collections.deque[tuple[float, float]] = collections.deque()  # (block duration, dB)
        self.active = False
        self.buf: list[np.ndarray] = []
        self.pre: list[np.ndarray] = []  # ~300 ms pre-roll so the first syllable isn't clipped
        self.speech_s = self.silence_s = 0.0
        self.warmup_s = 0.6  # learn the room before triggering

    @property
    def noise_db(self) -> float:
        return min((db for _, db in self._levels), default=-60.0)

    def reset(self) -> None:
        self.active, self.buf, self.pre = False, [], []

    def feed(self, x: np.ndarray) -> np.ndarray | None:
        """Feed 16 kHz audio; returns a finished utterance or None."""
        if len(x) == 0:
            return None
        db = 20 * np.log10(float(np.sqrt(np.mean(x**2))) + 1e-9)
        dur = len(x) / VAD_RATE
        self._levels.append((dur, db))
        total = sum(d for d, _ in self._levels)
        while self._levels and total - self._levels[0][0] >= self.floor_window_s:
            total -= self._levels.popleft()[0]
        if self.warmup_s > 0:
            self.warmup_s -= dur
            return None
        loud = db > self.noise_db + self.start_db
        if not self.active:
            self.pre = (self.pre + [x])[-8:]
            if loud:
                self.active, self.buf, self.speech_s, self.silence_s = True, list(self.pre), 0.0, 0.0
            return None
        self.buf.append(x)
        if loud:
            self.speech_s += dur
            self.silence_s = 0.0
        else:
            self.silence_s += dur
        if self.silence_s >= self.end_silence_s or sum(len(b) for b in self.buf) > VAD_RATE * 25:
            self.active = False
            utt = np.concatenate(self.buf)
            self.buf = []
            return utt if self.speech_s >= self.min_speech_s else None
        return None


class TaigiMode(VoiceMode):
    name = "taigi"
    output_rate = 32000  # GPT-SoVITS s2 renders at 32 kHz
    default_half_duplex = True  # no server-side echo handling: never let it transcribe its own voice

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.history: list[dict] = []
        self.http = httpx.Client(timeout=60)
        self._utterances: list[np.ndarray] = []
        self._cv = threading.Condition()

    # -- services ------------------------------------------------------------------------------------------------
    def check_services(self) -> list[str]:
        """Find ASR / LLM / TTS: configured URLs, their alternate local ports, then a LAN node over mDNS."""
        from reachy_motion.discovery import probe_taigi

        s = self.settings
        p = probe_taigi(s.taigi_asr_url, s.taigi_llm_url, s.taigi_tts_url, s.taigi_llm_model)
        s.taigi_asr_url, s.taigi_llm_url, s.taigi_tts_url = p.asr_url, p.llm_url, p.tts_url
        if p.llm_model:
            s.taigi_llm_model = p.llm_model
        if p.source != "configured":
            self.status(f"using Taiwanese services from {p.source}")
        return p.missing

    def transcribe(self, utt: np.ndarray) -> str:
        r = self.http.post(
            self.settings.taigi_asr_url,
            files={"file": ("u.wav", wav_bytes(utt, VAD_RATE), "audio/wav")},
            data={"model": "breeze-asr-26-mlx", "language": "zh", "response_format": "json"},
        )
        r.raise_for_status()
        return r.json().get("text", "").strip()

    def llm_stream(self, user: str):
        s = self.settings
        msgs = [{"role": "system", "content": s.taigi_persona}, *self.history[-8:], {"role": "user", "content": user}]
        body = {
            "model": s.taigi_llm_model, "messages": msgs, "stream": True, "temperature": 0.2, "max_tokens": 160,
            "reasoning_effort": "none",
        }
        with self.http.stream("POST", s.taigi_llm_url, json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                delta = json.loads(line[5:])["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        s = self.settings
        r = self.http.post(
            s.taigi_tts_url,
            json={"model": s.taigi_tts_model, "voice": s.taigi_tts_voice, "input": tts_text(text),
                  "response_format": "wav"},
        )
        r.raise_for_status()
        return read_wav(r.content)

    def say(self, text: str) -> None:
        """Speak a Taiwanese-Han sentence directly through the TTS (no LLM), with a gesture."""
        super().say(text)
        fut: Future = Future()
        self.director.speech_line(text, fut)
        try:
            x, sr = self.synthesize(text)
            fut.set_result(self.audio.speaker.write(x, sr))
        except Exception as e:  # noqa: BLE001
            fut.set_exception(e)
            logger.warning("notice TTS failed: %s", e)

    # -- loop ----------------------------------------------------------------------------------------------------
    def run(self, stop: threading.Event) -> None:
        missing = self.check_services()
        if missing:
            self.status("local Taiwanese services not reachable: " + ", ".join(missing) + " — see wiki/voices/Taigi Local Pipeline.md")
            return
        vad = EnergyVAD()
        vad_rs = StreamResampler(VAD_RATE)

        def on_mic(x: np.ndarray, sr: int) -> None:
            if self.audio.gate():  # never transcribe the robot's own voice
                vad.reset()
                return
            was_active = vad.active
            utt = vad.feed(vad_rs(x, sr))
            if vad.active and not was_active:
                self.director.listening()
            if utt is not None:
                with self._cv:
                    self._utterances.append(utt)
                    self._cv.notify()

        self.audio.mic.start(on_mic)
        self.status("live — 請講台語！(listening)")
        if self.announce:
            self.say(self.announce)
        try:
            while not stop.is_set():
                with self._cv:
                    self._cv.wait_for(lambda: self._utterances or stop.is_set(), timeout=0.25)
                    if not self._utterances:
                        continue
                    utt = self._utterances.pop()  # only answer the latest thing said
                    self._utterances.clear()
                self._turn(utt, stop)
        finally:
            self.audio.mic.stop()
            self.http.close()

    def _turn(self, utt: np.ndarray, stop: threading.Event) -> None:
        t0 = time.perf_counter()
        try:
            heard = self.transcribe(utt)
        except httpx.HTTPError as e:
            self.status(f"ASR failed: {e}")
            return
        if not heard:
            return
        t_asr = time.perf_counter()
        self.director.user_text(heard)

        # stream the LLM, cut sentences, synthesise them in order on a single FIFO TTS worker
        reply, buf, first_audio = "", "", [None]
        tts = ThreadPoolExecutor(max_workers=1, thread_name_prefix="taigi-tts")  # FIFO: sentence order is kept

        def speak(sentence: str, fut: Future) -> None:
            if stop.is_set():
                fut.cancel()
                return
            try:
                x, sr = self.synthesize(sentence)
            except Exception as e:  # noqa: BLE001 - HTTP errors, non-WAV bodies… always resolve the future
                fut.set_exception(e)
                self.status(f"TTS failed for {sentence!r}: {e}")
                return
            pos = self.audio.speaker.write(x, sr)
            if first_audio[0] is None:
                first_audio[0] = time.perf_counter()
            fut.set_result(pos)

        def emit(sentence: str) -> None:
            sentence = sentence.strip()
            if not re.search(r"[㐀-鿿A-Za-z]", sentence):
                return
            fut: Future = Future()
            self.director.speech_line(sentence, fut)  # plan in parallel with TTS; placed when audio is queued
            tts.submit(speak, sentence, fut)

        try:
            for delta in self.llm_stream(heard):
                if stop.is_set():
                    break
                reply += delta
                buf += delta
                m = SENTENCE_END.search(buf)
                if not m and len(buf) >= TTS_MAX_CHARS:
                    m = None
                    for sm in SOFT_END.finditer(buf):
                        m = sm
                    if m is None:
                        emit(buf[:TTS_MAX_CHARS])
                        buf = buf[TTS_MAX_CHARS:]
                        continue
                if m:
                    emit(buf[: m.end()])
                    buf = buf[m.end() :]
            if buf.strip():
                emit(buf)
        except (httpx.HTTPError, ValueError, KeyError) as e:
            self.status(f"LLM failed: {e}")
        tts.shutdown(wait=True, cancel_futures=stop.is_set())
        self.history += [{"role": "user", "content": heard}, {"role": "assistant", "content": reply}]
        lat = (first_audio[0] - t0) if first_audio[0] else float("nan")
        self.on_event({"type": "robot", "text": reply})
        self.on_event({"type": "timing", "asr_s": round(t_asr - t0, 2), "first_audio_s": round(lat, 2)})
