"""Runtime settings, from environment variables / ``.env`` (never hard-code keys)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_PERSONA = (
    "你是 Reachy Mini，一隻放在桌上、會動的小機器人，有一顆可以點頭歪頭的頭和兩支天線耳朵。"
    "用台灣華語（繁體中文）自然、溫暖、有點俏皮地聊天；對方講英文就用英文回答。"
    "每次回答一到三句，口語、簡短，像朋友聊天，不要列點。你的身體會跟著你說的話做動作，所以可以大方表達情緒。"
)

TAIGI_PERSONA = (
    "你是 Reachy Mini，一隻桌頂的細隻機器人。請用自然的臺灣台語漢字回答，毋通用華語。"
    "每擺一到兩句，總共三十五字以內，親切、有趣。"
)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for p in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if p.exists():
            load_dotenv(p, override=False)


_STANDARD_ENV = {"openai_api_key", "elevenlabs_api_key", "elevenlabs_agent_id", "elevenlabs_voice_id"}


@dataclass
class Settings:
    mode: str = "gpt-live"  # gpt-live | elevenlabs | taigi
    audio: str = "auto"  # auto | robot | local
    half_duplex: bool | None = None  # None = backend default (local: on, robot: off)

    # gesture planner
    planner_model: str = "gpt-5.4-nano"
    reflexes: bool = True

    # GPT-Live-1
    openai_api_key: str = field(default="", repr=False)
    gpt_live_model: str = "gpt-live-1"
    gpt_live_voice: str = "marin"
    gpt_live_backend_model: str = "gpt-5.4-mini"
    persona: str = DEFAULT_PERSONA

    # ElevenLabs Agents
    elevenlabs_api_key: str = field(default="", repr=False)
    elevenlabs_agent_id: str = ""
    elevenlabs_voice_id: str = ""
    elevenlabs_tts_model: str = "eleven_v4_turbo"
    elevenlabs_override: bool = False  # send voice/model overrides (must be enabled in the agent's Security tab)

    # Taiwanese local pipeline (services from the AIRI NTU-VH2026 lab, see wiki/voices/Taigi Local Pipeline.md)
    taigi_asr_url: str = "http://127.0.0.1:18001/v1/audio/transcriptions"
    taigi_llm_url: str = "http://127.0.0.1:11434/v1/chat/completions"
    taigi_llm_model: str = "hf.co/Speech-AI-Research-Center/SARC-Taigi-LLM-12b-GGUF:Q4_K_M"
    taigi_tts_url: str = "http://127.0.0.1:8883/v1/audio/speech"
    taigi_tts_model: str = "taigi-hanzi"
    taigi_tts_voice: str = "taigi-demo-reference"
    taigi_persona: str = TAIGI_PERSONA

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        _load_dotenv()
        s = cls()
        for f in fields(cls):
            env = os.environ.get(f"REACHY_MOTION_{f.name.upper()}")
            if env is None and f.name in _STANDARD_ENV:  # well-known names, e.g. OPENAI_API_KEY
                env = os.environ.get(f.name.upper())
            if env is None:
                continue
            if f.type in ("bool", "bool | None"):
                setattr(s, f.name, env.strip().lower() in ("1", "true", "yes", "on"))
            else:
                setattr(s, f.name, env)
        for k, v in overrides.items():
            if v is not None:
                setattr(s, k, v)
        return s

    def public(self) -> dict:
        """Settings safe to show in the web UI (no secrets)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if "api_key" in f.name:
                v = bool(v)
            out[f.name] = v
        return out
