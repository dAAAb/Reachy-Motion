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

# Appended to every persona / sent as context: the voice model must know it has a body (otherwise it says
# "I'm just a voice assistant, I can't move" — which is what our ElevenLabs agent did).
EMBODIMENT = (
    "【身體與能力】你現在住在一台 Reachy Mini 機器人裡：會動的頭（點頭、歪頭、轉頭）、兩支天線耳朵、會轉的身體，"
    "還有攝影機眼睛和麥克風。你說話時身體會依照內容自動做出對應的動作，被要求做動作（例如害羞、點頭、跳舞）時，"
    "直接開心答應並簡短描述你正在做的動作，不要說你沒有身體或不能動。"
    "被問到你看到什麼、對方手上拿什麼或穿什麼時，說「我看一下喔」就好，系統會用攝影機拍照並把畫面內容告訴你，"
    "不要說你看不到或不能辨識。"
    "系統也能幫你切換說話模式（GPT-Live、ElevenLabs、台語）、調整音量、看著對方；使用者提出時，簡短確認即可，"
    "系統會自動執行。"
)
TAIGI_EMBODIMENT = ("你蹛佇一台 Reachy Mini 機器人內底，有會振動的頭、兩支耳仔，閣有攝影機目睭看會著對方，"
                    "講話的時陣身軀會綴咧振動。人問你看著啥物，你就講「我看覓咧」，系統會共你看著的畫面講予你知。")

TAIGI_PERSONA = (
    "你是 Reachy Mini，一隻桌頂的細隻機器人。請用自然的臺灣台語漢字回答，毋通用華語。"
    "每擺一到兩句，總共三十五字以內，親切、有趣。"
)


def now_context(lang: str = "zh") -> str:
    """Current local date/time for the voice models (they don't know 'today'); the robot's clock is NTP-synced."""
    import datetime as _dt

    tz = _dt.timezone(_dt.timedelta(hours=8))  # Taipei; the robot's TZ is UTC by default
    now = _dt.datetime.now(tz)
    wk = "一二三四五六日"[now.weekday()]
    if lang == "taigi":
        return f"【這馬的時間】{now.year} 年 {now.month} 月 {now.day} 號，禮拜{wk}，{now.hour} 點 {now.minute} 分（台灣時間）。"
    return f"【現在時間】{now.year} 年 {now.month} 月 {now.day} 日 星期{wk} {now:%H:%M}（台灣時間）。"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # Earlier files win (override=False): explicit file, working dir, per-user config (used on the robot), repo checkout.
    candidates = [
        Path(os.environ["REACHY_MOTION_ENV_FILE"]) if os.environ.get("REACHY_MOTION_ENV_FILE") else None,
        Path.cwd() / ".env",
        Path.home() / ".config" / "reachy_motion" / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for p in candidates:
        if p is not None and p.exists():
            load_dotenv(p, override=False)


_STANDARD_ENV = {"openai_api_key", "elevenlabs_api_key", "elevenlabs_agent_id", "elevenlabs_voice_id"}


@dataclass
class Settings:
    mode: str = "gpt-live"  # gpt-live | elevenlabs | taigi
    audio: str = "auto"  # auto | robot | local
    half_duplex: bool | None = None  # None = backend default (local: on, robot: off)

    # gesture planner
    planner_model: str = "gpt-5.4-nano"
    vision_model: str = "gpt-5.4-mini"  # describes camera frames ("what do you see?")
    reflexes: bool = True

    # GPT-Live-1
    openai_api_key: str = field(default="", repr=False)
    gpt_live_model: str = "gpt-live-1"
    gpt_live_voice: str = "marin"
    gpt_live_backend_model: str = "gpt-5.4-mini"
    persona: str = DEFAULT_PERSONA + EMBODIMENT
    embodiment: str = EMBODIMENT
    gaze: bool = True  # look at the person (daemon-side face tracking)
    gaze_weight_idle: float = 0.85  # tracking weight while listening / idle
    gaze_weight_gesture: float = 0.45  # ... while a gesture plays, so the gesture shows through

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
    taigi_persona: str = TAIGI_PERSONA + TAIGI_EMBODIMENT

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
            if f.type == "float":
                setattr(s, f.name, float(env))
            elif f.type in ("bool", "bool | None"):
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
