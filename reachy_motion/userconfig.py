"""Per-device settings file edited from the settings page: API keys, agent / voice IDs, defaults.

Lives at ``~/.config/reachy_motion/.env`` (mode 600) on the machine that runs the app — never in the package or
the Hugging Face Space. Secrets are write-only through the API: the page can set or clear them and see whether they
are set, but their values are never sent back.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

CONFIG_PATH = Path.home() / ".config" / "reachy_motion" / ".env"

# key -> (secret?, label). Only these keys can be written from the page.
FIELDS: dict[str, tuple[bool, str]] = {
    "OPENAI_API_KEY": (True, "OpenAI API key (GPT-Live-1, gestures, vision, web)"),
    "ELEVENLABS_API_KEY": (True, "ElevenLabs API key (only for a private agent)"),
    "ELEVENLABS_AGENT_ID": (False, "ElevenLabs agent ID"),
    "ELEVENLABS_VOICE_ID": (False, "ElevenLabs voice ID (optional override)"),
    "REACHY_MOTION_MODE": (False, "Start in mode: gpt-live | elevenlabs | taigi"),
    "REACHY_MOTION_GPT_LIVE_VOICE": (False, "GPT-Live voice (e.g. marin)"),
    "REACHY_MOTION_MOTION_ENGINE": (False, "Motion engine: recipe | flow | local"),
    "REACHY_MOTION_MOTION_SERVER": (False, "Motion server: auto | http://host:18110 | off"),
    "REACHY_MOTION_LOCAL_PLANNER": (False, "Local planner model: 0.8b | 4b"),
    "REACHY_MOTION_LOCATION": (False, "Location for weather/news (optional)"),
    "REACHY_MOTION_TIMEZONE": (False, "Time zone, IANA name (e.g. Asia/Taipei)"),
}
_LINE = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$")


def read(path: Path = CONFIG_PATH) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue
            m = _LINE.match(line)
            if m:
                out[m.group(1)] = m.group(2).strip().strip("'\"")
    except OSError:
        pass
    return out


def write(updates: dict[str, str | None], path: Path = CONFIG_PATH) -> dict[str, str]:
    """Merge ``updates`` into the file (``None``/empty = remove), atomically, mode 600; mirror into os.environ."""
    for key, value in updates.items():
        if key not in FIELDS:
            raise ValueError(f"not an editable setting: {key}")
        if value is not None and (len(value) > 400 or any(c in value for c in "\r\n\0")):
            raise ValueError(f"invalid value for {key}")
    current = read(path)
    for key, value in updates.items():
        value = (value or "").strip()
        if value:
            current[key] = value
        else:
            current.pop(key, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    body = "# Reachy-Motion settings for this device. Keep private: contains API keys.\n" + "".join(
        f"{k}={v}\n" for k, v in current.items())
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".env.")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(body)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    for key in updates:
        if key in current:
            os.environ[key] = current[key]
        else:
            os.environ.pop(key, None)
    return current


def public_view(path: Path = CONFIG_PATH) -> dict:
    """What the page may see: secrets only as set / not set; plain settings with their value."""
    current = read(path)
    fields = {}
    for key, (secret, label) in FIELDS.items():
        value = current.get(key) or os.environ.get(key, "")
        fields[key] = {"label": label, "secret": secret, "set": bool(value), "value": None if secret else value}
    return {"path": str(path), "fields": fields}
