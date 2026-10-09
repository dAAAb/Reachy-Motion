"""Spoken control commands: switch voice mode, change volume, look at me / stop looking.

Matched on the person's *final* transcript in every mode (the Taiwanese ASR writes Mandarin Han, so Mandarin
patterns cover it too). A command needs an action word, so merely mentioning "台語" or "GPT" does not switch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Command:
    kind: str  # "mode" | "volume" | "gaze"
    arg: str | int | bool
    text: str = ""


# strong verbs switch to any mode; "講/用/說" only count right before 台語 ("用台語講", "講台語")
_SWITCH = r"(切換|切到|切成|換成|換到|換做|換去|改成|改用|轉成|轉做|轉去|switch( to)?|change( to)?)"
_SPEAK_TAIGI = r"(用|講|說|請講)(台語|臺語)|speak (taiwanese|hokkien|taigi)"
_MODES = [
    ("taigi", r"台語|臺語|台灣話|臺灣話|閩南語|河洛話|taigi|taiwanese|hokkien"),
    ("elevenlabs", r"eleven ?labs|11 ?labs|伊萊文|伊雷文|eleven|寶博的聲音|我的聲音"),
    ("gpt-live", r"gpt|g ?p ?t|openai|open ai|live 模式|吉批踢"),
]
_BACK_TO_DEFAULT = r"(華語|國語|中文|普通話|mandarin|chinese)"

_CN_NUM = {"零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_to_int(s: str) -> int | None:
    """'70' / '七十' / '一百' / '五十五' -> int."""
    if s.isdigit():
        return int(s)
    total, cur = 0, 0
    for ch in s:
        if ch == "百":
            total += (cur or 1) * 100
            cur = 0
        elif ch == "十":
            total += (cur or 1) * 10
            cur = 0
        elif ch in _CN_NUM:
            cur = _CN_NUM[ch]
        else:
            return None
    return total + cur


def parse_command(text: str) -> Command | None:
    t = text.strip().lower()
    if not t:
        return None

    # --- volume ---------------------------------------------------------------------------------------------------
    m = re.search(r"(音量|聲音|volume)\D{0,6}?(\d{1,3}|[零一二兩三四五六七八九十百]{1,4})", t)
    if m and re.search(r"(調|設|轉|改|開|set|turn|到|成)", t):
        n = _cn_to_int(m.group(2))
        if n is not None:
            return Command("volume", max(0, min(100, n)), text)
    if re.search(r"(最大聲|音量.{0,3}最大|max(imum)? volume|full volume)", t):
        return Command("volume", 100, text)
    if re.search(r"(大聲一點|大聲點|大聲一些|調大聲|聲音.{0,2}大一點|音量.{0,3}(調高|大一點|加大)|louder|volume up|"
                 r"turn it up|大聲些|較大聲)", t):
        return Command("volume", "+", text)
    if re.search(r"(小聲一點|小聲點|小聲一些|調小聲|聲音.{0,2}小一點|音量.{0,3}(調低|小一點|減)|quieter|softer|"
                 r"volume down|turn it down|較細聲|細聲一點)", t):
        return Command("volume", "-", text)

    # --- gaze -----------------------------------------------------------------------------------------------------
    if re.search(r"(不要|別|毋通|莫|stop|don'?t).{0,4}(看我|盯著我|看著我|look at me|staring)", t):
        return Command("gaze", False, text)
    if re.search(r"(看著我|看我這邊|看向我|盯著我|look at me|face me)", t):
        return Command("gaze", True, text)

    # --- voice mode -----------------------------------------------------------------------------------------------
    if re.search(_SPEAK_TAIGI, t):
        return Command("mode", "taigi", text)
    if re.search(_SWITCH, t):
        for mode, pat in _MODES:
            if re.search(pat, t):
                return Command("mode", mode, text)
        if re.search(_BACK_TO_DEFAULT, t):
            return Command("mode", "default", text)
    if re.search(r"(換回|改回|轉回|switch back)", t) and re.search(_BACK_TO_DEFAULT, t):
        return Command("mode", "default", text)
    return None
