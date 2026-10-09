import time

import pytest

from reachy_motion.commands import parse_command
from reachy_motion.intent import _cmd_from_call
from reachy_motion.motion.director import Director
from reachy_motion.voices.elevenlabs import _is_barge_in


@pytest.mark.parametrize("text,want", [
    ("切換成台語", ("mode", "taigi")), ("用台語講", ("mode", "taigi")), ("switch to GPT live", ("mode", "gpt-live")),
    ("換回國語", ("mode", "default")), ("大聲一點", ("volume", "+")), ("聲音調到七十", ("volume", 70)),
    ("不要看我", ("gaze", False)), ("看著我", ("gaze", True)),
    ("你可以說一下 GPT 是什麼嗎", None), ("今天台語課很好玩", None), ("我看過那部電影", None),
])
def test_keyword_fallback(text, want):
    c = parse_command(text)
    assert ((c.kind, c.arg) if c else None) == want


def test_tool_calls_to_commands():
    assert (lambda c: (c.kind, c.arg))(_cmd_from_call("set_volume", {"change": "set", "level": 140}, "x")) == ("volume", 100)
    assert (lambda c: (c.kind, c.arg))(_cmd_from_call("set_volume", {"change": "up"}, "x")) == ("volume", "+")
    assert (lambda c: (c.kind, c.arg))(_cmd_from_call("look_at_person", {"enabled": False}, "x")) == ("gaze", False)
    assert (lambda c: (c.kind, c.arg))(_cmd_from_call("switch_voice_mode", {"mode": "taigi"}, "x")) == ("mode", "taigi")
    assert _cmd_from_call("unknown_tool", {}, "x") is None


@pytest.mark.parametrize("heard,want", [("等一下", True), ("欸你先聽我說", True), ("停", True), ("請不用擔心", False),
                                        ("寶博士", False), ("嗯", False)])
def test_barge_in_vs_echo(heard, want):
    assert _is_barge_in(heard, "[溫柔] 聽得懂聽得懂，請不用擔心！我是寶博士，大家也叫我寶博。") is want


def test_tone_tags_reach_the_planner():
    seen = []

    class P:
        def plan(self, line, heard=None, tone=None, **kw):
            seen.append((line, tone))
            from reachy_motion.motion.planner import Gesture
            return Gesture("go .3 p=5", "x", "planner", line)

    d = Director(play=lambda m: None, planner=P(), time_of=lambda p: time.monotonic(), position=lambda: 0.0)
    d.set_reply_text("[害羞] 哎呀你這樣說我會不好意思。[開心] 不過謝謝你！")
    d.speech_text("哎呀你這樣說我會不好意思。")
    d.speech_text("不過謝謝你！")
    d.end_of_turn()
    time.sleep(0.3)
    d.close()
    assert seen == [("哎呀你這樣說我會不好意思。", "害羞"), ("不過謝謝你！", "開心")]
