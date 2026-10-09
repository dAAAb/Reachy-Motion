import time

import numpy as np
import pytest

from reachy_motion.motion import prompt
from reachy_motion.motion.director import Director, spoken_seconds
from reachy_motion.motion.planner import REFLEXES, TALK_FALLBACK, Gesture, reflex
from reachy_motion.motion.recipe import FPS, SAFE, RecipeError, check, expand, recipe_to_poses


def test_expand_duration_and_neutral_start():
    f = expand("go 1 e=150 p=22 z=-16 E=5 | osc 3 z 4 .9 E=6 | hold 1 E=4")
    assert f.shape[1] == 8
    assert 4.0 < len(f) / FPS < 6.5
    assert np.allclose(f[0], [15, 15, 0, 0, 0, 3, 0, 0.5])


@pytest.mark.parametrize("bad", ["", "jump 1", "go x p=1", "go 1 p=99", "osc 1 p 5 .1", "go 1 q=3", "go 20"])
def test_invalid_recipes(bad):
    assert check(bad) is not None
    with pytest.raises((RecipeError, ValueError)):
        expand(bad)


def test_poses_within_safety_envelope():
    poses = recipe_to_poses("go .2 e=-25 p=-30 r=25 y=50 z=25 b=60 E=12 | hold 2 E=12", seed=1)
    assert poses.shape[1] == 9
    assert np.all(np.abs(poses[:, 4]) <= SAFE["pitch"] + 1e-9)
    assert np.all(np.abs(poses[:, 3]) <= SAFE["roll"] + 1e-9)
    assert np.all(np.abs(poses[:, 2]) <= SAFE["z"] + 1e-9)


def test_ear_sign_convention():
    # drooped ears: right antenna negative, left positive (reachy-motion-generator convention)
    poses = recipe_to_poses("go .5 e=150 E=0 | hold .5 E=0", seed=0, detail=0)
    assert poses[-1, 6] < -2.0 and poses[-1, 7] > 2.0


def test_all_builtin_recipes_valid():
    for _, _, r in REFLEXES:
        assert check(r) is None, r
    assert check(TALK_FALLBACK) is None
    for r in list(prompt.EXAMPLES.values()) + list(prompt.TALK_EXAMPLES.values()):
        assert check(r) is None, r


def test_reflex_cues():
    assert reflex("哈哈哈，你好好笑").idea == "laughing"
    assert reflex("不行啦").idea == "no"
    assert reflex("你今天好嗎？").idea == "question"
    assert reflex("今天天氣不錯") is None


def test_parse_reply_tolerates_fences():
    idea, recipe = prompt.parse_reply('```json\n{"idea": "nod", "recipe": "go .3 p=5"}\n```')
    assert (idea, recipe) == ("nod", "go .3 p=5")


def test_spoken_seconds():
    assert 1.5 < spoken_seconds("今天天氣真的很好耶") < 2.5
    assert 1.0 < spoken_seconds("this is a short sentence") < 3.0


class FakePlanner:
    def __init__(self):
        self.lines = []

    def plan(self, line, heard=None, **kw):
        self.lines.append(line)
        return Gesture("go .3 p=5 | hold .3", "nod", "planner", line)


def test_director_cuts_clauses_and_schedules():
    played, events = [], []
    pos = [0.0]
    planner = FakePlanner()
    d = Director(play=played.append, planner=planner, time_of=lambda p: time.monotonic() + p - pos[0],
                 position=lambda: pos[0], on_event=events.append, lead_s=0.0)
    for piece in ["哈哈，", "你講的", "真有趣。", "我們下次", "再聊"]:
        d.speech_text(piece)
    d.end_of_turn()
    time.sleep(0.5)
    d.close()
    assert planner.lines == ["哈哈，你講的真有趣。", "我們下次再聊"]
    sources = [e["source"] for e in events if e["type"] == "gesture"]
    assert "reflex" in sources and sources.count("planner") >= 1
    assert len(played) == len(sources)


def test_director_interrupt_drops_pending():
    played = []
    d = Director(play=played.append, planner=None, time_of=lambda p: time.monotonic() + 5.0,
                 position=lambda: 0.0, lead_s=0.0)
    d.speech_text("哈哈好啊")  # reflex scheduled 5 s in the future
    d.interrupt()
    time.sleep(0.3)
    d.close()
    assert played == []


def test_director_skips_tags_punctuation_and_abbreviations():
    planner = FakePlanner()
    d = Director(play=lambda m: None, planner=planner, time_of=lambda p: time.monotonic(), position=lambda: 0.0)
    for piece in ["[溫柔] 當然可以呀", "！", "我是 Dr. Ko 的助理。"]:
        d.speech_text(piece)
    d.end_of_turn()
    time.sleep(0.3)
    d.close()
    assert planner.lines == ["當然可以呀！", "我是 Dr. Ko 的助理。"]


@pytest.mark.parametrize("text,idea", [
    ("They said it was fine", None), ("Let me show you", None), ("你是不是很累？", "question"),
    ("他很勇敢", None), ("不是這樣啦", "no"), ("Hey there!", "greeting"), ("wow, nice", "amazed"),
])
def test_reflex_false_positives(text, idea):
    g = reflex(text)
    assert (g.idea if g else None) == idea


def test_hard_stop_after_digit_and_pm():
    planner = FakePlanner()
    d = Director(play=lambda m: None, planner=planner, time_of=lambda p: time.monotonic(), position=lambda: 0.0)
    for piece in ["I am 25. ", "See you at 3 p.", "m. tomorrow. ", "Pi is 3.14 ok"]:
        d.speech_text(piece)
    d.end_of_turn()
    time.sleep(0.3)
    d.close()
    assert planner.lines == ["I am 25.", "See you at 3 p.m. tomorrow.", "Pi is 3.14 ok"]


def test_time_of_unqueued_position_counts_from_now():
    from reachy_motion.audio_io import Speaker

    spk = Speaker(16000)
    spk.write(np.zeros(1600, np.float32), 16000)  # 0.1 s queued
    time.sleep(0.3)  # ... and already played
    t = spk.time_of(spk.position + 0.5)
    assert t >= time.monotonic() + 0.45
