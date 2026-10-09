"""Text -> gesture planning.

Two tiers, so the robot never stands frozen waiting for a model:

* **reflex** — a tiny lexicon of strong cue words (yes / no / wow / laughing / sorry ...) mapped straight to a
  recipe. Zero latency, used the instant a clause starts.
* **planner** — an LLM writes a full motion recipe for the whole clause (the "teacher" path of
  reachy-motion-generator: same recipe language, rendered without the flow-matching generator).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from reachy_motion.motion import prompt
from reachy_motion.motion.recipe import check

logger = logging.getLogger(__name__)


@dataclass
class Gesture:
    recipe: str
    idea: str = ""
    source: str = "planner"  # planner | reflex | fallback
    line: str = ""
    latency_ms: float = 0.0
    meta: dict = field(default_factory=dict)


# Ordered: the first matching pattern wins. Patterns are matched against the lowercased clause.
REFLEXES: list[tuple[str, str, str]] = [
    (r"哈哈|呵呵|嘻嘻|\bhaha|\blol\b|\bhehe|好笑", "laughing",
     "go .25 e=-10 p=-10 z=10 E=4 | osc 1.4 z 4 .4 E=6 | go .5 p=-2 z=5 E=2"),
    (r"^(哇|欸|咦)|^(wow|whoa)\b|太厲害|太棒|好厲害|\bamazing\b|\bawesome\b|真的假的", "amazed",
     "go .2 e=-15 p=-12 z=14 E=5 | hold .6 E=2 | go .6 p=-4 z=8 E=1.5"),
    # questions before "no": 「你是不是很累」 is a question, not a refusal
    (r"[?？]\s*$|[嗎呢乎]\s*[?？]?\s*$|(?<!勇)敢|是不是|^(what|why|how|which|who|where|when)\b", "question",
     "go .45 e=5 r=14 p=-5 z=5 E=1.2 | hold .9 E=.8 | go .5 r=6 E=1"),
    (r"不行|不要|(?<!是)不是|不對|毋是|毋通|^no\b|\bnope\b", "no",
     "go .3 e=35 p=3 E=1.5 | osc 1.6 y 14 .65 E=2 | go .4 y=0 e=25 E=1"),
    (r"^(對|是的|沒錯|好啊|好的|當然|嗯嗯|著|是啦)|^(yes|yeah|yep|sure|of course|right)\b|同意", "yes",
     "go .3 e=0 p=-3 z=5 E=2 | osc 1.4 p 7 .55 E=2 | go .4 p=0 E=1"),
    (r"對不起|抱歉|歹勢|不好意思|\bsorry\b|\bmy bad\b", "sorry",
     "go .6 e=120 p=12 z=-8 E=.8 | hold .8 E=.5 | go .6 e=60 p=5 z=-2 E=1"),
    (r"謝謝|感謝|多謝|\bthank", "thanks",
     "go .4 e=20 p=10 z=-2 E=1 | go .5 p=-2 z=5 e=5 E=1.5 | hold .5"),
    (r"嗨|哈囉|你好|您好|\bhello\b|^hi\b|^hey\b|早安|午安|晚安|再見|拜拜|\bbye\b", "greeting",
     "go .3 e=-10 r=10 z=8 p=-4 E=2 | osc 1.2 eR 40 .6 E=2 | go .5 r=0 z=4 E=1"),
]
_REFLEX_RE = [(re.compile(p, re.IGNORECASE), name, recipe) for p, name, recipe in REFLEXES]


def reflex(clause: str) -> Gesture | None:
    text = clause.strip().lower()
    for rx, name, recipe in _REFLEX_RE:
        if rx.search(text):
            return Gesture(recipe=recipe, idea=name, source="reflex", line=clause)
    return None


TALK_FALLBACK = "go .4 e=10 p=-3 z=5 r=6 E=1.5 | osc 1.6 r 6 .9 E=1.8 | go .5 r=0 E=1"


class GesturePlanner:
    """Ask an LLM (OpenAI chat API) for a motion recipe that fits a spoken line."""

    def __init__(self, model: str = "gpt-4.1-mini", api_key: str | None = None, timeout_s: float = 6.0) -> None:
        from openai import OpenAI  # imported lazily: the dashboard loads apps' modules

        self.model = model
        self.client = OpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)

    def _ask(self, line: str | None, heard: str | None, error: str | None, tone: str | None = None) -> tuple[str, str]:
        kwargs: dict = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt.SYSTEM},
                {"role": "user", "content": prompt.user_message(line, heard, error, tone)},
            ],
            response_format={"type": "json_object"},
        )
        if self.model.startswith(("gpt-5", "gpt-6", "o")):
            # gpt-5 / -mini / -nano only go down to "minimal"; gpt-5.1+ accept "none" (no reasoning = lowest latency)
            legacy = self.model.split("-2")[0] in ("gpt-5", "gpt-5-mini", "gpt-5-nano")
            kwargs["reasoning_effort"] = "minimal" if legacy else ("low" if self.model.startswith("o") else "none")
        else:
            kwargs["temperature"] = 0.8
            kwargs["max_tokens"] = 220
        r = self.client.chat.completions.create(**kwargs)
        return prompt.parse_reply(r.choices[0].message.content or "")

    def react(self, heard: str) -> Gesture | None:
        """Anticipatory gesture: how the robot reacts to what the person said, before its reply is known."""
        t0 = time.perf_counter()
        try:
            idea, recipe = self._ask(None, heard, None)
        except Exception as e:  # noqa: BLE001
            logger.warning("reaction planning failed: %s", e)
            return None
        if check(recipe) is not None:
            return None
        return Gesture(recipe, idea, "reaction", heard, (time.perf_counter() - t0) * 1000)

    def plan(self, line: str, heard: str | None = None, retries: int = 1, tone: str | None = None) -> Gesture:
        t0 = time.perf_counter()
        error = None
        for _ in range(retries + 1):
            try:
                idea, recipe = self._ask(line, heard, error, tone)
            except Exception as e:  # network / JSON problems -> fall back below
                logger.warning("planner call failed: %s", e)
                break
            error = check(recipe)
            if error is None:
                return Gesture(recipe, idea, "planner", line, (time.perf_counter() - t0) * 1000)
            logger.info("planner wrote an invalid recipe (%s): %s", error, recipe)
        fb = reflex(line) or Gesture(TALK_FALLBACK, "talking", "fallback", line)
        fb.source = "fallback"
        fb.latency_ms = (time.perf_counter() - t0) * 1000
        return fb
