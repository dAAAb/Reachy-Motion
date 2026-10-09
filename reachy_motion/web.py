"""Live knowledge for every voice: answer time-sensitive questions with OpenAI's web search (Responses API).

The ElevenLabs agent and the Taiwanese LLM cannot browse; the body agent calls ``search_web`` and the result is
handed to the current voice to answer in its own words (same pattern as the camera, see ``vision.py``).
"""

from __future__ import annotations

import logging
import time

from reachy_motion.config import now_context

logger = logging.getLogger(__name__)


class WebLookup:
    def __init__(self, model: str = "gpt-5.4-mini", api_key: str | None = None, timeout_s: float = 25.0,
                 location: str = "") -> None:
        from openai import OpenAI

        self.model, self.location = model, location  # e.g. "台灣台北" (REACHY_MOTION_LOCATION); empty = unknown
        self.client = OpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)

    def answer(self, question: str) -> str | None:
        t0 = time.perf_counter()
        try:
            r = self.client.responses.create(
                model=self.model,
                tools=[{"type": "web_search"}],
                instructions=("用網路搜尋回答。這段話會被機器人唸出來：繁體中文、口語、一兩句，只講最重要的那個答案"
                              "（例如股價只講現在價格和漲跌），不要時區換算、不要網址、不要列點。"
                              + (f"使用者所在地：{self.location}。" if self.location else "")
                              + now_context()),
                input=question,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("web lookup failed: %s", e)
            return None
        text = (getattr(r, "output_text", "") or "").strip()
        text = __import__("re").sub(r"[*_`#]+|\[([^\]]*)\]\([^)]*\)", lambda m: m.group(1) or "", text)  # speakable
        logger.info("web lookup %.0f ms: %r -> %s", (time.perf_counter() - t0) * 1000, question, text[:80])
        return text or None
