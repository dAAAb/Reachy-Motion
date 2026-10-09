"""The robot's eyes: grab a camera frame and let a vision LLM describe it, to answer "what do you see?".

Reachy Mini has no on-board scene understanding (its built-in vision is YuNet face detection for tracking). Like
Pollen's conversation app ``camera`` tool, we take a JPEG with ``media.get_frame_jpeg()`` and send it to a cloud
vision model; the description is then handed to the current voice model, which answers in its own voice.
"""

from __future__ import annotations

import base64
import logging
import time

logger = logging.getLogger(__name__)

PROMPT = ("你是一台桌上型機器人 Reachy Mini 的眼睛（頭上的廣角攝影機，畫面就是機器人正在看的方向）。"
          "根據這張畫面，用繁體中文、兩三句話，具體回答使用者的問題；看不清楚就老實說。"
          "描述人物時只講外觀、動作、手上的東西與周遭，不要猜測身分。")


class Eyes:
    def __init__(self, robot, model: str = "gpt-5.4-mini", api_key: str | None = None, timeout_s: float = 15.0):
        from openai import OpenAI

        self.robot, self.model = robot, model
        self.client = OpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)

    def snapshot(self) -> bytes | None:
        media = getattr(self.robot, "media", None)
        if media is None:
            return None
        for _ in range(10):  # the first frames after start can be empty
            jpg = media.get_frame_jpeg()
            if jpg:
                return jpg
            time.sleep(0.1)
        return None

    def describe(self, question: str) -> str | None:
        jpg = self.snapshot()
        if jpg is None:
            logger.warning("no camera frame available")
            return None
        t0 = time.perf_counter()
        kwargs: dict = dict(model=self.model, messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"使用者問：{question}"},
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode(), "detail": "low"}},
            ]},
        ])
        if self.model.startswith(("gpt-5.", "gpt-6")):
            kwargs["reasoning_effort"] = "none"
        r = self.client.chat.completions.create(**kwargs)
        text = (r.choices[0].message.content or "").strip()
        logger.info("vision %.0f ms (%d KB): %s", (time.perf_counter() - t0) * 1000, len(jpg) // 1024, text[:80])
        return text
