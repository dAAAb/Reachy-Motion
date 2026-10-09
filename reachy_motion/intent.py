"""The body agent: an LLM with tools decides whether what the person said asks the robot to change something.

The voice model (GPT-Live / ElevenLabs agent / Taiwanese LLM) owns the conversation. This small side agent reads
each final utterance plus a little context and may call tools — switch voice mode, change volume, look at the person.
It understands indirect requests ("我有點聽不清楚" → louder, 「轉過來看我」 → look) that keyword rules miss.
``commands.parse_command`` (regex) is only the offline fallback when no OpenAI key is configured.
"""

from __future__ import annotations

import json
import logging
import time

from reachy_motion.commands import Command

logger = logging.getLogger(__name__)

TOOLS = [
    {"type": "function", "function": {
        "name": "switch_voice_mode",
        "description": "Switch how the robot talks: 'gpt-live' (OpenAI GPT-Live voice), 'elevenlabs' (the owner's "
                       "cloned voice agent), 'taigi' (Taiwanese Hokkien / 台語, needs local services), or 'default' "
                       "(ONLY when the robot is currently in 'taigi' and the person asks to go back to Mandarin).",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["mode"],
                       "properties": {"mode": {"type": "string",
                                               "enum": ["gpt-live", "elevenlabs", "taigi", "default"]}}}}},
    {"type": "function", "function": {
        "name": "set_volume",
        "description": "Change the robot's speaker volume. Use 'up'/'down' for relative requests (including "
                       "indirect ones like 'I can't hear you' or 'too loud'), 'set' with level 0-100 for an "
                       "explicit level.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["change"],
                       "properties": {"change": {"type": "string", "enum": ["up", "down", "set"]},
                                      "level": {"type": "integer", "minimum": 0, "maximum": 100}}}}},
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Look up live / time-sensitive information on the web: weather, news, prices, scores, "
                       "opening hours, recent events. Not for general knowledge or chit-chat.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["query"],
                       "properties": {"query": {"type": "string",
                                                "description": "the question to look up, self-contained"}}}}},
    {"type": "function", "function": {
        "name": "look_and_describe",
        "description": "Take a picture with the robot's camera and look, to answer a question about what is in front "
                       "of it: what it sees, what the person is holding or wearing, how they look, what is around.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["question"],
                       "properties": {"question": {"type": "string",
                                                   "description": "what to look for, in the person's words"}}}}},
    {"type": "function", "function": {
        "name": "look_at_person",
        "description": "Turn face tracking on (keep looking at the person with the camera) or off (stop staring).",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["enabled"],
                       "properties": {"enabled": {"type": "boolean"}}}}},
]

SYSTEM = """You control the BODY settings of Reachy Mini, a small desk robot with a moving head, two antenna ears,
a camera, a microphone and a speaker. A separate voice assistant holds the conversation; you never talk.
For the person's latest utterance (Mandarin, Taiwanese written in Mandarin characters, or English), decide whether
it asks the robot to change one of its settings, directly or indirectly, and if so call the matching tool(s).
- Only act on requests addressed to the robot. Mentioning a topic is not a request ("GPT 是什麼？" -> nothing,
  「今天台語課很好玩」 -> nothing, 「你看我今天穿得好看嗎」 -> look_at_person(true) is fine since it asks to be seen).
- Volume is about the ROBOT'S SPEAKER being too quiet/loud for the person. Questions about whether the robot can
  hear the person (its microphone) are NOT volume requests: 「你聽得到我嗎」「你聽得到我講話嗎」「有聽到嗎」
  「哈囉，聽得到嗎」"can you hear me?" -> do nothing.
- Asking ABOUT the volume is not a request to change it: 「這樣已經是最大聲了嗎？」「現在音量多少」 -> do nothing.
- Speaking STYLE is the voice model's job, not volume: 「像講悄悄話一樣講」「小聲跟我說個秘密」"whisper to me",
  「用興奮的語氣講」 -> do nothing.
- Indirect requests count: 「我聽不太清楚」/"what? I can't hear you" -> set_volume(up); 「有點吵」/"too loud" ->
  set_volume(down); 「轉過來看我」/"face me" -> look_at_person(true); 「不要一直盯著我」 -> look_at_person(false);
  「你會講台語嗎？講給我聽」/"talk to me in Taiwanese" -> switch_voice_mode(taigi).
- Agreement to the robot's own offer counts if the context shows the offer (robot: "要我切換成台語嗎？" person:
  「好啊」 -> switch_voice_mode(taigi)).
- Questions about what the robot can SEE need its camera: 「你看到什麼」「我手上拿的是什麼」「我今天穿這樣好看嗎」
  「你看我比什麼手勢」"what am I holding?" -> look_and_describe(question). (Turning to face the person is
  look_at_person; describing what is visible is look_and_describe; both can apply.)
- Live information needs the web: 「今天台北天氣如何」「最新新聞」「台積電股價」"who won last night" ->
  search_web(query). Today's date / the time is already known to the robot -> do nothing.
- The transcript comes from speech recognition and often garbles names (homophones, Simplified characters).
  Write the search query in Traditional Chinese with the most likely intended entity, using the conversation and
  Taiwan context: 「蒋安安 市政发表会」 -> 「台北市長蔣萬安 市政發表會」. Merge fragments of the same question.
- Do not switch to the mode that is already active. When in doubt, do nothing."""


def _cmd_from_call(name: str, args: dict, text: str) -> Command | None:
    if name == "switch_voice_mode" and args.get("mode"):
        return Command("mode", args["mode"], text)
    if name == "set_volume":
        ch = args.get("change")
        if ch == "set" and isinstance(args.get("level"), int):
            return Command("volume", max(0, min(100, args["level"])), text)
        if ch in ("up", "down"):
            return Command("volume", "+" if ch == "up" else "-", text)
    if name == "search_web" and args.get("query"):
        return Command("web", str(args["query"]), text)
    if name == "look_and_describe" and args.get("question"):
        return Command("look", str(args["question"]), text)
    if name == "look_at_person" and "enabled" in args:
        return Command("gaze", bool(args["enabled"]), text)
    return None


class BodyAgent:
    def __init__(self, model: str = "gpt-5.4-nano", api_key: str | None = None, timeout_s: float = 6.0) -> None:
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)

    def decide(self, utterance: str, state: dict, recent: list[tuple[str, str]] | None = None) -> list[Command]:
        """Return the commands the utterance asks for (usually none). ``recent`` = [(role, text)] context."""
        ctx = "\n".join(f"{role}: {t}" for role, t in (recent or [])[-4:])
        user = (f"Robot state: {json.dumps(state, ensure_ascii=False)}\n"
                + (f"Recent conversation:\n{ctx}\n" if ctx else "")
                + f"Person's latest utterance: {utterance}")
        kwargs: dict = dict(model=self.model, tools=TOOLS, tool_choice="auto",
                            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        if self.model.startswith(("gpt-5", "gpt-6")):
            legacy = self.model.split("-2")[0] in ("gpt-5", "gpt-5-mini", "gpt-5-nano")
            kwargs["reasoning_effort"] = "minimal" if legacy else "none"
        t0 = time.perf_counter()
        r = self.client.chat.completions.create(**kwargs)
        calls = r.choices[0].message.tool_calls or []
        out = []
        for c in calls:
            try:
                cmd = _cmd_from_call(c.function.name, json.loads(c.function.arguments or "{}"), utterance)
            except json.JSONDecodeError:
                continue
            if cmd is not None:
                out.append(cmd)
        logger.info("body agent %.0f ms: %r -> %s", (time.perf_counter() - t0) * 1000, utterance,
                    [(c.kind, c.arg) for c in out])
        return out
