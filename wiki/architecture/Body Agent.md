---
title: Body Agent
date: 2026-10-09
tags: [architecture, agent, tools, intent]
aliases: [spoken control, intent agent, search_web, look_and_describe]
---

# Body agent (v0.2)

`reachy_motion/intent.py` + `Engine._decide`. The voice model (GPT-Live / ElevenLabs agent / Taiwanese LLM) owns
the conversation; a small **side agent** reads every final user utterance (plus the last few turns) and may call
tools. It never talks — results go back to the current voice, which answers in its own words.

| Tool | Does | Executed by |
|---|---|---|
| `switch_voice_mode(mode)` | gpt-live / elevenlabs / taigi / default (back from 台語) | `Engine.switch_mode` — for 台語 it first probes the services ([[Taigi Local Pipeline]], LAN node over mDNS) and, if missing, has the current voice explain what is missing |
| `set_volume(up/down/set)` | robot speaker volume via the daemon `/api/volume/set`, remembered across daemon restarts (`~/.config/reachy_motion/state.json`) | `Engine.set_volume` |
| `look_at_person(bool)` | daemon-side YuNet face tracking, weight 0.85 idle / 0.45 while a gesture plays | `Engine._gaze_loop` |
| `look_and_describe(question)` | camera JPEG → vision LLM (`gpt-5.4-mini`, ~2 s) → `voice.answer_from_sight` | `reachy_motion/vision.py` |
| `search_web(query)` | OpenAI Responses API + `web_search` (~4 s) → `voice.answer_from_web` | `reachy_motion/web.py` |

Model: the planner model (`gpt-5.4-nano`, `reasoning_effort="none"`), ~1 s. `commands.parse_command` (regex) is
only the offline fallback when there is no OpenAI key — the user rightly pointed out that keyword rules don't
understand indirect requests.

## Measured `(2026-10-09, on the robot)`
- 12/12 on a test set incl. indirect requests (「我有點聽不清楚」→ louder) and context agreement (robot offers 台語,
  person says 「好啊」→ switch).
- Weather / news / stock lookups worked end to end in GPT-Live mode.

## Misreads found in live use (and the fixes)
- 「你聽得到我講話嗎」 → louder ✗ → rule: questions about the robot *hearing* are not volume.
- 「像講悄悄話一樣講」 → quieter ✗ → rule: speaking style belongs to the voice model.
- 「小聲跟我說一個秘密」 → "default" mode ✗ → `default` only valid while in 台語 (engine guard).
- 「我聽不太懂…到底多少錢」 → louder ✗ (twice, even with a rule) → code guard: "聽不懂" without any loudness word
  never changes volume. Small models need a deterministic safety net for this one.
- A question split across transcript fragments triggered two lookups → GPT-Live utterances end after 1.6 s of
  quiet (or when the robot answers) + identical lookups within 20 s are answered once.

## Not done on purpose
- **No location assumptions** in the public code: `REACHY_MOTION_LOCATION` / `REACHY_MOTION_TIMEZONE` are settings
  (an ASR name-correction tuned for Taipei was reverted at the owner's request).
- General agent work (email, calendar…) belongs to an external agent (OpenClaw), see [[Backlog]].

Related: [[Architecture]], [[Gesture Director]].
