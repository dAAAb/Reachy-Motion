---
title: Code Review 2026-10-08
date: 2026-10-08
tags: [finding, review, bugs]
---

# Adversarial code review — v0.1 (2026-10-08)

An independent review pass over the whole package before the first push. 16 findings; all fixed except where
noted. Regression tests were added for the ones marked 🧪.

| # | Finding | Fix |
|---|---|---|
| 1 🧪 | `time_of(pos)` for text that arrives **before** its audio counted from a stale end-of-queue → gestures dropped or early | positions beyond the queue count from `max(queued_until, now + latency)` |
| 2 | mode switch: `stop()` gave up after 5 s while a Taiwanese turn (LLM + TTS) kept running → two sessions; the old one's cleanup could stop the shared robot audio pipeline | turn loop checks `stop`; stop waits (20 s) before starting the next; `running_mode` only cleared by its own session |
| 3 | `start`/`stop` not serialised (double-click → two sessions, one unstoppable) | lifecycle lock |
| 4 | `/api/state` iterated the event deque while voice threads appended ("deque mutated during iteration") | copy first |
| 5 | energy VAD stuck "active" under steady noise above the initial floor | minimum-statistics floor (quietest block of the last 3 s) + 0.6 s warm-up |
| 6 | `speaker.start()` outside `try` → silent thread death | moved inside |
| 7 | TTS errors other than HTTP (non-WAV body) left a `Future` unresolved, blocking planner workers 30 s | catch everything, always resolve |
| 8 | Taiwanese mode answered the **oldest** queued utterance | answer the latest |
| 9 | per-chunk resampling → clicks at chunk edges and length drift | `StreamResampler` (soxr `ResampleStream`) per stream |
| 10 | robot speaker: no latency estimate, `flush` didn't clear the robot's player | `latency_s=0.15` (to be measured, [[Backlog]]), `clear_player()` on flush |
| 11 | Taiwanese mode defaulted to full duplex on the robot (no server-side echo handling) | per-mode `default_half_duplex = True` |
| 12 🧪 | reflex false positives: "they" → hey, "show" → how, 「你是不是很累」 → no, 勇敢 → 敢 | word boundaries, questions checked before "no", look-behinds |
| 13 | `tts_text` replaced names inside words ("drink" → 博士ink) | whole Latin words only |
| 14 🧪 | no sentence cut after a digit ("I am 25."), streamed "p." + "m." cut mid-word | `.` must be followed by whitespace; `a.m./p.m.` excluded |
| 15 | gestures tagged with the turn at push time, so a planner result racing `interrupt()` survived | carry the planning turn into `_schedule` |
| 16 | app mode read `REACHY_MOTION_AUDIO` before `.env` was loaded; unprefixed env fallbacks (`MODE`, `AUDIO`…) | load `.env` first; only well-known names (`OPENAI_API_KEY`, `ELEVENLABS_*`) unprefixed |

Also found while testing (not by the review): Ctrl+C didn't stop the CLI — an imported library swallows SIGINT —
fixed with explicit `SIGINT`/`SIGTERM` handlers ([[Gotchas]]).

Known risk, not a defect: the [[GPT-Live-1]] transcript→audio mapping assumes the server streams audio without gaps
(observed, not documented).
