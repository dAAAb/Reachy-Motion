---
title: Latency Measurements
date: 2026-10-08
tags: [finding, latency, measured]
---

# Latency measurements (all `measured`, 2026-10-08, Taipei, M4 Max Mac)

## Gesture planner (cloud, 5 test lines zh/台語/en, `response_format=json_object`)

| Model | Median | Notes |
|---|---|---|
| `gpt-4.1-nano` | ~1.0 s | valid recipes |
| `gpt-5.4-nano` (`reasoning_effort="none"`) | ~1.2 s | **default** — best quality/latency balance |
| `gpt-5.4-mini` (`none`) | ~1.3 s | |
| `gpt-4.1-mini` | ~1.8 s | |
| `gpt-5.6-luna` (`none`) | ~2.0 s | |

gpt-5.1+ reject `reasoning_effort="minimal"` (400) — they take `none`; gpt-5 / -mini / -nano take `minimal`.
Network round trip dominates; see [[Backlog]] for a local MLX planner.

## Voice modes (end-to-end runs, `scripts/record_session.py`)

| Mode | Text vs its audio | Reply start after user stops | Gesture placement |
|---|---|---|---|
| [[GPT-Live-1]] | transcript ~0.3 s **after** audio | ~0 s (instant) | reaction at reply start (planned while user talks); clause gestures ≤ ~1.8 s late |
| [[ElevenLabs Agents]] | alignment arrives with audio, audio **faster than real time** | ~1–2 s | on the clause |
| [[Taigi Local Pipeline]] | sentence known **before** TTS | 1.7 s warm (9 s cold) | on the sentence (planned during TTS) |

Reflex gestures: 0 ms (regex on the first text of a clause).

Related: [[Playback Clock]], [[Gesture Director]].
