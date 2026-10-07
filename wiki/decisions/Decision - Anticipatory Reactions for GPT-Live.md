---
title: "Decision - Anticipatory Reactions for GPT-Live"
date: 2026-10-08
tags: [decision, gpt-live, gestures]
status: accepted
---

# Decision: plan the reaction while the person is still talking

**Context.** With [[GPT-Live-1]] the reply starts the instant the person stops, and the robot's own transcript
arrives ~0.3 s after its audio. A clause-level planner (~1.3 s) can therefore never be on time for the first beat
`(measured)`.

**Decision.** The user's *input* transcript streams in real time. Every ~6 new characters the director asks the
planner for a **reaction** ("Person said: …; the robot is about to answer — write its immediate body-language
reaction"). When the first robot text of a reply arrives, a reflex word wins if present (哇/對/哈哈…), otherwise the
latest reaction plays at the reply's position. Clause gestures follow with `late_grace_s = 1.8`.

**Result** `(measured, e2e run)`: reaction gesture started at 9.31 s, the robot's first word 「耶」 at 9.46 s;
later clause gestures all played.

**Cost.** ~1 extra planner call per 6 characters the user speaks (small; `gpt-5.4-nano`).

Related: [[Gesture Director]], [[Latency Measurements]].
