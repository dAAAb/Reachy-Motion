---
title: Gesture Director
date: 2026-10-08
tags: [architecture, gestures, director]
---

# Gesture director

`reachy_motion/motion/director.py`. Voice modes call it from any thread:

| Call | Meaning |
|---|---|
| `speech_text(delta, pos)` | streamed robot text; `pos` = where it lands on the [[Playback Clock]] |
| `speech_line(line, pos_or_future)` | a whole sentence (Taiwanese TTS path) |
| `user_text(text)` / `anticipate(partial)` / `listening()` | the person's words; plan a reaction; lean in |
| `end_of_turn()` / `interrupt()` | flush the last clause / barge-in (drops everything pending) |

## Clauses

Text is cut at hard stops (`。！？!?` newline, `.` not after `Dr|Mr|Mrs|Ms|St|vs` or digits) or at soft stops
(`，、；：…`) once ≥ 6 chars, or forcibly at 36 chars. Punctuation-only fragments are glued, not planned.
ElevenLabs v3/v4 audio tags like `[溫柔]` / `[laughs]` are stripped (they are not spoken).

## Four gesture tiers

| Tier | Latency | Source | When |
|---|---|---|---|
| **listening** | 0 | fixed recipe (lean in, ears perk) | the person starts talking, if nothing else plays |
| **reflex** | 0 | regex lexicon: laugh / wow / no / yes / sorry / question / thanks / greeting (zh + Taiwanese + en) | first text of a clause |
| **reaction** | planned while the person talks | `planner.react(heard)` | first clause of a reply, if no reflex fired — built for [[GPT-Live-1]] ([[Decision - Anticipatory Reactions for GPT-Live]]) |
| **planner** | ~1–1.3 s (cloud LLM) | `planner.plan(clause, heard)` → recipe | every clause |

Every gesture is scheduled at `time_of(pos) - lead (0.1 s)` and **dropped** if it would start after
`pos + max(1 s, spoken_seconds(clause) + late_grace)`; `late_grace` = 0.4 s by default, 1.8 s for GPT-Live (text
arrives after audio; one reply usually keeps one mood). Newer gestures crossfade over older ones
(`Animator.play`, newest wins).

`spoken_seconds`: ~4.5 CJK chars/s, ~2.6 English words/s (rough, used for positions inside a chunk and deadlines).

Related: [[Motion Recipes]], [[Latency Measurements]].
