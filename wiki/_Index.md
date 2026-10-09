---
title: Reachy-Motion Knowledge Base — Index
date: 2026-10-08
tags: [moc, index, reachy-mini, reachy-motion]
aliases: [Reachy-Motion MOC, Reachy-Motion Wiki]
---

# Reachy-Motion knowledge base

An Obsidian-compatible linked wiki for this repo: research, decisions and measured numbers behind
*live voice chat where Reachy Mini moves with what it says*. Open the `wiki/` folder as a vault, or read it on GitHub.

> **Provenance tags** used throughout: `(official doc)`, `(repo code)`, `(article)`, `(model card)`,
> `(measured)` = we ran it on 2026-10-08, `(inference)` = our reasoning, not verified.

## Map of content

### Architecture (how this repo works)
- [[Architecture]] — the pipeline end to end: voice mode → speaker clock → director → animator → robot
- [[Playback Clock]] — stream positions ↔ wall-clock, why every gesture is scheduled against the speaker
- [[Gesture Director]] — clause cutting, reflexes, reactions, planner, deadlines
- [[Motion Recipes]] — the `go / hold / osc` language and how we render it without the flow generator
- [[Body Agent]] — 🆕 v0.2: an LLM with tools for spoken control (mode, volume, gaze), camera vision and web lookups
- [[Running on the Robot]] — 🆕 install on Reachy Mini Wireless, config, shared apps venv, daemon gotchas

### Voice modes
- [[GPT-Live-1]] — OpenAI full-duplex voice model: protocol facts + measured timing behaviour
- [[ElevenLabs Agents]] — Agents WebSocket, Eleven v4 Turbo, alignment, client events
- [[Taigi Local Pipeline]] — 台語: Breeze-ASR-26 → SARC-Taigi-LLM → KaedeTai GPT-SoVITS, all local

### Sources (what we learned from)
- [[Binh Pham Expressive Harness]] — the 2026-10-02 article + `reachy-motion-generator` / `reachy-animation` / `reachy-explain`
- [[reachy-animation]] — the animator we depend on (idle, crossfade, speech sway)

### Findings & decisions
- [[Latency Measurements]] — every number we measured, in one place
- [[Gotchas]] — things that broke and why
- [[Code Review 2026-10-08]] — 16 review findings and their fixes
- [[Decision - Recipe Path without the Flow Generator]]
- [[Decision - Raw WebSocket for ElevenLabs]]
- [[Decision - Anticipatory Reactions for GPT-Live]]
- [[Backlog]] — what is next (local MLX motion server, robot speaker from a Mac, …)

## Three sentences for the next person

1. Gestures must be scheduled on the **speaker's playback clock**, not on "when the text arrived" — each voice API
   delivers text at a different time relative to its audio (see [[Playback Clock]]).
2. LLM planning costs ~1–1.3 s per clause (cloud). Hide it with **reflexes** (0 ms), **text that arrives ahead of
   audio** (ElevenLabs, Taiwanese TTS) or **anticipation from the user's words** (GPT-Live).
3. The motion language is Binh Pham's recipe DSL (Apache-2.0); a local MLX port of his fine-tuned planner +
   flow-matching generator is the obvious upgrade ([[Backlog]]).
