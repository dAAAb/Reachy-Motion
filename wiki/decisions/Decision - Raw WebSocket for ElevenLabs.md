---
title: "Decision - Raw WebSocket for ElevenLabs"
date: 2026-10-08
tags: [decision, elevenlabs]
status: accepted
---

# Decision: speak the ElevenLabs Agents WebSocket directly

**Context.** `elevenlabs.conversational_ai.Conversation` (SDK 2.71) handles the protocol, but
`conversation_initiation_metadata` is consumed internally — the negotiated `agent_output_audio_format` is never
surfaced, and `DefaultAudioInterface` hard-codes 16 kHz `(repo code)`. We also need each audio chunk's alignment
tied to *our* speaker position.

**Decision.** ~150 lines of `websockets` in `voices/elevenlabs.py`: read formats from the metadata, resample both
ways, place alignment text on the [[Playback Clock]], answer pings, drop audio older than the last `interruption`
event id, reply to client tool calls with a no-op.

**Consequences.** One less beta dependency on the hot path (the SDK is still installed for future use), but we own
protocol drift — re-check the event list in [[ElevenLabs Agents]] when ElevenLabs ships SDK 3.x.
