---
title: ElevenLabs Agents
date: 2026-10-08
tags: [voice, elevenlabs, agents, eleven-v4]
aliases: [ElevenAgents, Eleven v4 Turbo]
---

# ElevenLabs Agents (mode `elevenlabs`)

Implementation: `reachy_motion/voices/elevenlabs.py` (raw WebSocket — see
[[Decision - Raw WebSocket for ElevenLabs]]).

## Protocol facts `(official doc)`

- `wss://api.elevenlabs.io/v1/convai/conversation?agent_id=…`. A **public** agent needs no key; a private one needs
  `GET /v1/convai/conversation/get-signed-url?agent_id=…` with `xi-api-key` (we do this whenever a key is set).
- Client → server: `conversation_initiation_client_data` (optional `conversation_config_override`), raw
  `{"user_audio_chunk": "<b64 pcm>"}` (no `type`!), `pong`, `client_tool_result`, `contextual_update`, `user_message`.
- Server → client: `conversation_initiation_metadata` (negotiated `agent_output_audio_format` /
  `user_input_audio_format`), `audio` (`audio_base_64`, `event_id`, `alignment {chars, char_start_times_ms,
  char_durations_ms}`), `agent_response`, `agent_response_correction`, `agent_chat_response_part`,
  `user_transcript`, `tentative_user_transcript`, `interruption {event_id}`, `ping`, `client_tool_call`, …
- Some events are **only sent if enabled** in the agent's `conversation.client_events` (dashboard → Advanced):
  `user_transcript`, `tentative_user_transcript`, `agent_response_correction`, `agent_chat_response_part`, …
- Audio formats are set **per agent** (`pcm_8000…pcm_48000`, `ulaw_8000`), not overridable per session.
- Session overrides (`tts.voice_id`, `tts.model_id`, prompt, first message, language) are **off by default** and must be
  enabled per field in the agent's Security tab.
- TTS models for agents include `eleven_v4` and `eleven_v4_turbo`; v4 Turbo is the real-time one (~100 ms median
  inference) and the default for new agents. Python SDK `elevenlabs==2.71.0` (3.0 is alpha).

Sources: elevenlabs.io/docs/agents-platform/api-reference/agents-platform/websocket,
…/customization/events/client-events, …/customization/personalization/overrides, elevenlabs.io/agents/v4-turbo.

## Our agent `(measured, 2026-10-08)`

- Already configured with `eleven_v4_turbo`, the owner's cloned voice, `pcm_16000` in/out, Scribe realtime ASR —
  so no session overrides are needed (`REACHY_MOTION_ELEVENLABS_OVERRIDE` stays off).
- `client_events` originally had only `audio, interruption, agent_response`; we added `user_transcript,
  tentative_user_transcript, agent_response_correction` (config backed up locally first). We deliberately did
  **not** enable `agent_chat_response_part`: alignment already arrives early and carries exact timing.
- `agent_response` (full text) arrives with the **first** audio chunk of a turn, not the last → it must not close
  the turn ([[Gotchas]]).
- Audio streams **faster than real time**, each chunk with alignment, so text is known seconds before it is
  heard — the planner has time and gestures land on their clause.
- The agent's LLM emits v4 audio tags in its text (`[gentle] 當然可以啊，[slow] …`); alignment chars exclude them.

Related: [[Playback Clock]], [[Latency Measurements]].
