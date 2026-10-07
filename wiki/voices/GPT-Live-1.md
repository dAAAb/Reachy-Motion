---
title: GPT-Live-1
date: 2026-10-08
tags: [voice, openai, gpt-live, realtime]
aliases: [gpt-live-1, OpenAI Live API]
---

# GPT-Live-1 (mode `gpt-live`)

OpenAI's full-duplex voice model, in the API since 2026-09-10. Implementation: `reachy_motion/voices/gpt_live.py`.

## Protocol facts `(official doc)`

- WebSocket `wss://api.openai.com/v1/live/sessions`, header `Authorization: Bearer …`, **no query params**.
  First message **must** be `session.start` with a nested `session`; wait for `session.started`.
- Python: `openai>=3.12` (`client.live.connect()`; not `client.realtime`). We use 3.26.
- `session`: `model: "gpt-live-1"`, `instructions` (≤ 16k tokens, immutable), `audio.format`
  `{"type":"audio/pcm","rate":24000}` (default) or 16000 / pcmu / pcma, `audio.output.voice` (default `marin`,
  immutable), `delegation` (`client` or `responses` with a backend model + `function`/`web_search` tools).
- Client → server: `session.input_audio.append {audio: b64}`, `…mute/unmute`, `session.instructions.append`,
  `session.thinking.append`, `session.commentary.append`, `session.update` (Responses settings only), `session.close`.
- Server → client: `session.output_audio.delta`, `session.output_transcript.delta {delta, start_ms, end_ms}`,
  `session.input_transcript.delta`, `session.delegation.created`, `response.event`, `session.usage.updated`,
  `session.closed {reason}`, `error`, `info`.
- **No** turn-detection config, **no** speech_started / response.done / interrupted events, **no** `response.cancel`:
  the model owns turn-taking; the client controls playback.
- Price: $0.05 / min of session (billed per second, silence included) + backend model usage. Concurrent-session
  limits by tier (Free tier not supported).
- Models available on the user's key on 2026-10-08 included `gpt-live-1`, `gpt-live-transcribe`,
  `gpt-realtime-2.1` `(measured: /v1/models)`.

Sources: developers.openai.com/api/docs/guides/live, …/voice-websockets?api=live, …/live-delegation,
…/live-migration, …/models/gpt-live-1, openai.com/index/introducing-gpt-live-1-in-the-api/.

## How it behaves `(measured, 2026-10-08)`

- **Audio streams continuously from session start, silence included** — first audio delta ~0.7 s after
  `session.started` even with nobody talking.
- **Output transcript arrives ~0.25–0.35 s *after* its audio**; `start_ms` is on the session timeline and maps
  onto the received audio stream (offset = time of first audio delta). Text never arrives ahead of audio.
- **Replies start essentially instantly** — the first robot word arrived in the same 10 ms as the user's last
  transcript delta. It can also start answering during a pause mid-question.
- Input transcript of Mandarin came back in Simplified unless the user's speech was clearly Traditional context;
  the reply follows the persona (Traditional Chinese).

## Consequences for gestures

1. Clause-level LLM gestures are always ≥ 1.3 s late → `late_grace_s = 1.8` for this mode.
2. The first beat of a reply comes from a **reaction planned while the person is talking**
   ([[Decision - Anticipatory Reactions for GPT-Live]]) or a reflex word (哇/耶/對…).
3. Echo gating must look at *voiced* audio only ([[Gotchas]]).

Related: [[Playback Clock]], [[Latency Measurements]].
