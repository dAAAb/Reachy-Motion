---
title: Playback Clock
date: 2026-10-08
tags: [architecture, timing, audio]
---

# Playback clock

Every chunk written to the `Speaker` gets a **stream position** (seconds of audio queued since the stream started).
`time_of(pos)` converts a position into the `time.monotonic()` instant it will be heard:

```
queued_until = max(queued_until, now + device_latency) + chunk_duration      # on write
time_of(pos) = queued_until - (pos_end - pos)
```

Why: the three voice APIs deliver text at very different moments relative to the audio it describes
([[Latency Measurements]]):

| Mode | When text arrives vs its audio | Position we use |
|---|---|---|
| [[ElevenLabs Agents]] | per audio chunk, with char alignment; audio itself streams **faster than real time** | `write()` position of the chunk + first char's `char_start_times_ms` |
| [[GPT-Live-1]] | transcript ~0.3 s **after** the audio; audio streams continuously, silence included | `stream_base + start_ms/1000`, `stream_base` calibrated at the first audio delta |
| [[Taigi Local Pipeline]] | we have the sentence **before** TTS | a `Future` resolved with the `write()` position once the sentence is synthesised |

The speaker also hands every chunk to `Animator.feed_speech` when it is queued; the animator's `SpeechSway` keeps its
own back-to-back timeline, so the head sway follows the same clock (see [[reachy-animation]]).

**Barge-in** (`flush`): queued audio is dropped and `queued_until` jumps to *now*; positions stay monotonic so later
mappings still hold. The director's turn counter invalidates scheduled gestures.

**Echo gating** uses `_voice_until`, updated only for chunks whose peak > ~-40 dBFS, so continuous silence doesn't
keep the mic muted forever — the bug that made GPT-Live hear only "Hi" in our first run ([[Gotchas]]).

Related: [[Gesture Director]], [[Architecture]].
