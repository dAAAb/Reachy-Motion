---
title: Gotchas
date: 2026-10-08
tags: [finding, gotchas, bugs]
---

# Gotchas (things that actually broke)

1. **GPT-Live streams silence continuously** → a "speaker is busy" check based on queued audio is *always* true →
   half-duplex gating muted the mic forever; the model heard only "Hi". Fix: gate on **voiced** audio only
   (`Speaker._voice_until`, peak > ~-40 dBFS). See [[GPT-Live-1]].
2. **ElevenLabs `agent_response` arrives at the *start* of a turn**, with the first audio chunk. Treating it as
   end-of-turn flushed half-sentences into the planner (and a lone `！` became a "startled" gesture). Fix: end the turn
   after 0.8 s without new aligned text; never plan punctuation-only fragments. See [[ElevenLabs Agents]].
3. **Eleven v4 audio tags** (`[gentle]`, `[溫柔]`) appear in agent text; strip them before planning.
4. **`Dr.` is not a sentence end** — the hard-stop regex skips common abbreviations.
5. **KaedeTai/Taibun returns 422 on Latin letters or digits** (e.g. the LLM saying "Reachy Mini" or "100分") →
   `tts_text()` maps names, spells numbers in Han, drops other Latin. See [[Taigi Local Pipeline]].
6. **Re-entrant lock**: `speech_text → _try_reflex → _schedule` takes the director lock twice — a plain `Lock`
   deadlocked the first test run. It's an `RLock`.
7. **Clause gestures for GPT-Live were all dropped as late** — its text arrives after the audio and planning takes
   ~1.3 s. Fix: per-mode `late_grace_s` + anticipatory reactions ([[Decision - Anticipatory Reactions for GPT-Live]]).
8. **ElevenLabs SDK hides the negotiated audio format**; an agent on pcm_24000/44100 would play at the wrong speed
   with a 16 kHz interface ([[Decision - Raw WebSocket for ElevenLabs]]).
9. `reachy-mini-app-assistant create --template conversation` fails without **git-lfs** installed
   (`git-lfs filter-process: command not found`). We built the app by hand from the documented contract instead.
10. ElevenLabs `client_events` gate several useful events (user transcript, corrections, streaming text) — if they
    are missing, nothing errors; you just never get them.
11. **Ctrl+C was swallowed**: with the reachy_mini / uvicorn stack imported, SIGINT never became a
    `KeyboardInterrupt` in the CLI's main thread. Install explicit `SIGINT`/`SIGTERM` handlers.
12. **AIRI lab service ports differ** between the research launchers (8001 / 8883) and the packaged desktop app
    (18001 / 18883); the Taiwanese mode falls back to the other port set if the configured one is down. The research
    ASR venv may lack PyAV (`ModuleNotFoundError: av` → HTTP 500 on transcription).

13. **ElevenLabs does not treat talking over playback as an interruption**: its audio arrives much faster than real
    time and the server considers the turn done once sent. Barge in locally on `tentative_user_transcript` while the
    speaker is busy, with echo rejection (ignore text that overlaps the reply being spoken), and drop the rest of
    the old reply's audio until the next `agent_response`.
14. **Voice models deny having a body** ("我是語音助手，沒辦法做動作") unless told: send the embodiment text
    (ElevenLabs `contextual_update`, GPT-Live instructions). Then they even emit motion tags like `[歪頭]` `[點頭]`.
15. More robot-only gotchas: [[Running on the Robot]]; intent misreads: [[Body Agent]].

Related: [[Latency Measurements]], [[Code Review 2026-10-08]].
