---
title: Taigi Local Pipeline
date: 2026-10-08
tags: [voice, taiwanese, taigi, hokkien, local, mlx]
aliases: [台語, Taiwanese mode, Hokkien pipeline]
---

# 台語 local pipeline (mode `taigi`)

Implementation: `reachy_motion/voices/taigi.py`. It reuses the local services from the AIRI NTU-VH2026 speech lab
(a fork of moeru-ai/airi, `course/ntu-vh2026/`). Everything runs on an Apple-Silicon Mac; no cloud.

```
mic ─energy VAD─▶ Breeze-ASR-26 (MLX 4-bit) ─▶ SARC-Taigi-LLM-12b (Ollama, streamed)
      ─sentence─▶ tts_text() ─▶ KaedeTai GPT-SoVITS (32 kHz WAV) ─▶ speaker
                └──────────────▶ director.speech_line(sentence, Future[pos])
```

| Stage | Model / service | Endpoint (default) | Notes |
|---|---|---|---|
| ASR | `RayyTien/Breeze-ASR-26-mlx-4bit` (MediaTek Breeze-ASR-26) | `POST :18001/v1/audio/transcriptions` (OpenAI-compatible) | segments ≤ 120 s; **outputs Mandarin Han characters**, not Taiwanese |
| LLM | `hf.co/Speech-AI-Research-Center/SARC-Taigi-LLM-12b-GGUF:Q4_K_M` (Gemma-3 12B fine-tune) | Ollama `:11434/v1/chat/completions` | writes 台語漢字; completion only (no tools) |
| TTS | KaedeTai `gpt-sovits-tw` (S1 trilingual + S2 r4 e15) | `POST :8883/v1/audio/speech` model `taigi-hanzi` | Han → Taibun POJ inside; ≤ 60 Han chars; single-flight; CPU |

Ports: the packaged "AIRI Local" desktop app uses 18001/18880/18883/18884; the research launchers use
8001/8880/8883/8884. The mode tries the other port set automatically if the configured service is down.
Override with `REACHY_MOTION_TAIGI_{ASR,LLM,TTS}_URL`.

## Facts that matter `(repo code / AIRI lab notes)`

- Taibun (inside KaedeTai) **rejects Latin letters and digits** (HTTP 422) → `tts_text()` maps names
  (`Reachy Mini → 瑞奇迷你`), spells numbers in Han (`100 → 一百`), drops leftover Latin. Found the hard way
  ([[Gotchas]]).
- Taibun transliterates; it does **not** translate Mandarin grammar to Taiwanese — the LLM must write Taiwanese.
- mlx-audio 0.4.3 has a tokenizer bug (`<|nocaptions|>` vs `<|nospeech|>`) that the lab's server patches in memory.
- Licenses: KaedeTai code MIT (RVC-Boss upstream), weights card MIT; Meta MMS `mms-tts-nan` is CC-BY-NC (not used
  here). We do **not** redistribute any weights or reference audio.

## Measured `(2026-10-08, M4 Max)`

| Step | Cold | Warm |
|---|---|---|
| ASR (≈3 s utterance) | 0.9 s | 0.4 s |
| end of speech → first reply audio | 9.2 s (SARC model loading + a 422) | **0.9–1.7 s** |
| KaedeTai, short sentence | — | ~0.8–1.1 s |

Gestures are planned **in parallel with TTS** and bound to the sentence's audio position via a `Future`, so they
start with the audio (the first planner gesture landed within ~0.1 s of the first audio).

## Running it

1. Start the services (from the AIRI lab): ASR-26, Ollama with the SARC model, KaedeTai (`start-server.sh`).
2. `reachy-motion --mode taigi --audio local` on the Mac, robot over Wi-Fi (or `--no-robot` to preview).
3. On the robot instead: expose the three services on the LAN (bind 0.0.0.0) and point the URLs at the Mac —
   [[Backlog]].

Related: [[Architecture]], [[Latency Measurements]].
