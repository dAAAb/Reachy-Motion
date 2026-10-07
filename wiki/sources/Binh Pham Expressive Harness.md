---
title: Binh Pham Expressive Harness
date: 2026-10-08
tags: [source, motion-generation, flow-matching, planner, reachy-mini]
aliases: [reachy-motion-generator, Building the most expressive robot harness so far]
---

# Binh Pham — "Building the most expressive robot harness so far" (2026-10-02)

Article: https://garden.binhph.am/articles/the-best-expressive-harness-for-robots
Repos (all read at the commits noted; only the Apache-2.0 ones are reused here):

| Repo | License | What it is |
|---|---|---|
| `pham-tuan-binh/reachy-motion-generator` @96497d6 | Apache-2.0 | planner LLMs, flow-matching generator, inference server, training |
| `pham-tuan-binh/reachy-motion-generator-api` | Apache-2.0 | stdlib client for the server |
| `pham-tuan-binh/reachy-animation` @90e25b5 | Apache-2.0 | real-time animator — our dependency, see [[reachy-animation]] |
| `pham-tuan-binh/reachy-explain` | **no license** (all rights reserved) | offline talk/slide harness — ideas only, **no code copied** |

## Pipeline `(article + repo code)`

prompt → **planner LLM** writes `{"idea", "recipe"}` → recipe expanded into randomised **plans** (keys every
0.25 s when serving) → **flow-matching generator** (21.8 M params, d=384, 8 blocks, non-causal over ≤ 720 frames,
8 Euler steps, CFG 1.5, 4 Hz low-pass) → reachability projection → SDK recorded-move JSON at 25 Hz, 9-DoF.

- Planners: Qwen3.5-0.8B / Qwen3.5-4B / Qwen3.8-27B fine-tunes (`binhpham/reachy-mini-motion-planner-{0.8b,4b,27b}`),
  distilled from frontier "teacher" models writing recipes. OOD probe quality 0.66 / 0.91 / 0.96 `(model card)`.
- Latency on an RTX PRO 6000: ~170 / 290 / 790 ms total; the generator itself ~20 ms after optimisation
  (FP8, speculative decoding, MTP heads); planner decoding ≈ 90 % of the time `(article, model card)`.
- Training data: Pollen's 85 emotions + 19 dances (~9 min real motion) + 10,000 synthetic examples;
  released as *Reachy Mini Massive Motion Library* (10,872 episodes, ~15 h, LeRobot format) `(article)`.
- The server (`inference/server.py`: `POST /generate-dense`, `/generate-sparse`) is **CUDA-only** as written (vLLM,
  hard-coded `"cuda"`); `Planner(backend="hf")` and `generator.sample.load()` do run on MPS/CPU `(repo code)`.
- **No hosted endpoint**: the HF Spaces are a static viewer and a robot app that both need your own server.

## reachy-explain timing idea `(repo code, re-implemented, not copied)`

Pre-rendered talks: TTS with word timestamps (Inworld), `{beat}` markers in the script, each cue fires at the next
word's start + offset; the **sound card is the master clock** (`outputBufferDacTime`) and the animator gets a
`speech_latency_s` equal to the stream latency. Our [[Playback Clock]] is the live-streaming version of that idea.

## What we took / didn't

- **Took:** the recipe DSL + validator + `expand` (vendored in `motion/recipe.py`), the teacher-prompt channel
  description and examples (adapted in `motion/prompt.py`), `reachy-animation` as a dependency. See NOTICE.
- **Didn't (yet):** the fine-tuned planners and the flow generator — no CUDA here, and the cloud "teacher" path
  already writes good recipes in ~1.2 s. See [[Decision - Recipe Path without the Flow Generator]] and [[Backlog]].
- Prompts for the fine-tuned planners are English-only ("word. one sentence of context.") — a Chinese/Taiwanese
  conversation would need a labelling step first `(repo docs)`.

Related: [[Motion Recipes]], [[reachy-animation]].
