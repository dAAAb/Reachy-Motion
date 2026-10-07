---
title: Backlog
date: 2026-10-08
tags: [backlog, roadmap]
---

# Backlog

## Next
- [ ] **Real-robot acceptance** (robot was offline on 2026-10-08): all three modes on a Reachy Mini Wireless;
      verify echo behaviour of the robot mic in full duplex, speech-sway latency (`speech_latency_s`), and that
      `set_target` at 60 Hz over Wi-Fi is smooth.
- [ ] **Local motion server on Apple Silicon** — implement Binh's `/generate-dense` contract on the Mac:
      planner 0.8B/4B via MLX (`mlx_lm.convert -q`, check the `Qwen3_5ForConditionalGeneration` weight layout),
      flow generator on MPS (`generator.sample.load(ckpt, "mps")`), `Reach.project`. Needs an English
      "word. context." prompt from the Chinese line (one cheap labelling call, or the reaction planner's `idea`).
      See [[Decision - Recipe Path without the Flow Generator]].
- [ ] **Taiwanese mode on the robot**: bind the ASR / Ollama / KaedeTai services to the LAN and point
      `REACHY_MOTION_TAIGI_*_URL` at the Mac, or run Reachy-Motion on the Mac and play audio through the robot's
      speaker (SDK WebRTC media from macOS is not supported yet → investigate a small audio relay).
- [ ] Taiwanese ASR: Breeze-ASR-26 returns Mandarin Han; try a Taiwanese-Han output model when one exists.

## Later
- [ ] Use Pollen's 104 library clips as extra "reflexes" (laughing1, yes1, no1, surprised1 …) mixed with recipes.
- [ ] Face tracking (from the sibling Look-At-Me app) as a base layer under gestures while listening.
- [ ] Barge-in for local audio with real AEC (macOS voice-processing I/O) instead of half duplex.
- [ ] GPT-Live sideband (`/v1/live/sessions/{id}/attach`) for audio with timestamps, if latency allows.
- [ ] Publish as a Hugging Face Space for one-click install from the Reachy Mini dashboard.

## Done (v0.1, 2026-10-08)
- [x] Three voice modes, headless end-to-end runs recorded and rendered in MuJoCo.
- [x] Director with reflex / reaction / planner tiers on a playback clock.
- [x] This wiki.
