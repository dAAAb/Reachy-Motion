---
title: Backlog
date: 2026-10-08
tags: [backlog, roadmap]
---

# Backlog

## Next
- [ ] **External agent for real work** (email via BaseMail, calendar…): keep voice in front, delegate to OpenClaw —
      GPT-Live `delegation: client` → `session.delegation.created` → OpenClaw gateway → `commentary.append`;
      spoken confirmation + recipient allow-list for side effects. (Discussed 2026-10-09, not started.)
- [ ] Measure `speech_latency_s` / `RobotSpeaker.latency_s` on the robot (sway vs audio alignment).
- [ ] Anticipatory reactions for ElevenLabs too (first-clause planner gestures are often late on the robot).
- [ ] **Local motion server on Apple Silicon** — implement Binh's `/generate-dense` contract on the Mac:
      planner 0.8B/4B via MLX (`mlx_lm.convert -q`, check the `Qwen3_5ForConditionalGeneration` weight layout),
      flow generator on MPS (`generator.sample.load(ckpt, "mps")`), `Reach.project`. Needs an English
      "word. context." prompt from the Chinese line (one cheap labelling call, or the reaction planner's `idea`).
      See [[Decision - Recipe Path without the Flow Generator]].
- [ ] Taiwanese ASR: Breeze-ASR-26 returns Mandarin Han; try a Taiwanese-Han output model when one exists.

## Later
- [ ] Use Pollen's 104 library clips as extra "reflexes" (laughing1, yes1, no1, surprised1 …) mixed with recipes.
- [ ] Barge-in for local audio with real AEC (macOS voice-processing I/O) instead of half duplex.
- [ ] GPT-Live sideband (`/v1/live/sessions/{id}/attach`) for audio with timestamps, if latency allows.
- [ ] Publish as a Hugging Face Space for one-click install from the Reachy Mini dashboard.

## Done (v0.2, 2026-10-09)
- [x] Real-robot acceptance on a Wireless (reflashed to ReachyMiniOS v0.3.2, daemon 1.11.0); runs fully on the robot.
- [x] Body agent: spoken mode switch / volume / gaze, camera vision, web lookups ([[Body Agent]]).
- [x] Taiwanese services found on the LAN (`reachy-motion-node`, mDNS) + standalone `reachy-motion-asr`.
- [x] Look at the person (daemon face tracking blended under gestures); embodiment; tone tags as gesture hints;
      ElevenLabs local barge-in; current date/time for every voice.

## Done (v0.1, 2026-10-08)
- [x] Three voice modes, headless end-to-end runs recorded and rendered in MuJoCo.
- [x] Director with reflex / reaction / planner tiers on a playback clock.
- [x] This wiki.
