---
title: reachy-animation
date: 2026-10-08
tags: [source, animation, dependency]
aliases: [Animator, SpeechSway]
---

# reachy-animation (dependency)

`pip install git+https://github.com/pham-tuan-binh/reachy-animation@90e25b5` — not on PyPI; core needs only numpy.
Apache-2.0. Pinned by commit in `pyproject.toml`.

## API we use `(repo code)`

- `Animator(fps=60, idle=Breathing(), blend_s=0.4, speech_latency_s=…)` — a drift-free tick thread.
- `.on_pose(cb)` — we call `robot.set_target(*to_target(pose))` (head 4×4, antennas, body_yaw).
- `.play(motion)` — **newest wins**: drops playing/queued motions, crossfades (smoothstep, 0.4 s) into the new one,
  back to idle breathing when it ends. A callable runs on a thread; stale results are dropped by a generation counter.
- `.feed_speech(pcm, sr)` / `.interrupt_speech()` — `SpeechSway`: 10 ms RMS → two envelopes (sway 50/250 ms,
  accent 15/80 ms) → six sinusoids on x/y/z/roll/pitch/yaw + a pitch dip on syllable onsets; chunks queue
  back-to-back, so bursty realtime TTS stays aligned.
- `Clip.from_frames(poses, fps, name)`; `clips.get("cheerful1")` = 104 Pollen emotions/dances (downloaded on use).
- Pose = `x y z (m) | roll pitch yaw (rad) | antenna_right antenna_left (rad) | body_yaw (rad)`.

## Notes

- `reachy_animation.sim.Sim` renders a pose trace in the SDK's MuJoCo model headlessly — `scripts/record_session.py`
  uses it to turn a real voice session into an MP4 without a robot (extra deps: `imageio-ffmpeg`, `pillow`).
- It does **not** project to reachable poses; we clamp in `recipe.py` and the SDK clamps again.

Related: [[Binh Pham Expressive Harness]], [[Architecture]].
