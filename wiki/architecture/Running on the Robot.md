---
title: Running on the Robot
date: 2026-10-09
tags: [deployment, robot, wireless]
---

# Running on the robot (v0.2, accepted on a Reachy Mini Wireless 2026-10-09)

Robot: ReachyMiniOS v0.3.2 (reflashed), daemon 1.11.0. Everything runs on the robot except the Taiwanese models.

```bash
# on the robot (ssh pollen@reachy-mini.local)
/venvs/apps_venv/bin/pip install "git+https://github.com/dAAAb/Reachy-Motion"
mkdir -p ~/.config/reachy_motion && chmod 700 ~/.config/reachy_motion
# ~/.config/reachy_motion/.env  (chmod 600): OPENAI_API_KEY, ELEVENLABS_AGENT_ID, [ELEVENLABS_API_KEY],
#   REACHY_MOTION_MODE=elevenlabs, REACHY_MOTION_AUDIO=robot, REACHY_MOTION_TIMEZONE=..., REACHY_MOTION_LOCATION=...
```
Start / stop it from the dashboard (app `reachy_motion`) or `POST :8000/api/apps/start-app/reachy_motion`.
Settings page: `http://reachy-mini.local:8042`.

## Measured
- 60 Hz `set_target` stream: 60.1 Hz, 0 errors, median call 0.34 ms (`scripts/robot_motion_test.py`, over Wi-Fi).
- Robot mic + speaker full duplex: no self-hearing observed; barge-in works (local, see [[Gotchas]]).
- Gesture planner from the robot: 1.3–2.4 s per clause (a bit slower than from the Mac).

## Gotchas specific to the robot
- **Shared apps venv**: Pollen's conversation app pins `openai==2.28.0`. Reachy-Motion therefore speaks the GPT-Live
  and ElevenLabs WebSockets directly and only needs `openai>=1.50` (chat + responses); installing adds 4 packages
  and upgrades nothing shared.
- **Boots asleep**: motors disabled until `enable_motors()`; ease with `goto_target` before streaming
  (reachy_mini#1430).
- **Volume resets to 62** whenever the daemon restarts → we re-apply the last spoken volume at start.
- **Daemon stuck in `stopping`** after a slow app shutdown (the process was gone): only
  `sudo systemctl restart reachy-mini-daemon` cleared it. The app now bounds its shutdown to 6 s.
- **Streaming resampler returns empty chunks** sometimes; GPT-Live rejects empty `input_audio.append` with
  `invalid_audio` (and floods errors). Never send empty audio.
- `reachyminios_check` reports "IMU not found" when run over a non-login SSH shell: `/usr/sbin/i2cdetect` is not on
  PATH. The IMU answers at 0x18 / 0x69 on bus 4.
- Robot SSH default `pollen` / `root` (official docs).

Related: [[Architecture]], [[Body Agent]], [[Taigi Local Pipeline]].
