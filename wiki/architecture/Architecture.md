---
title: Architecture
date: 2026-10-08
tags: [architecture, reachy-motion]
---

# Architecture

```
            ┌────────────── voice mode (one of three) ──────────────┐
 mic ──────▶│ GPT-Live-1 │ ElevenLabs Agent │ Taigi: ASR→LLM→TTS     │
            └──────┬────────────────────┬───────────────────────────┘
          speech audio             speech text (+ where it lands)
                   ▼                    ▼
            ┌────────────┐   pos   ┌──────────┐  reflex / reaction / planner
            │  Speaker   │◀───────▶│ Director │──────────────┐
            │ (clock)    │ time_of └──────────┘              ▼
            └─────┬──────┘                          recipe → 25 Hz poses → Clip
       on_chunk   │ audio out                                │
                  ▼                                          ▼
            Animator.feed_speech ──────────────▶  Animator (60 Hz: idle + crossfade + speech sway)
                                                             │ on_pose
                                                             ▼
                                                  robot.set_target(head, antennas, body_yaw)
```

| Piece | File | Role |
|---|---|---|
| Voice modes | `reachy_motion/voices/*.py` | talk to an API / local services; push audio to the speaker and text to the director |
| Speaker / mic | `reachy_motion/audio_io.py` | local (sounddevice) or robot (`reachy_mini.media`); the [[Playback Clock]] |
| Director | `reachy_motion/motion/director.py` | see [[Gesture Director]] |
| Planner | `reachy_motion/motion/planner.py` | reflex lexicon + LLM recipe planner |
| Recipes | `reachy_motion/motion/recipe.py` | see [[Motion Recipes]] |
| Animator | `reachy_animation` (dependency) | see [[reachy-animation]] |
| Engine | `reachy_motion/engine.py` | wires one mode + clock + director + animator, switchable at runtime |
| App / CLI | `reachy_motion/main.py` | `ReachyMiniApp` for the dashboard; `reachy-motion` CLI; settings page on :8042 |

## Two deployment shapes

1. **On the robot** (dashboard app). Robot mic + speaker, full duplex by default. Works for [[GPT-Live-1]] and
   [[ElevenLabs Agents]] (cloud). The Taiwanese services live on the Mac, so the robot would need their URLs on the
   LAN — see [[Taigi Local Pipeline]].
2. **On the Mac, robot over the network** (`reachy-motion --audio local`). Mac mic + speaker, robot only moves
   (`media_backend="no_media"`). This is the natural shape for the Taiwanese mode, and the "the robot must be linked to
   this computer" answer. Robot-speaker-from-Mac is in the [[Backlog]] (the SDK's WebRTC media is Linux-first).

## Echo

A laptop mic hears the laptop speaker. Local audio defaults to **half duplex** (mic sent as silence while voiced
audio plays + 0.35 s tail). Only *voiced* audio counts — see [[Gotchas]] (GPT-Live streams silence continuously).

Related: [[Gesture Director]], [[Latency Measurements]].
