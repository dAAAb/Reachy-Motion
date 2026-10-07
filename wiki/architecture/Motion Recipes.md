---
title: Motion Recipes
date: 2026-10-08
tags: [motion, recipe, dsl]
aliases: [recipe DSL, go hold osc]
---

# Motion recipes

The language is Binh Pham's (Apache-2.0, `planner/dsl.py` in reachy-motion-generator) — see
[[Binh Pham Expressive Harness]]. Segments separated by `|`, starting from neutral `e=15 p=0 r=0 y=0 z=3 b=0 E=0.5`:

```
go D k=v ...        cosine-ease to targets over D s (0.15–0.3 s = a snap)
hold D [E=v]        keep the pose (optionally change energy)
osc D ch amp per    sinusoid on one channel, period ≥ 0.3 s (fast shaking belongs in E)
```

| Key | Meaning | Unit | Validator range |
|---|---|---|---|
| `e` / `eR` / `eL` | ear droop (0 up, -15 perked, 150 drooped) | deg | -25…175 |
| `p` | pitch, **+ = head down** | deg | ±30 |
| `r` / `y` | roll / yaw | deg | ±25 / ±50 |
| `z` | head height | mm | ±25 |
| `b` | body yaw | deg | ±60 |
| `E` | energy = RMS of fast (> 1 Hz) detail | deg | 0…12 |

Durations 0.05–10 s per segment, ≤ 30 s per recipe `(repo code)`.

## How we render it (ours)

`expand()` gives a 25 Hz `(T, 8)` posture. Binh's flow-matching generator would add learned overshoots and flicks;
we instead render `E` as **band-limited noise** (1–4 Hz head, 1.5–6 Hz ears, unit RMS × E) and convert to the
9-DoF pose `x y z roll pitch yaw antenna_r antenna_l body_yaw`:

- `antenna_right = -rad(earR)`, `antenna_left = +rad(earL)` (inverse of `common.plan.posture`) `(repo code)`
- `pitch/roll/yaw/body = rad(...)`, `z = mm / 1000`
- clamped to a safety envelope a bit inside the SDK limits (pitch 30°, roll 25°, yaw 50°, z 25 mm, body 60°)

Example a planner wrote for 「不行啦，那樣太危險了。」 `(measured, gpt-5.4-nano)`:

```
go .25 e=40 p=4 z=2 E=1.5 | osc 1.2 y 12 .55 E=2.2 | osc 1.0 p 4 .6 E=1.8 | go .35 e=25 p=3 z=3 E=1
```

The planner prompt (`motion/prompt.py`) is Binh's teacher prompt adapted to conversation: it receives *the line the
robot is about to say* (any language) plus what the person said, and returns `{"idea", "recipe"}`; invalid recipes are
retried once with the validator's error, then fall back to a reflex or a generic "talking" recipe.

Related: [[Decision - Recipe Path without the Flow Generator]], [[Gesture Director]].
