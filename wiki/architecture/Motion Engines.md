---
title: Motion Engines
date: 2026-10-14
tags: [architecture, motion, flow-matching, mlx, reachy-mini]
aliases: [motion engine, flow engine, local engine, reachy-motion-server]
---

# Motion engines (v0.4)

Three ways to turn a spoken clause into motion, switchable live on the app page (**Motion**) or with
`REACHY_MOTION_MOTION_ENGINE`. Code: `reachy_motion/motion/backends.py`, `flow.py`, `motion_server.py`.

| engine | recipe written by | recipe → motion | per clause `(measured, M4 Max)` |
|---|---|---|---|
| `recipe` | cloud LLM (`gpt-5.4-nano`) | eased keys + band-limited noise ([[Motion Recipes]]) | ~1.2 s |
| `flow` | cloud LLM | Binh Pham's flow-matching generator | ~1.3–1.5 s (+~0.2 s render) |
| `local` | Binh Pham's fine-tuned planner, 0.8B on MLX | the generator | **~0.7–0.85 s**, no cloud |

The recipe language is the same in all three ([[Binh Pham Expressive Harness]]), so any planner feeds any renderer.

## The generator, without torch
- `flow.py` ports `generator/model.py` + `sample.py` to **numpy**: 21.8 M params, d=384, 6 heads, 8 blocks, 8 Euler
  steps, CFG 1.5, 4 Hz low-pass, then `Reach.project` (SDK IK) `(repo code)`.
- `generator.pt` is a torch zip/pickle; `load_checkpoint` reads it with a restricted unpickler (only
  `_rebuild_tensor_v2`, `*Storage`, `OrderedDict`) → no torch anywhere, robot venv included.
- Parity with the torch reference: max |Δ| 1.4e-5 on one forward pass; plans identical to `planner/dsl.variants`
  `(measured)`.
- Speed: first numpy port took 3.3 s per gesture — float64 promotion (a float64 mask) and scipy's `erf` in GELU.
  Fixed with float32 everywhere and an A&S 7.1.26 erf (|err| < 5e-7): **~0.17 s / 4 s gesture** `(measured)`.
- Weights: `hf_hub_download("binhpham/reachy-mini-motion-planner-0.8b", "generator.pt")` (87 MB, Apache-2.0); every
  bundle ships the same file.

## The planner on Apple Silicon
- `mlx_lm.load()` opens the bundle's `planner/` (Qwen3.5-0.8B fine-tune, `Qwen3_5ForConditionalGeneration`) directly
  — no conversion `(measured, mlx-lm 0.32.0)`.
- System prompt must be **verbatim** `COMPACT` from `planner/prompt.py` (= `prompt.STUDENT_SYSTEM`).
- 0.35–0.5 s per recipe after a warm-up generation (the first one is ~1.3 s) `(measured)`.
- It was trained on English `"word. context."` prompts but handles the Chinese clauses directly — 「我今天好累喔」→
  slow sink + drooped ears; 「不行啦，那樣太危險了」→ freeze + shake. Tone tags become `"<tone>. <line>"`; reactions use
  `"listening. Someone says to you: …"`.

## Where it runs
- Robot = Raspberry Pi CM4: too slow for live rendering, no MLX. Run **`reachy-motion-server`** on a Mac
  (`pip install -e ".[local]"`): `/render`, `/plan`, `/generate-dense` (reference-server compatible), advertised as
  `_reachy-motion._tcp` (port 18110). `REACHY_MOTION_MOTION_SERVER=auto` → the robot looks for it when it needs it;
  on a Mac everything runs in-process.
- `Renderer.warm()` pre-renders 3 variants of every reflex recipe (+ listening, talking fallback) so reflexes stay
  0 ms. It times the first render after a warm-up; slower than 0.25 s per motion-second → **reflexes only** use the
  generator and planned gestures fall back to the noise rendering (shown on the app page).
- Planned gestures are rendered in the planning thread, before scheduling ([[Gesture Director]]); any render failure
  (loading, server down) falls back to noise at play time — never blocks speech.

## Server control from the app page (v0.4.2)
- `reachy-motion-server` has a **state**: `off` (no weights, ~0.2–0.4 GB RSS) → `loading` → `on` (~2.3 GB with the
  0.8B planner; loads in ~1.5 s once cached) | `error`. `POST /control {action: start|stop}`; `--standby` starts off.
- The robot app runs a `ServerWatch` thread (mDNS browse when not found, `/health` every 3–8 s) → page panel
  **Mac motion server** with Start / Stop (`GET/POST /api/motion-server`); when the server turns on/off the engine is
  rebuilt onto / off it automatically.
- The page can't start a process on the Mac by itself, so something must be running there: either start it by hand,
  or `--install-login-item` (LaunchAgent `com.reachy-motion.server`, KeepAlive, standby, logs in
  `~/Library/Logs/reachy-motion-server.log`).
- On the robot, `flow` / `local` **without** a server now fall back to the noise rendering instead of rendering on the
  CM4 (pre-rendering 30 reflex variants in-process would hog the CPU the camera and audio need).
- Client: 1 s connect timeout, 30 s backoff after a failure; a server in `off` is not used.

## Quality `(measured)`
Same 19 recipes (from three recorded sessions + "sneezing") rendered both ways: peak head speed 84 vs 88 °/s, ear
travel 65° vs **75°**, mean |jerk| 0.34 vs **0.11** — the generator moves the ears more and is ~3× smoother than
noise. Side-by-side recordings: `recordings/motion_{recipe,flow,local}.mp4`, `motion_compare.mp4` (gitignored).

## Gotchas
- Loading models in a background thread while the main thread creates `OpenAI()` crashed with "partially
  initialized module 'httpx'": openai ≥ 3 looks up a *legacy* httpx in `sys.modules`. Import `httpx` /
  `huggingface_hub` on the main thread before starting loader threads.
- `uv run` inside the repo picks up `pyproject.toml` (and writes `uv.lock`); use `--no-project` for throw-away tools.

Related: [[Decision - Recipe Path without the Flow Generator]], [[Latency Measurements]], [[Backlog]].
