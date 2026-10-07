---
title: "Decision - Recipe Path without the Flow Generator"
date: 2026-10-08
tags: [decision, motion]
status: accepted (v0.1)
---

# Decision: LLM-written recipes, rendered without the flow-matching generator (v0.1)

**Context.** Binh Pham's stack = fine-tuned planner (0.8B–27B) + flow-matching generator; its server is CUDA-only;
nothing is hosted; this machine is an M4 Max ([[Binh Pham Expressive Harness]]).

**Options.**
1. Port the server to Apple Silicon (MLX planner + MPS generator) — best fidelity, most work, English-only prompts.
2. Use a cloud LLM as the "teacher" planner writing the same recipe DSL, render recipes ourselves.
3. Library clips only (Pollen emotions by keyword).

**Decision.** (2) for v0.1, with (3)-style reflexes for zero latency. The recipe language and validator are
identical to Binh's, so recipes stay compatible with his generator when we add it.

**Consequences.**
- Motion is less "alive" than the generator's learned detail — energy is band-limited noise
  ([[Motion Recipes]]).
- Planner latency ~1.2 s per clause from the cloud ([[Latency Measurements]]), hidden by the director's tiers.
- Works directly on Chinese / Taiwanese / English lines (the fine-tuned planners are English-only).
- Upgrade path in [[Backlog]]: local `/generate-dense`-compatible server on the Mac.
