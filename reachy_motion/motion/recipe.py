"""Motion recipes -> playable 9-DoF clips.

The recipe language (``go`` / ``hold`` / ``osc``), its channels, limits and the ``expand`` routine are vendored
from Binh Pham's reachy-motion-generator (Apache-2.0, ``planner/dsl.py`` + ``common/plan.py`` @ 96497d6), see
NOTICE. What is ours: turning the expanded posture into a full pose trajectory without the flow-matching
generator — the ``E`` (energy) channel is rendered as band-limited noise instead of learned detail.

A recipe is a sequence of segments separated by ``|``::

    go D k=v ...           cosine-ease to the target over D seconds
    hold D [E=v]           stay in the current pose (optionally changing energy)
    osc D ch amp per [k=v] sinusoid on channel ``ch`` (amplitude ``amp``, period ``per`` s) around the pose

Channels: ``e`` (both ears) ``eR`` ``eL`` ``p`` (pitch, + = head down) ``r`` (roll) ``y`` (yaw) ``z`` (height, mm)
``b`` (body yaw); ``E`` is the energy (RMS of fast detail, deg). Every recipe starts from ``NEUTRAL``.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

FPS = 25
KEYMAP = {"eR": 0, "eL": 1, "p": 2, "r": 3, "y": 4, "z": 5, "b": 6, "E": 7}
NEUTRAL = np.array([15.0, 15.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.5])
LIMITS = {
    "e": (-25, 175), "eR": (-25, 175), "eL": (-25, 175), "p": (-30, 30), "r": (-25, 25), "y": (-50, 50),
    "z": (-25, 25), "b": (-60, 60), "E": (0, 12),
}
MAX_SECONDS = 30.0


class RecipeError(ValueError):
    pass


def _kv(toks: list[str], tgt: np.ndarray, amp: float) -> None:
    for tk in toks:
        if "=" not in tk:
            raise RecipeError(f"expected key=value, got {tk!r}")
        k, v = tk.split("=", 1)
        if k not in LIMITS:
            raise RecipeError(f"unknown channel {k!r} (use e eR eL p r y z b E)")
        try:
            v = float(v)
        except ValueError:
            raise RecipeError(f"bad number in {tk!r}") from None
        lo, hi = LIMITS[k]
        if not lo <= v <= hi:
            raise RecipeError(f"{k}={v} outside [{lo}, {hi}]")
        if k in ("p", "r", "y", "z", "b"):
            v *= amp
        if k == "e":
            tgt[0] = tgt[1] = v
        else:
            tgt[KEYMAP[k]] = v


def expand(recipe: str, rng: np.random.Generator | None = None, amp: float = 1.0, tsc: float = 1.0) -> np.ndarray:
    """Recipe -> (T, 8) per-frame ``[earR earL pitch roll yaw z body energy]`` at ``FPS`` (plan units).

    ``amp`` scales head/body targets, ``tsc`` the tempo; ``rng`` adds ±15% per-segment timing jitter.
    """
    rng = rng or np.random.default_rng(0)
    cur = NEUTRAL.copy()
    out = [cur.copy()]
    segs = [s.split() for s in recipe.split("|") if s.strip()]
    if not segs:
        raise RecipeError("empty recipe")
    for tok in segs:
        cmd = tok[0]
        if cmd not in ("go", "hold", "osc") or len(tok) < 2:
            raise RecipeError(f"bad segment {' '.join(tok)!r}")
        try:
            d = float(tok[1])
        except ValueError:
            raise RecipeError(f"bad duration in {' '.join(tok)!r}") from None
        if not 0.05 <= d <= 10:
            raise RecipeError(f"duration {d} outside [0.05, 10] s")
        d *= tsc * rng.uniform(0.85, 1.15)
        n = max(1, int(round(d * FPS)))
        if cmd in ("go", "hold"):
            tgt = cur.copy()
            _kv(tok[2:], tgt, amp)
            if cmd == "hold":
                tgt[:7] = cur[:7]
            w = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / n)
            seg = cur + w[:, None] * (tgt - cur)
        else:
            if len(tok) < 5:
                raise RecipeError(f"osc needs: osc D ch amp period, got {' '.join(tok)!r}")
            ch = tok[2]
            if ch not in ("e", "eR", "eL", "p", "r", "y", "z", "b"):
                raise RecipeError(f"osc on unknown channel {ch!r}")
            try:
                a, per = float(tok[3]), float(tok[4])
            except ValueError:
                raise RecipeError(f"bad number in {' '.join(tok)!r}") from None
            if per < 0.3:
                raise RecipeError("osc period below 0.3 s: fast shaking belongs in E (energy), not osc")
            a, per = a * amp * rng.uniform(0.8, 1.2), per * rng.uniform(0.85, 1.15)
            tgt = cur.copy()
            _kv(tok[5:], tgt, amp)
            u = np.arange(1, n + 1) / FPS
            s = np.sin(2 * np.pi * u / per) * np.minimum(1, np.minimum(u, u[-1] - u + 1 / FPS) / 0.3)
            seg = np.repeat(cur[None], n, 0)
            seg[:, 7] = np.linspace(cur[7], tgt[7], n)
            for j in [0, 1] if ch == "e" else [KEYMAP[ch]]:
                seg[:, j] += a * s
        out += list(seg)
        cur = seg[-1].copy()
    frames = np.array(out)
    if len(frames) / FPS > MAX_SECONDS:
        raise RecipeError(f"recipe lasts {len(frames) / FPS:.1f} s; keep it under {MAX_SECONDS:.0f} s")
    return frames


def check(recipe: str) -> str | None:
    """None if the recipe is valid, else the error message."""
    try:
        expand(recipe)
        return None
    except (RecipeError, ValueError, IndexError) as e:
        return str(e)


# ---------------------------------------------------------------------------------------------------------------
# Posture -> pose (ours). Pose layout matches reachy_animation.pose.DOF:
#   x y z (m) | roll pitch yaw (rad) | antenna_right antenna_left (rad) | body_yaw (rad)
# Inverse of reachy-motion-generator ``common.plan.posture``: earR = -deg(ant_r), earL = deg(ant_l).
# ---------------------------------------------------------------------------------------------------------------

# Hard safety envelope applied after noise (a bit inside the SDK limits; the SDK clamps too).
SAFE = {"pitch": 0.52, "roll": 0.44, "yaw": 0.87, "z": 0.025, "body": 1.05, "ant": 3.05}


def _detail_noise(n: int, rng: np.random.Generator, lo: float = 1.0, hi: float = 4.0) -> np.ndarray:
    """Unit-RMS band-limited noise (lo..hi Hz) — stands in for the generator's learned fast detail."""
    if n < 20:
        return np.zeros(n)
    sos = butter(2, [lo / (FPS / 2), hi / (FPS / 2)], btype="band", output="sos")
    x = sosfiltfilt(sos, rng.standard_normal(n + 40))[20:-20]
    rms = float(np.sqrt(np.mean(x**2))) or 1.0
    return x / rms


def posture_to_poses(frames: np.ndarray, rng: np.random.Generator | None = None, detail: float = 1.0) -> np.ndarray:
    """(T, 8) plan-unit frames -> (T, 9) robot poses, rendering ``E`` as fast detail on ears/pitch/roll/yaw."""
    rng = rng or np.random.default_rng()
    T = len(frames)
    ear_r, ear_l, pitch, roll, yaw, z, body, energy = (frames[:, i].copy() for i in range(8))
    if detail > 0:
        e = energy * detail
        pitch += e * _detail_noise(T, rng) * 0.6
        roll += e * _detail_noise(T, rng) * 0.5
        yaw += e * _detail_noise(T, rng) * 0.6
        ear_r += e * _detail_noise(T, rng, 1.5, 6.0) * 2.0  # ears flick more than the head
        ear_l += e * _detail_noise(T, rng, 1.5, 6.0) * 2.0
    rad = np.radians
    poses = np.zeros((T, 9))
    poses[:, 2] = np.clip(z / 1000.0, -SAFE["z"], SAFE["z"])
    poses[:, 3] = np.clip(rad(roll), -SAFE["roll"], SAFE["roll"])
    poses[:, 4] = np.clip(rad(pitch), -SAFE["pitch"], SAFE["pitch"])
    poses[:, 5] = np.clip(rad(yaw), -SAFE["yaw"], SAFE["yaw"])
    poses[:, 6] = np.clip(-rad(ear_r), -SAFE["ant"], SAFE["ant"])
    poses[:, 7] = np.clip(rad(ear_l), -SAFE["ant"], SAFE["ant"])
    poses[:, 8] = np.clip(rad(body), -SAFE["body"], SAFE["body"])
    return poses


def recipe_to_poses(
    recipe: str, seed: int | None = None, amp: float = 1.0, tempo: float = 1.0, detail: float = 1.0
) -> np.ndarray:
    """Recipe -> (T, 9) pose trajectory at ``FPS``; raises ``RecipeError`` on an invalid recipe."""
    rng = np.random.default_rng(seed)
    return posture_to_poses(expand(recipe, rng, amp=amp, tsc=tempo), rng, detail=detail)
