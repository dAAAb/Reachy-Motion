"""Binh Pham's flow-matching motion generator, run with numpy (no torch): recipe -> plan -> learned 25 Hz motion.

Adapted from reachy-motion-generator (Apache-2.0, @96497d6, see NOTICE):

* ``FlowNet.__call__`` is ``generator/model.py`` (``MotionGenerator.forward``) rewritten in numpy;
* ``generate_batch`` is ``generator/sample.py`` (Euler steps from noise, classifier-free guidance, 4 Hz low-pass);
* ``to_plan`` / ``plan_frames`` are ``planner/dsl.py`` ``to_plan`` and ``common/plan.py`` ``frames``;
* ``Reach`` is ``common/reach.py`` (project every frame onto what the Stewart platform can reach).

The weights are the published ``generator.pt`` (the same file ships in every planner bundle on the Hub). It is a
torch checkpoint; ``load_checkpoint`` reads its zip/pickle layout directly, so torch is never needed — the robot's
apps venv doesn't have it.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import threading
import time
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from reachy_motion.motion.recipe import FPS, SAFE, expand

logger = logging.getLogger(__name__)

HUB_REPO = "binhpham/reachy-mini-motion-planner-0.8b"  # smallest bundle; every bundle carries the same generator.pt
CH = ["earR", "earL", "pitch", "roll", "yaw", "z", "body", "energy"]
N_DOF, N_PLAN = 9, 8
# served settings (bundle serve.json): 8 Euler steps, CFG 1.5, recipes expanded with a 2 Hz / 0.25 s plan
STEPS, CFG, EXPAND_FC, EXPAND_KDT = 8, 1.5, 2.0, 0.25


# -- checkpoint ------------------------------------------------------------------------------------------------------
def load_checkpoint(path: str | os.PathLike) -> tuple[dict[str, np.ndarray], dict]:
    """``generator.pt`` -> (state dict as float32 numpy arrays, normalisation stats), without importing torch."""
    zf = zipfile.ZipFile(path)
    root = zf.namelist()[0].split("/")[0]
    dtypes = {"FloatStorage": np.float32, "HalfStorage": np.float16, "BFloat16Storage": None}

    def rebuild(pid, offset, size, stride, *_):
        stype, key = pid
        dt = dtypes.get(stype)
        if dt is None:
            raise ValueError(f"unsupported tensor storage {stype}")
        buf = np.frombuffer(zf.read(f"{root}/data/{key}"), dtype=dt)
        flat = buf[offset:]
        return np.array(np.lib.stride_tricks.as_strided(flat, size, [s * flat.itemsize for s in stride]), np.float32)

    class _Reader(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return rebuild
            if module == "torch" and name.endswith("Storage"):
                return name
            if (module, name) == ("collections", "OrderedDict"):
                return OrderedDict
            raise pickle.UnpicklingError(f"unexpected object in checkpoint: {module}.{name}")

        def persistent_load(self, pid):  # ('storage', storage_type, key, location, numel)
            return (pid[1], pid[2])

    ck = _Reader(zf.open(f"{root}/data.pkl")).load()
    return dict(ck["sd"]), {k: np.asarray(v, dtype=np.float64) for k, v in ck["stats"].items()}


def checkpoint_path() -> str:
    """Local ``generator.pt``: ``REACHY_MOTION_GENERATOR`` if set, else the Hugging Face copy (downloaded once)."""
    p = os.environ.get("REACHY_MOTION_GENERATOR")
    if p:
        return p
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HUB_REPO, "generator.pt")


# -- network ---------------------------------------------------------------------------------------------------------
def _layernorm(x: np.ndarray, w=None, b=None, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    return y if w is None else y * w + b


def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact (erf) GELU as torch.nn.GELU(); erf by Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), ~10x faster
    than scipy's erf on float32 arrays."""
    z = np.abs(x) * np.float32(1 / math.sqrt(2.0))
    t = np.float32(1) / (np.float32(1) + np.float32(0.3275911) * z)
    y = t * (np.float32(0.254829592) + t * (np.float32(-0.284496736) + t * (np.float32(1.421413741)
             + t * (np.float32(-1.453152027) + t * np.float32(1.061405429)))))
    erf = np.copysign(np.float32(1) - y * np.exp(-z * z), x)
    return np.float32(0.5) * x * (np.float32(1) + erf)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class FlowNet:
    """The 21.8 M-parameter flow-matching transformer (d=384, 6 heads, 8 blocks, ≤ 720 frames), in numpy."""

    def __init__(self, sd: dict[str, np.ndarray], heads: int = 6) -> None:
        self.sd, self.heads = sd, heads
        self.layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        self.maxlen = sd["pos"].shape[1]
        self.d = sd["pos"].shape[2]

    def _lin(self, x: np.ndarray, name: str) -> np.ndarray:
        return x @ self.sd[name + ".weight"].T + self.sd[name + ".bias"]

    def __call__(self, x: np.ndarray, t: np.ndarray, plan: np.ndarray, has: np.ndarray, pad: np.ndarray) -> np.ndarray:
        """x (B,T,9) noisy motion; t (B,) flow time (1 = noise); plan (B,T,8); has (B,1,1); pad (B,T) True = padding."""
        sd, B, T, d, H = self.sd, x.shape[0], x.shape[1], self.d, self.heads
        cond = np.concatenate([plan * has, np.broadcast_to(has, (B, T, 1))], -1)
        h = self._lin(np.concatenate([x, cond], -1), "inp") + sd["pos"][:, :T]
        half = 128
        f = np.exp(-math.log(10000) * np.arange(half) / half)
        a = (t[:, None] * f[None] * 1000.0).astype(np.float32)
        c = self._lin(_silu(self._lin(np.concatenate([np.sin(a), np.cos(a)], -1), "temb.0")), "temb.2")
        neg = np.where(pad, np.float32(-1e9), np.float32(0))[:, None, None, :]  # (B,1,1,T) key padding
        for i in range(self.layers):
            p = f"blocks.{i}."
            a1, b1, g1, a2, b2, g2 = np.split(self._lin(_silu(c), p + "ada.1")[:, None, :], 6, -1)
            u = _layernorm(h) * (1 + b1) + a1
            qkv = u @ sd[p + "att.in_proj_weight"].T + sd[p + "att.in_proj_bias"]
            q, k, v = (z.reshape(B, T, H, d // H).transpose(0, 2, 1, 3) for z in np.split(qkv, 3, -1))
            s = q @ k.transpose(0, 1, 3, 2) * np.float32(1 / math.sqrt(d // H)) + neg
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            att = (s @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
            h = h + g1 * self._lin(att, p + "att.out_proj")
            u = _layernorm(h) * (1 + b2) + a2
            h = h + g2 * self._lin(_gelu(self._lin(u, p + "mlp.0")), p + "mlp.2")
        return self._lin(_layernorm(h, sd["nf.weight"], sd["nf.bias"]), "out")


# -- plans -----------------------------------------------------------------------------------------------------------
def _lowpass(x: np.ndarray, fc: float, order: int = 2) -> np.ndarray:
    if len(x) < 16:
        return x.copy()
    b, a = butter(order, fc / (FPS / 2))
    return filtfilt(b, a, x, axis=0, padlen=min(len(x) - 1, 9))


def to_plan(F: np.ndarray, rng: np.random.Generator, mirror: bool = False, ear_jitter: float = 6.0,
            fc: float = EXPAND_FC, kdt: float = EXPAND_KDT) -> dict:
    """Expanded recipe frames (T, 8) -> plan ``{duration, keys: [{t, earR, ..., energy}]}`` (posture low-passed)."""
    F = F.copy()
    F[:, 0] += rng.normal(0, ear_jitter)
    F[:, 1] += rng.normal(0, ear_jitter)  # real clips are never symmetric
    if mirror:
        F[:, [3, 4, 6]] *= -1
        F[:, [0, 1]] = F[:, [1, 0]]
    S = _lowpass(F[:, :7], fc) if fc else F[:, :7]
    T, keys = len(F), []
    for t in np.arange(0, T / FPS + 1e-9, kdt):
        i = min(T - 1, int(round(t * FPS)))
        keys.append(dict(t=round(float(t), 2), **{c: round(float(S[i, k]), 1) for k, c in enumerate(CH[:7])},
                         energy=round(float(max(0, F[i, 7])), 1)))
    return dict(duration=round(T / FPS, 2), keys=keys)


def plan_frames(plan: dict) -> np.ndarray:
    """Plan -> (T, 8) per-frame conditioning, linearly interpolated between keys."""
    T = max(2, int(round(plan["duration"] * FPS)))
    last = {c: 0.0 for c in CH}
    rows = []
    for k in sorted(plan["keys"], key=lambda k: k["t"]):
        for c in CH:
            if c in k:
                last[c] = float(k[c])
        rows.append([k["t"]] + [last[c] for c in CH])
    R, u = np.array(rows), np.arange(T) / FPS
    return np.stack([np.interp(u, R[:, 0], R[:, 1 + j]) for j in range(len(CH))], -1)


def recipe_plans(recipe: str, n: int = 1, seed: int = 0) -> list[dict]:
    """``planner/dsl.py`` ``variants``: amplitude ×0.75–1.25, tempo ×0.8–1.25, odd variants mirrored."""
    rng = np.random.default_rng(seed)
    out = []
    for v in range(n):
        F = expand(recipe, rng, amp=rng.uniform(0.75, 1.25), tsc=rng.uniform(0.8, 1.25))
        out.append(to_plan(F, rng, mirror=bool(v % 2)))
    return out


# -- sampling --------------------------------------------------------------------------------------------------------
def generate_batch(net: FlowNet, stats: dict, plans: list[dict], seeds: list[int] | None = None,
                   steps: int = STEPS, cfg: float = CFG, lowpass_hz: float = 4.0) -> list[np.ndarray]:
    """Plans -> (T_i, 9) trajectories: Euler integration of the flow from noise (t=1) to data (t=0), with
    classifier-free guidance on the plan, then a 4 Hz low-pass. All plans and both guidance branches share one
    forward pass per step."""
    MU, SD, PMU, PSD = stats["MU"], stats["SD"], stats["PMU"], stats["PSD"]
    Ps = [plan_frames(pl)[: net.maxlen] for pl in plans]
    Ts = [len(P) for P in Ps]
    B, T = len(Ps), max(Ts)
    Q = np.zeros((B, T, N_PLAN), np.float32)
    pad = np.ones((B, T), bool)
    x = np.zeros((B, T, N_DOF), np.float32)
    for i, P in enumerate(Ps):
        Q[i, : Ts[i]] = (P - PMU) / PSD
        pad[i, : Ts[i]] = False
        g = np.random.default_rng(int(seeds[i]) if seeds is not None else i)
        x[i, : Ts[i]] = g.standard_normal((Ts[i], N_DOF))
    guided = cfg != 1.0
    has = np.ones((B, 1, 1), np.float32)
    if guided:
        has = np.concatenate([has, np.zeros_like(has)])
        Q, pad = np.concatenate([Q, Q]), np.concatenate([pad, pad])
    for k in range(steps):
        t = np.full((len(has),), 1.0 - k / steps, np.float32)
        v = net(np.concatenate([x, x]) if guided else x, t, Q, has, pad)
        if guided:
            v = v[B:] + cfg * (v[:B] - v[B:])
        x = (x - v / steps).astype(np.float32)
    out = []
    b, a = butter(4, lowpass_hz / (FPS / 2))
    for i in range(B):
        A = x[i, : Ts[i]].astype(np.float64) * SD + MU
        if Ts[i] > 20 and lowpass_hz:
            A = filtfilt(b, a, A, axis=0, padlen=min(Ts[i] - 1, 15))
        out.append(A)
    return out


# -- reachability ----------------------------------------------------------------------------------------------------
def _rpy_to_mat(r: np.ndarray) -> np.ndarray:
    cr, cp, cy = np.cos(r).T
    sr, sp, sy = np.sin(r).T
    R = np.zeros((len(r), 3, 3))
    R[:, 0, 0] = cy * cp; R[:, 0, 1] = cy * sp * sr - sy * cr; R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp; R[:, 1, 1] = sy * sp * sr + cy * cr; R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp; R[:, 2, 1] = cp * sr; R[:, 2, 2] = cp * cr
    return R


def _mat_to_rpy(R: np.ndarray) -> np.ndarray:
    sy = np.hypot(R[..., 0, 0], R[..., 1, 0])
    return np.stack([np.arctan2(R[..., 2, 1], R[..., 2, 2]), np.arctan2(-R[..., 2, 0], sy),
                     np.arctan2(R[..., 1, 0], R[..., 0, 0])], -1)


class Reach:
    """Line-search every unreachable head pose back toward the last reachable one, with the SDK's analytical IK
    (the daemon rejects unreachable targets and holds the last one, which looks like a freeze)."""

    def __init__(self) -> None:
        import json

        import reachy_mini
        from reachy_mini_rust_kinematics import ReachyMiniRustKinematics

        with open(Path(reachy_mini.__file__).parent / "assets" / "kinematics_data.json", "rb") as f:
            d = json.load(f)
        self.z = d["head_z_offset"]
        self.kin = ReachyMiniRustKinematics(d["motor_arm_length"], d["rod_length"])
        for m in d["motors"]:
            self.kin.add_branch(m["branch_position"], np.linalg.inv(m["T_motor_world"]), 1 if m["solution"] else -1)

    def ok(self, H: np.ndarray, yaw: float) -> bool:
        p = np.array(H, float)
        p[2, 3] += self.z
        j = self.kin.inverse_kinematics_safe(p.tolist(), body_yaw=float(yaw), max_relative_yaw=np.deg2rad(65),
                                             max_body_yaw=np.deg2rad(160))
        return bool(np.all(np.isfinite(np.array(j))))

    @staticmethod
    def interp(H0: np.ndarray, H1: np.ndarray, a: float) -> np.ndarray:
        R0, R1 = H0[:3, :3], H1[:3, :3]
        Rrel = R0.T @ R1
        ang = np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))
        if ang < 1e-8:
            Ra = R0
        else:
            w = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]]) / (2 * np.sin(ang))
            K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            Ra = R0 @ (np.eye(3) + np.sin(a * ang) * K + (1 - np.cos(a * ang)) * K @ K)
        H = np.eye(4)
        H[:3, :3] = Ra
        H[:3, 3] = (1 - a) * H0[:3, 3] + a * H1[:3, 3]
        return H

    def project(self, A: np.ndarray, iters: int = 12) -> tuple[np.ndarray, float]:
        """(T, 9) poses -> reachable (T, 9) poses (antennas and timing untouched), fraction of frames fixed."""
        A = A.copy()
        Hs = np.tile(np.eye(4), (len(A), 1, 1))
        Hs[:, :3, :3] = _rpy_to_mat(A[:, 3:6])
        Hs[:, :3, 3] = A[:, :3]
        last, fixed = None, 0
        for i, H in enumerate(Hs):
            yaw = float(A[i, 8])
            if self.ok(H, yaw):
                last = (H, yaw)
                continue
            fixed += 1
            H0, y0 = last if last is not None else (np.eye(4), 0.0)
            lo, hi = 0.0, 1.0
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                if self.ok(self.interp(H0, H, mid), (1 - mid) * y0 + mid * yaw):
                    lo = mid
                else:
                    hi = mid
            Hp, yp = self.interp(H0, H, lo), (1 - lo) * y0 + lo * yaw
            A[i, :3], A[i, 3:6], A[i, 8] = Hp[:3, 3], _mat_to_rpy(Hp[:3, :3]), yp
            last = (Hp, yp)
        return A, fixed / max(1, len(A))


# -- the renderer used by the director -------------------------------------------------------------------------------
class FlowRenderer:
    """Recipe -> learned motion. Loads in the background (the first start downloads ~87 MB once); until it is
    ready, ``render`` raises ``RuntimeError`` so callers fall back instead of waiting."""

    def __init__(self, steps: int = STEPS, cfg: float = CFG) -> None:
        self.steps, self.cfg = steps, cfg
        self._net: FlowNet | None = None
        self._stats: dict | None = None
        self._reach: Reach | None = None
        self._loading = threading.Lock()
        self._lock = threading.Lock()
        self.error: str | None = None

    def start_loading(self) -> None:
        if self._net is None and self.error is None and not self._loading.locked():
            threading.Thread(target=self.load, name="flow-load", daemon=True).start()

    def load(self) -> bool:
        with self._loading:
            if self._net is None and self.error is None:
                try:
                    t0 = time.perf_counter()
                    sd, self._stats = load_checkpoint(checkpoint_path())
                    try:
                        self._reach = Reach()
                    except Exception as e:  # noqa: BLE001 - no IK available: clamp only
                        logger.warning("reachability projection unavailable (%s); clamping only", e)
                    self._net = FlowNet(sd)
                    logger.info("flow generator loaded in %.1f s", time.perf_counter() - t0)
                except Exception as e:  # noqa: BLE001
                    self.error = f"{type(e).__name__}: {e}"
                    logger.warning("flow generator unavailable: %s", self.error)
        return self._net is not None

    @property
    def ready(self) -> bool:
        return self._net is not None

    def unload(self) -> None:
        with self._loading, self._lock:
            self._net = self._stats = self._reach = None
            self.error = None

    def render(self, recipe: str, seed: int | None = None) -> np.ndarray:
        """Recipe -> (T, 9) reachable poses. Raises ``RecipeError`` for a bad recipe, ``RuntimeError`` if not loaded."""
        if self._net is None:
            self.start_loading()
            raise RuntimeError(self.error or "flow generator still loading")
        seed = int(np.random.default_rng().integers(1 << 31)) if seed is None else seed
        plan = recipe_plans(recipe, 1, seed)[0]
        with self._lock:  # one at a time keeps latency predictable (numpy already uses every core)
            A = generate_batch(self._net, self._stats, [plan], [seed], self.steps, self.cfg)[0]
        if self._reach is not None:
            A, _ = self._reach.project(A)
        A[:, 6:8] = np.clip(A[:, 6:8], -SAFE["ant"], SAFE["ant"])
        A[:, 8] = np.clip(A[:, 8], -SAFE["body"], SAFE["body"])
        return A
