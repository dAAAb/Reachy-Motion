"""Motion engines: the numpy flow generator, the renderer cache / fallback, and the director using them."""

import glob
import math
import time

import numpy as np
import pytest

from reachy_motion.motion import backends, flow
from reachy_motion.motion.director import Director
from reachy_motion.motion.planner import Gesture

NOD = "go .3 e=0 p=-3 z=5 E=2 | osc 1.4 p 7 .55 E=2 | go .4 p=0 E=1"


def test_gelu_matches_erf():
    from scipy.special import erf

    x = np.linspace(-6, 6, 20001).astype(np.float32)
    ref = 0.5 * x.astype(np.float64) * (1 + erf(x.astype(np.float64) / math.sqrt(2)))
    assert np.abs(flow._gelu(x) - ref).max() < 1e-5


def _tiny_net(d=12, heads=2, layers=2, maxlen=64):
    rng = np.random.default_rng(0)
    w = lambda *s: (rng.standard_normal(s) * 0.1).astype(np.float32)
    sd = {"pos": w(1, maxlen, d), "inp.weight": w(d, 18), "inp.bias": w(d), "temb.0.weight": w(d, 256),
          "temb.0.bias": w(d), "temb.2.weight": w(d, d), "temb.2.bias": w(d), "nf.weight": w(d), "nf.bias": w(d),
          "out.weight": w(9, d), "out.bias": w(9)}
    for i in range(layers):
        p = f"blocks.{i}."
        sd |= {p + "ada.1.weight": w(6 * d, d), p + "ada.1.bias": w(6 * d), p + "att.in_proj_weight": w(3 * d, d),
               p + "att.in_proj_bias": w(3 * d), p + "att.out_proj.weight": w(d, d), p + "att.out_proj.bias": w(d),
               p + "mlp.0.weight": w(4 * d, d), p + "mlp.0.bias": w(4 * d), p + "mlp.2.weight": w(d, 4 * d),
               p + "mlp.2.bias": w(d)}
    return flow.FlowNet(sd, heads=heads)


def test_flownet_padding_does_not_leak():
    net = _tiny_net()
    x = np.random.default_rng(1).standard_normal((1, 20, 9)).astype(np.float32)
    P = np.random.default_rng(2).standard_normal((1, 20, 8)).astype(np.float32)
    has, t = np.ones((1, 1, 1), np.float32), np.array([0.5], np.float32)
    short = net(x[:, :15], t, P[:, :15], has, np.zeros((1, 15), bool))
    pad = np.zeros((1, 20), bool)
    pad[0, 15:] = True
    padded = net(x, t, P, has, pad)
    assert np.allclose(short, padded[:, :15], atol=1e-5)


def test_plans_are_deterministic_and_generate_shapes():
    a, b = flow.recipe_plans(NOD, 2, seed=3), flow.recipe_plans(NOD, 2, seed=3)
    assert a == b and a[0] != a[1]  # odd variants are mirrored
    stats = {"MU": np.zeros(9), "SD": np.ones(9), "PMU": np.zeros(8), "PSD": np.ones(8)}
    out = flow.generate_batch(_tiny_net(maxlen=200), stats, a, seeds=[0, 1], steps=2)
    assert [o.shape[1] for o in out] == [9, 9] and all(np.isfinite(o).all() for o in out)


class _SlowBackend:
    def render(self, recipe, seed):
        time.sleep(0.3)
        return np.zeros((25, 9))


def test_renderer_goes_reflex_only_when_too_slow():
    r = backends.Renderer(_SlowBackend(), "slow")
    r.warm([NOD])
    for _ in range(50):
        if r.instant(NOD) is not None:
            break
        time.sleep(0.05)
    assert r.instant(NOD) is not None  # reflexes still pre-rendered
    assert not r.live
    with pytest.raises(RuntimeError):
        r.render(NOD)


def test_build_recipe_engine_has_no_renderer():
    planner, renderer, desc = backends.build("recipe", "cloud")
    assert planner == "cloud" and renderer is None and desc.startswith("recipe")
    assert backends.build("nonsense", "cloud")[1] is None


class _FixedRenderer:
    name = "fake"

    def render(self, recipe):
        return np.full((30, 9), 0.01)

    def instant(self, recipe):
        return None


def test_director_plays_pre_rendered_poses():
    class P:
        def plan(self, line, heard=None, tone=None):
            return Gesture(NOD, "nod", "planner", line)

    played, events, t0 = [], [], time.monotonic()
    d = Director(play=played.append, planner=P(), renderer=_FixedRenderer(), time_of=lambda p: t0 + p,
                 position=lambda: 0.0, on_event=events.append, lead_s=0.0, use_reflex=False)
    d.speech_line("我們走吧。", 0.0)
    time.sleep(0.4)
    d.close()
    g = [e for e in events if e["type"] == "gesture"]
    assert g and g[0]["render"] == "generator"
    assert played and abs(played[0].duration - 29 / 25) < 0.1


CKPT = glob.glob(str(__import__("pathlib").Path.home() / ".cache/huggingface/hub/models--binhpham--reachy-mini-motion-"
                     "planner-*/snapshots/*/generator.pt"))


@pytest.mark.skipif(not CKPT, reason="generator.pt not downloaded")
def test_real_generator_renders_reachable_motion(monkeypatch):
    monkeypatch.setenv("REACHY_MOTION_GENERATOR", CKPT[0])
    r = flow.FlowRenderer()
    assert r.load()
    A = r.render(NOD, seed=1)
    assert A.shape[1] == 9 and np.isfinite(A).all()
    assert np.abs(A[:, 4]).max() < 0.6 and np.abs(A).max() < math.pi


def test_remote_backs_off_after_failure(monkeypatch):
    import httpx

    class FakeClient:
        calls = 0

        def __init__(self, *a, **k):
            pass

        def get(self, url, timeout=None):
            return httpx.Response(200, json={"planner": "0.8b"}, request=httpx.Request("GET", url))

        def post(self, url, json=None):
            FakeClient.calls += 1
            raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "Client", FakeClient)
    rm = backends.RemoteMotion("http://mac:18110")
    with pytest.raises(httpx.ConnectError):
        rm.render(NOD)
    with pytest.raises(RuntimeError):  # skipped without a network call
        rm.render(NOD)
    assert FakeClient.calls == 1
    assert rm.plan("我們走吧。").source == "fallback"  # planner falls back to a reflex / talking gesture at once


def test_robot_without_server_does_not_render_in_process(monkeypatch):
    monkeypatch.setattr(backends, "fast_machine", lambda: False)
    monkeypatch.setattr(backends, "discover", lambda timeout_s=2.5: None)
    planner, renderer, desc = backends.build("flow", "cloud")
    assert planner == "cloud" and renderer is None and "waiting for the Mac motion server" in desc


def test_remote_motion_refuses_a_server_in_standby(monkeypatch):
    import httpx

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def get(self, url, timeout=None):
            return httpx.Response(200, json={"state": "off"}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "Client", FakeClient)
    with pytest.raises(RuntimeError):
        backends.RemoteMotion("http://mac:18110")


def test_server_models_switch_on_and_off(monkeypatch):
    from reachy_motion import motion_server

    monkeypatch.setattr(motion_server, "can_run_mlx", lambda: False)
    m = motion_server.Models(None)
    monkeypatch.setattr(m.gen, "load", lambda: True)
    monkeypatch.setattr(type(m.gen), "ready", property(lambda self: True))
    monkeypatch.setattr(m.gen, "render", lambda *a, **k: np.zeros((10, 9)))
    assert m.health()["state"] == "off"
    m.start()
    for _ in range(50):
        if m.state == "on":
            break
        time.sleep(0.02)
    assert m.health()["state"] == "on"
    m.stop()
    assert m.health()["state"] == "off"
