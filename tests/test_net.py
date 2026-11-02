"""vitals_net: the numpy network matches PyTorch, and the input preparation behaves."""

import os

import numpy as np
import pytest

import vitals_net as vn
from face_layout import FACE_TRACE

FIXTURE = os.path.join(os.path.dirname(__file__), "data", "net_fixture.npz")


def fixture_net(tmp_path):
    """A small random network exported by train_net.export, with PyTorch's output for one input."""
    d = np.load(FIXTURE)
    weights = {k[2:]: d[k] for k in d.files if k.startswith("w_")}
    path = tmp_path / "w.npz"
    np.savez(path, **weights)
    return vn.PulseNet(str(path)), d["x"], d["wave"], d["logits"]


def test_numpy_forward_matches_pytorch(tmp_path):
    net, x, wave, logits = fixture_net(tmp_path)
    w, lg = net.forward(x.astype(np.float64))
    assert np.allclose(w, wave, atol=1e-4) and np.allclose(lg, logits, atol=1e-4)


def test_estimate_returns_a_probability_and_a_rate(tmp_path, monkeypatch):
    net, x, _, _ = fixture_net(tmp_path)
    monkeypatch.setattr(vn, "_net", net)
    r = vn.estimate(x)
    assert abs(r["posterior"].sum() - 1) < 1e-9 and r["posterior"].shape == vn.BPM_BINS.shape
    assert 40 <= r["bpm"] <= 180 and 0 <= r["confidence"] <= 1
    assert r["pulse"].shape == (x.shape[1],)


def _traces(rng, n=300, bpm=72.0):
    t = np.arange(n) / 30.0
    pulse = np.sin(2 * np.pi * bpm / 60 * t)
    skin = np.array([170.0, 120.0, 95.0])
    rgb = [skin * (1 + 0.002 * pulse[:, None] * np.array([0.33, 0.77, 0.53])) + rng.normal(0, 0.3, (n, 3))
           for _ in range(vn.N_TRACES - 1)]
    return t, rgb + [np.full((n, 3), 180.0) + rng.normal(0, 0.3, (n, 3))]


def test_window_input_shape_scale_invariance_and_flags():
    rng = np.random.default_rng(0)
    _, rgb = _traces(rng)
    x = vn.window_input(rgb)
    assert x.shape == (vn.N_IN, 300) and x.dtype == np.float32
    assert np.allclose(vn.window_input([r * 1.7 for r in rgb]), x, atol=1e-4)     # brighter room, same input
    rgb[3] = None
    flags = vn.window_input(rgb)[-vn.N_TRACES:, 0]
    assert flags[3] == 0 and flags.sum() == vn.N_TRACES - 1


def test_prepare_needs_a_face_colour():
    rng = np.random.default_rng(1)
    t, rgb = _traces(rng)
    traces = dict(zip(vn.TRACES[:-1], rgb[:-1]))
    assert vn.prepare(t, traces, rgb[-1]).shape[0] == vn.N_IN
    traces.pop(FACE_TRACE)
    assert vn.prepare(t, traces, rgb[-1]) is None


@pytest.mark.skipif(not vn.available(), reason="no trained weights")
def test_trained_network_finds_a_clean_pulse():
    rng = np.random.default_rng(2)
    for bpm in (60.0, 85.0, 110.0):
        t, rgb = _traces(rng, n=300, bpm=bpm)
        r = vn.estimate(vn.window_input(rgb))
        assert abs(r["bpm"] - bpm) < 5, (bpm, r["bpm"])
