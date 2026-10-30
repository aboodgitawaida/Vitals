"""web/features.js (the browser's colour measurement) matches the Python code it ports."""

import base64
import json
import os
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import face_layout
from face_layout import PATCHES
from frame_features import FeatureExtractor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")

CFG = face_layout.layout_config()


def make_frame(rng, w=640, h=360):
    """Smooth skin-like gradient + sensor noise, some dark 'hair' and saturated 'glare' spots."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([60 + 40 * xx / w, 110 + 50 * yy / h, 160 + 30 * np.sin(xx / 40)], axis=-1)  # BGR
    img = base + rng.normal(0, 6, base.shape)
    img[rng.random((h, w)) < 0.05] = 255                     # glare
    img[rng.random((h, w)) < 0.08] *= 0.3                    # dark pixels (hair / brows)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_landmarks(rng, w, h, shift=(0.0, 0.0)):
    """478 points in a face-sized box (normalised 0..1, like MediaPipe); may poke out of the frame."""
    cx, cy = 0.5 + shift[0], 0.55 + shift[1]
    pts = np.column_stack([cx + rng.uniform(-0.18, 0.18, 478), cy + rng.uniform(-0.3, 0.3, 478)])
    return pts


def run_js(frames):
    payload = {"cfg": CFG, "frames": [
        {"rgba": base64.b64encode(
            np.dstack([f[:, :, ::-1], np.full(f.shape[:2], 255, np.uint8)]).tobytes()).decode(),
         "width": f.shape[1], "height": f.shape[0],
         "landmarks": None if lm is None else (lm * [f.shape[1], f.shape[0]]).tolist()}
        for f, lm in frames]}
    out = subprocess.run(["node", os.path.join(ROOT, "tests", "run_web_features.mjs")],
                         input=json.dumps(payload), capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def run_py(frames):
    fx, out = FeatureExtractor(), []
    for f, lm in frames:
        face = None if lm is None else SimpleNamespace(
            landmark=[SimpleNamespace(x=x, y=y) for x, y in lm])
        out.append(fx.process(f, face))
    return out


@needs_node
def test_js_features_match_python():
    rng = np.random.default_rng(0)
    frames = []
    lm = make_landmarks(rng, 640, 360)
    for k in range(6):
        f = make_frame(rng)
        if k == 3:
            frames.append((f, None))                          # face lost
            continue
        jitter = rng.normal(0, 0.002, lm.shape)
        moved = lm + (0.08 if k == 5 else 0.0) + jitter      # last frame: big move (fast motion)
        frames.append((f, moved))
    frames.append((make_frame(rng), make_landmarks(rng, 640, 360, shift=(-0.38, -0.3))))  # off the edge

    js, py = run_js(frames), run_py(frames)
    compared = 0
    for j, p in zip(js, py):
        assert j["face"] == p.face
        if not p.face:
            continue
        assert j["fast_motion"] == p.fast_motion
        assert j["movement"] == pytest.approx(p.movement, rel=1e-4, abs=1e-6)
        assert np.allclose(j["pose"], p.pose, rtol=1e-4, atol=1e-6)
        for n in PATCHES:
            a, b = j["colors"][n], p.colors[n]
            assert (a is None) == (b is None), n
            if a is not None:
                # Polygon edges are rasterised slightly differently from OpenCV: a few
                # edge pixels differ, so the means agree to ~0.1 of 255 but not to the bit.
                assert np.allclose(a, b, atol=0.3), (n, a, b)
                compared += 1
        if p.bg_rgb is None:
            assert j["bg_rgb"] is None
        else:
            assert np.allclose(j["bg_rgb"], p.bg_rgb, atol=1e-6)
        for region, lum in p.overlay["stats"]["luminance"].items():
            assert j["stats"]["luminance"][region] == pytest.approx(lum, abs=0.3)
        assert j["stats"]["clipped_fraction"] == pytest.approx(p.overlay["stats"]["clipped_fraction"], abs=0.01)
    assert compared > 50


def test_layout_json_matches_face_layout():
    """web/face_layout.json is generated from face_layout.py; regenerate it after changing a setting:
    python face_layout.py > web/face_layout.json"""
    with open(os.path.join(ROOT, "web", "face_layout.json")) as f:
        assert json.load(f) == json.loads(json.dumps(face_layout.layout_config()))
