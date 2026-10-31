"""Whole-face colour: which pixels it measures, and that the analysis uses it for the pulse."""

import numpy as np

from face_layout import FACE_EXCLUDE, FACE_OVAL, FACE_TRACE
from face_regions import face_color, fill_convex
from synthetic_benchmark import make_trial
from vitals_dsp import analyze_heart_rate


def test_fill_convex_clips_at_frame_edges():
    mask = np.zeros((10, 10), np.uint8)
    fill_convex(mask, [(-8, 2), (-3, 2), (-3, 6), (-8, 6)], 1)          # entirely left of the frame
    assert mask.sum() == 0
    fill_convex(mask, [(-5, -5), (4, -5), (4, 3), (-5, 3)], 1)          # sticks out top-left
    assert mask.sum() == 5 * 4 and mask[:4, :5].all()


def _landmarks():
    """A face-like layout: oval around the centre, eyes/brows and lips as small clusters inside it."""
    lm = np.full((468, 2), 320.0)
    a = np.linspace(0, 2 * np.pi, len(FACE_OVAL), endpoint=False)
    lm[FACE_OVAL] = np.column_stack([320 + 80 * np.cos(a), 240 + 110 * np.sin(a)])
    for idx, (cx, cy) in zip(FACE_EXCLUDE, [(285, 200), (355, 200), (320, 300)]):
        b = np.linspace(0, 2 * np.pi, len(idx), endpoint=False)
        lm[idx] = np.column_stack([cx + 22 * np.cos(b), cy + 12 * np.sin(b)])
    return lm.astype(np.float32)


def test_face_color_skips_eyes_lips_and_glare():
    frame = np.full((480, 640, 3), (60, 110, 170), np.uint8)       # skin (BGR)
    frame[194:206, 275:295] = 0                                    # left eye: black
    frame[294:306, 310:330] = 0                                    # lips: black
    frame[240:250, 300:310] = 255                                  # glare spot
    r, g, b = face_color(frame, _landmarks())
    assert (r, g, b) == (170, 110, 60)


def test_face_color_none_when_face_too_small():
    lm = _landmarks()
    lm = (lm - 320) * 0.05 + 320
    assert face_color(np.full((480, 640, 3), 128, np.uint8), lm) is None


def test_pulse_comes_from_face_when_available():
    t, traces, bg, true_bpm = make_trial("steady light", "lighter", np.random.default_rng(1))
    rng = np.random.default_rng(2)
    noisy = {n: v + rng.normal(0, 5, v.shape) for n, v in traces.items()}         # useless patches
    face = np.mean(list(traces.values()), axis=0)                                  # clean face
    r = analyze_heart_rate(t, {**noisy, FACE_TRACE: face}, bg, patch_map=False)
    assert r["source"] == "face"
    assert abs(r["bpm"] - true_bpm) < 3
    assert set(r["patch_weight"]) == set(noisy) and abs(sum(r["patch_weight"].values()) - 1) < 1e-9
    face_missing = face.copy()
    face_missing[: len(t) // 2] = np.nan                           # face colour mostly missing: use patches
    assert analyze_heart_rate(t, {**traces, FACE_TRACE: face_missing}, bg, patch_map=False)["source"] == "patches"


def test_too_dark_patch_is_skipped_not_divided_by_zero():
    t, traces, bg, true_bpm = make_trial("steady light", "lighter", np.random.default_rng(3))
    traces = dict(traces)
    dark = np.zeros_like(traces["forehead_p0"])
    dark[:, 0] = 3.0                                   # black hair: red barely above 0, green and blue 0
    traces["forehead_p0"] = dark
    r = analyze_heart_rate(t, traces, bg)              # patch_map on: every patch is analysed
    assert r["patch_snr"]["forehead_p0"] is None and "forehead_p0" not in r["patch_weight"]
    assert abs(r["bpm"] - true_bpm) < 3
