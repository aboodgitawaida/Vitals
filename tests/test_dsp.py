"""Signal-processing building blocks behave the way the comments claim."""

import numpy as np
import pytest

from synthetic_benchmark import make_trial
from vitals_dsp import (FS, HR_HIGH, HR_LOW, _fill_missing, analyze_heart_rate, bandpass,
                        pick_peak, pos_pulse, power_spectrum, remove_background, resample_uniform,
                        snr_db)


def test_resample_makes_even_grid():
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.uniform(0.02, 0.05, 300))
    t_u, v = resample_uniform(t, np.sin(t))
    assert np.allclose(np.diff(t_u), 1 / FS)
    assert np.allclose(v, np.sin(t_u), atol=1e-2)


@pytest.mark.parametrize("bpm", [48, 72, 110, 170])
def test_peak_of_clean_sine(bpm):
    t = np.arange(0, 15, 1 / FS)
    x = bandpass(np.sin(2 * np.pi * bpm / 60 * t), HR_LOW, HR_HIGH)
    f, p = power_spectrum(x)
    assert abs(pick_peak(f, p, HR_LOW, HR_HIGH) * 60 - bpm) < 1.0


def test_snr_high_for_sine_low_for_noise():
    t = np.arange(0, 15, 1 / FS)
    rng = np.random.default_rng(1)
    f, p = power_spectrum(bandpass(np.sin(2 * np.pi * 1.2 * t), HR_LOW, HR_HIGH))
    assert snr_db(f, p, 1.2, HR_LOW, HR_HIGH) > 10
    f, p = power_spectrum(bandpass(rng.normal(size=t.size), HR_LOW, HR_HIGH))
    assert snr_db(f, p, pick_peak(f, p, HR_LOW, HR_HIGH), HR_LOW, HR_HIGH) < 0


def test_pos_cancels_pure_brightness_change():
    """POS's core promise: light getting brighter/dimmer (all channels equally) vanishes."""
    t = np.arange(0, 15, 1 / FS)
    flicker = 1 + 0.01 * np.sin(2 * np.pi * 1.5 * t)          # in the heart-rate band
    rgb = np.array([180.0, 130.0, 110.0]) * flicker[:, None]
    pos = pos_pulse(rgb)
    green = rgb[:, 1] / rgb[:, 1].mean() - 1
    assert pos.std() < 1e-3 * green.std()


def test_background_removal_cancels_shared_light():
    t = np.arange(0, 15, 1 / FS)
    rng = np.random.default_rng(2)
    light = 1 + 0.01 * bandpass(rng.normal(size=t.size), 0.7, 3.0)[:, None] * np.array([1.0, 0.5, -0.3])
    skin = np.array([180.0, 130.0, 110.0]) * light
    wall = np.array([185.0, 180.0, 170.0]) * light
    wall_band = bandpass(wall / wall.mean(axis=0) - 1, HR_LOW, HR_HIGH)
    before = bandpass(skin / skin.mean(axis=0) - 1, HR_LOW, HR_HIGH).std()
    cleaned = remove_background(skin, wall_band)
    after = bandpass(cleaned / cleaned.mean(axis=0) - 1, HR_LOW, HR_HIGH).std()
    assert after < 0.05 * before


def test_fill_missing_bridges_short_gaps_and_rejects_long_ones():
    t = np.arange(0, 10, 1 / FS)
    rgb = np.column_stack([t, t, t]).astype(float)
    short = rgb.copy()
    short[100:110] = np.nan
    filled = _fill_missing(t, short)
    assert not np.isnan(filled).any() and np.allclose(filled, rgb)
    long_gap = rgb.copy()
    long_gap[:100] = np.nan                                   # 33% missing
    assert _fill_missing(t, long_gap) is None


@pytest.mark.parametrize("scenario", ["steady light", "coloured flicker", "dim room"])
def test_full_pipeline_on_synthetic(scenario):
    rng = np.random.default_rng(5)
    errors = []
    for _ in range(6):
        t, patches, bg, truth = make_trial(scenario, "medium", rng)
        errors.append(abs(analyze_heart_rate(t, patches, bg)["bpm"] - truth))
    assert np.median(errors) < 3.0


def test_patch_map_off_gives_same_answer():
    rng = np.random.default_rng(6)
    t, patches, bg, _ = make_trial("steady light", "lighter", rng)
    a = analyze_heart_rate(t, patches, bg)
    b = analyze_heart_rate(t, patches, bg, patch_map=False)
    assert a["bpm"] == b["bpm"] and a["snr_db"] == b["snr_db"]


# ---------------- motion correction ----------------

def _fidget_patch(rng, coupling):
    """A patch whose brightness follows 4 head-pose signals, plus a pulse."""
    t = np.arange(0, 15, 1 / FS)
    pose = np.column_stack([bandpass(rng.normal(size=t.size), 0.5, 2.0) for _ in range(4)])
    pose /= pose.std(axis=0)
    shading = 1 + pose @ (coupling * np.array([1.0, -0.6, 0.4, 0.8]))
    pulse = 0.001 * np.sin(2 * np.pi * 1.2 * t)
    rgb = np.array([170.0, 120.0, 95.0]) * (shading + pulse)[:, None]
    pose_band = bandpass(pose, HR_LOW, HR_HIGH)
    return t, rgb, pose_band


def test_motion_regression_removes_pose_artefact():
    from vitals_dsp import remove_nuisance
    t, rgb, pose_band = _fidget_patch(np.random.default_rng(3), 0.004)
    g = lambda x: bandpass(x[:, 1] / x[:, 1].mean() - 1, HR_LOW, HR_HIGH)
    cleaned = remove_nuisance(rgb, pose_band)
    f, p = power_spectrum(g(rgb))
    assert abs(pick_peak(f, p, HR_LOW, HR_HIGH) - 1.2) > 0.05         # motion wins before
    f, p = power_spectrum(g(cleaned))
    assert abs(pick_peak(f, p, HR_LOW, HR_HIGH) - 1.2) < 0.05         # pulse wins after


def test_motion_gate_off_when_still_on_when_moving():
    from vitals_dsp import MOTION_R2_MIN, motion_explained
    rng = np.random.default_rng(4)
    _, still, pose_band = _fidget_patch(rng, 0.0)
    _, moving, pose_band2 = _fidget_patch(rng, 0.004)
    assert motion_explained(still, pose_band) < MOTION_R2_MIN
    assert motion_explained(moving, pose_band2) > MOTION_R2_MIN


def test_pose_with_missing_frames_is_handled():
    rng = np.random.default_rng(7)
    t, patches, bg, truth = make_trial("steady light", "lighter", rng)
    pose = rng.normal(size=(t.size, 4))
    pose[50:60] = np.nan
    r = analyze_heart_rate(t, patches, bg, pose=pose)
    assert abs(r["bpm"] - truth) < 3 and r["motion_patches"] == 0


# ---------------- CHROM and method agreement ----------------

def test_chrom_finds_pulse_and_ignores_brightness():
    from vitals_dsp import chrom_pulse
    t = np.arange(0, 15, 1 / FS)
    pulse = 0.002 * np.sin(2 * np.pi * 1.3 * t)
    flicker = 1 + 0.01 * np.sin(2 * np.pi * 2.2 * t)                 # brightness only
    direction = np.array([0.33, 0.77, 0.53])
    rgb = np.array([180.0, 130.0, 110.0]) * flicker[:, None] * (1 + pulse[:, None] * direction / 0.77)
    f, p = power_spectrum(bandpass(chrom_pulse(rgb), HR_LOW, HR_HIGH))
    assert abs(pick_peak(f, p, HR_LOW, HR_HIGH) - 1.3) < 0.05


def test_agreement_reported_and_high_on_clean_data():
    rng = np.random.default_rng(9)
    t, patches, bg, truth = make_trial("steady light", "lighter", rng)
    r = analyze_heart_rate(t, patches, bg, patch_map=False)
    assert set(r["method_bpm"]) == {"pos", "chrom", "green"}
    assert r["agreement"] == 3
