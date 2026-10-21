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
