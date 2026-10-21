"""
synthetic_benchmark.py -- ablation study on fake data where we KNOW the true
heart rate. Each pipeline feature is switched on one at a time, so you can see
what each one contributes.

Run:  python synthetic_benchmark.py          (takes ~1 minute)

Fake data uses the standard skin reflection model from the POS paper:

  colour(t) = light(t) * (skin + pulse(t) * pulse_direction + shine(t) * white)
              + camera_noise

where light(t) can flicker in brightness and in COLOUR, the background wall
sees the same light (but has no pulse), and some patches can be in shadow.

HONEST CAVEAT: the data is generated from the same physics the methods are
built around, so it flatters them. It proves the code does what the theory
says. It is NOT validation -- real accuracy comes from recordings alongside a
pulse oximeter / smartwatch (replay.py, Phase 4).
"""

import numpy as np
from scipy.signal import butter, filtfilt

from face_regions import PATCHES, patch_region
from vitals_dsp import analyze_heart_rate

PULSE_DIRECTION = np.array([0.33, 0.77, 0.53])   # how blood volume changes R,G,B (Wang et al. 2017)
PULSE_DIRECTION /= np.linalg.norm(PULSE_DIRECTION)
SKIN_TONES = {                                    # mean RGB of skin, relative pulse strength
    "lighter": (np.array([205, 155, 135]), 1.0),  # melanin absorbs light: darker skin reflects
    "medium": (np.array([165, 115, 90]), 0.8),    # less light AND has a weaker pulse component,
    "darker": (np.array([95, 65, 50]), 0.6),      # so the same camera gets a lower SNR
}
WALL = np.array([185, 180, 170])
SCENARIOS = ["steady light", "screen glow", "coloured flicker", "uneven light", "small motion",
             "dim room", "dim + uneven", "dim + flicker"]

CONFIGS = {   # name: (method, patches?, background?, harmonics?)
    "green":           ("green", False, False, False),
    "POS, 3 regions":  ("pos", False, False, False),   # = v2
    "+ patches":       ("pos", True, False, False),
    "+ background":    ("pos", True, True, False),
    "+ harmonics":     ("pos", True, True, True),
    "auto (v3)":       ("auto", True, True, True),     # = v3: everything on, picks green or POS per update
}


def band_noise(t, low, high, amplitude, rng):
    """Random wiggle containing only frequencies between low and high Hz."""
    b, a = butter(2, [low / 15, high / 15], btype="band")
    x = filtfilt(b, a, rng.normal(size=t.size))
    return amplitude * x / (x.std() + 1e-12)


def make_trial(scenario, tone, rng, seconds=12):
    t = np.cumsum(rng.normal(1 / 30, 0.004, int(seconds * 30)))   # jittery frame times
    t -= t[0]
    true_bpm = rng.uniform(50, 150)
    inst_hz = true_bpm / 60 * (1 + 0.02 * np.sin(2 * np.pi * 0.25 * t))   # mild variability
    phase = 2 * np.pi * np.cumsum(inst_hz * np.gradient(t))
    pulse = np.sin(phase) + 0.35 * np.sin(2 * phase + 0.6)                # 2nd harmonic

    light = np.ones((t.size, 3)) * (1 + 0.004 * t / seconds)[:, None]    # slow drift
    shine = np.zeros_like(t)
    if scenario == "screen glow":           # brightness flicker (same in R, G, B)
        light += band_noise(t, 0.7, 3.0, 0.006, rng)[:, None]
    if scenario in ("coloured flicker", "dim + flicker"):   # light changing colour (warm <-> cool)
        light += band_noise(t, 0.7, 3.0, 0.012, rng)[:, None] * np.array([1.0, 0.45, -0.3])
    if scenario == "small motion":
        light += band_noise(t, 0.3, 2.0, 0.003, rng)[:, None]
        shine = band_noise(t, 0.3, 2.0, 0.004, rng)

    skin, strength = SKIN_TONES[tone]
    patches = {}
    for name in PATCHES:
        region = patch_region(name)
        shade = 1.0
        if scenario in ("uneven light", "dim + uneven") and (region == "right cheek" or name in
                                           ("forehead_p2", "forehead_p3", "forehead_p6", "forehead_p7")):
            shade = 0.3                      # this side of the face is in shadow
        amp = 0.0012 * strength * (1.0 if region == "forehead" else 0.7)
        surface = skin * shade
        clean = light * (surface + amp * np.linalg.norm(surface) * pulse[:, None] * PULSE_DIRECTION
                         + shine[:, None] * np.linalg.norm(surface) * np.ones(3) / np.sqrt(3))
        # Camera noise per patch. In a dim room the camera raises its gain, so noise is ~4x larger
        # (this is what pushed the real-world SNR below 0 dB).
        noise = 0.6 if scenario.startswith("dim") else 0.15
        patches[name] = clean + rng.normal(0, noise, clean.shape)
    background = light * WALL + rng.normal(0, 0.2 if scenario.startswith("dim") else 0.05, (t.size, 3))
    return t, patches, background, true_bpm


def merge_regions(patches):
    """Average patches back into 3 big regions (what v2 did)."""
    regions = {}
    for name, rgb in patches.items():
        regions.setdefault(patch_region(name), []).append(rgb)
    return {r: np.mean(v, axis=0) for r, v in regions.items()}


def run(t, patches, background, config):
    method, use_patches, use_bg, use_harm = CONFIGS[config]
    traces = patches if use_patches else merge_regions(patches)
    return analyze_heart_rate(t, traces, background, method=method,
                              use_background=use_bg, use_harmonics=use_harm)


def main(trials=40, seed=1):
    rng = np.random.default_rng(seed)
    tones = list(SKIN_TONES)
    tone_errors = {tone: [] for tone in tones}
    print("Mean absolute error in BPM (lower is better), % of estimates within 5 BPM in brackets")
    print(f"skin tones mixed evenly, {trials} trials per scenario, 12 s windows\n")
    print(f"{'scenario':<17}" + "".join(f"{c:>16}" for c in CONFIGS))
    print("-" * (17 + 16 * len(CONFIGS)))
    for scenario in SCENARIOS:
        errors = {c: [] for c in CONFIGS}
        for i in range(trials):
            tone = tones[i % len(tones)]
            t, patches, background, truth = make_trial(scenario, tone, rng)
            for config in CONFIGS:
                err = abs(run(t, patches, background, config)["bpm"] - truth)
                errors[config].append(err)
                if config == "auto (v3)":
                    tone_errors[tone].append(err)
        cells = [f"{np.mean(e):5.1f} ({np.mean(np.array(e) <= 5):4.0%})" for e in errors.values()]
        print(f"{scenario:<17}" + "".join(f"{c:>16}" for c in cells))

    print("\nFull v3 pipeline by skin tone (all scenarios):")
    for tone, e in tone_errors.items():
        e = np.array(e)
        print(f"  {tone:<8} MAE {e.mean():5.1f} BPM, within 5 BPM {np.mean(e <= 5):4.0%}")


if __name__ == "__main__":
    main()
