"""
session_benchmark.py -- end-to-end test of what the USER sees, on simulated
2-minute sessions streamed exactly like the live app (new frame every ~33 ms,
new estimate every 0.5 s, 15 s window).

synthetic_benchmark.py asks "is one 12 s window estimated correctly?".
This file asks the question that matters on screen:
  * how much of the time is a number shown?      (coverage)
  * how often does it vanish and come back?      (dropouts per minute)
  * when shown, how wrong is it?                 (MAE, % within 5 BPM)
  * how often is it confidently WRONG?           (% of shown time > 10 BPM off)

Each session contains things that happen at a real desk: the heart rate
drifts, FaceMesh loses the face for a few frames, the head moves, and the
camera noise is set so the SNR hovers around 0 dB (like a dim bedroom).

Run:  python session_benchmark.py              (~1-3 minutes, uses all CPU cores)
      python session_benchmark.py --vanish     (does the number disappear when the pulse does?)
      python session_benchmark.py --calibrate  (measure tracker thresholds on pure noise)

HONEST CAVEAT: simulated data, built from the same reflection model the
methods assume. It shows the display logic does what it is designed to do. It
is NOT accuracy validation -- that needs a reference device (Phase 4).
"""

import argparse

import numpy as np

from face_regions import PATCHES, patch_region
from synthetic_benchmark import PULSE_DIRECTION, SKIN_TONES, WALL, band_noise
from vitals_dsp import analyze_heart_rate
from vitals_stream import HeartRateTracker, LegacyDisplay, simulate_display

WINDOW_SEC, MIN_SEC, UPDATE_EVERY, MOTION_HOLD = 15.0, 8.0, 0.5, 1.5
NOISE_LEVELS = {"ok light": 0.6, "dim": 1.1, "very dim": 1.6}   # camera noise (brightness levels)


def make_session(tone, noise, rng, seconds=120, pulse=True, events=True):
    """
    Simulated per-frame patch colours for one sitting.
    Returns times, {patch: (N,3)}, background (N,3), face_visible (N,), moving (N,), true_bpm (N,)
    """
    t = np.cumsum(rng.normal(1 / 30, 0.004, int(seconds * 30)))
    t -= t[0]
    # Heart rate drifts slowly (random walk, smoothed) around a resting value.
    walk = np.cumsum(rng.normal(0, 0.15, t.size))
    walk = np.convolve(walk, np.ones(90) / 90, mode="same")
    true_bpm = np.clip(rng.uniform(58, 90) + walk - walk.mean(), 45, 150)
    phase = 2 * np.pi * np.cumsum(true_bpm / 60 * np.gradient(t))
    wave = (np.sin(phase) + 0.35 * np.sin(2 * phase + 0.6)) if pulse else np.zeros_like(t)

    light = np.ones((t.size, 3)) * (1 + 0.004 * t / seconds)[:, None]
    light += band_noise(t, 0.7, 3.0, 0.004, rng)[:, None] * np.array([1.0, 0.45, -0.3])  # mild colour flicker
    shine = np.zeros_like(t)

    # Events: 4 brief face losses (3-15 frames) and 3 head movements (~1 s each).
    face = np.ones(t.size, dtype=bool)
    for start in (rng.uniform(10, seconds - 5, 4) if events else []):
        i = np.searchsorted(t, start)
        face[i:i + rng.integers(3, 16)] = False
    moving = np.zeros(t.size, dtype=bool)
    for start in (rng.uniform(15, seconds - 5, 3) if events else []):
        sel = (t >= start) & (t < start + 1.0)
        moving[sel] = True
        shine[sel] += 0.02 * np.sin(2 * np.pi * 1.3 * (t[sel] - start))
        light[sel] *= 1 + 0.01 * np.sin(2 * np.pi * 0.9 * (t[sel] - start))[:, None]

    skin, strength = SKIN_TONES[tone]
    patches = {}
    for name in PATCHES:
        amp = 0.0012 * strength * (1.0 if patch_region(name) == "forehead" else 0.7)
        clean = light * (skin + amp * np.linalg.norm(skin) * wave[:, None] * PULSE_DIRECTION
                         + shine[:, None] * np.linalg.norm(skin) * np.ones(3) / np.sqrt(3))
        patches[name] = clean + rng.normal(0, noise, clean.shape)
    background = light * WALL + rng.normal(0, noise / 3, (t.size, 3))
    return t, patches, background, face, moving, true_bpm


def stream(session, display, max_gap_sec, drop_motion=True):
    """Run a simulated session through the app's logic; adds the true BPM to each update."""
    t, patches, background, face, moving, true_bpm = session
    res = simulate_display(t, patches, background, face, moving, display, max_gap_sec,
                           drop_motion=drop_motion, window_sec=WINDOW_SEC, min_sec=MIN_SEC,
                           update_every=UPDATE_EVERY, motion_hold=MOTION_HOLD)
    return [(now, bpm, state, float(np.interp(now, t, true_bpm))) for now, bpm, state in res if now >= MIN_SEC]


def score(results, minutes):
    """Pooled over all sessions in a group, so every second counts equally."""
    shown = [(b, truth, s) for _, b, s, truth in results if b is not None]
    err = np.array([abs(b - truth) for b, truth, _ in shown])
    flags = [b is not None for _, b, _, _ in results]
    dropouts = sum(1 for a, b in zip(flags, flags[1:]) if a and not b)
    nan = float("nan")
    return {
        "coverage": len(shown) / max(len(results), 1),
        "drops/min": dropouts / minutes,
        "MAE": err.mean() if len(err) else nan,
        "<=5": (err <= 5).mean() if len(err) else nan,
        ">10": (err > 10).mean() if len(err) else nan,
        "held": np.mean([s == "holding" for _, _, s in shown]) if shown else nan,
    }


def run_one(job):
    """One session through both display logics (runs in a worker process)."""
    label, noise, tone, seed, tracker_settings = job
    session = make_session(tone, noise, np.random.default_rng(seed))
    old = stream(session, LegacyDisplay(), max_gap_sec=0.0, drop_motion=False)
    new = stream(session, HeartRateTracker(**tracker_settings), max_gap_sec=1.0)
    return label, tone, old, new


def print_row(label, name, results, minutes):
    m = score(results, minutes)
    print(f"{label:<10}{name:<16}{m['coverage']:>9.0%}{m['drops/min']:>10.1f}{m['MAE']:>7.1f}"
          f"{m['<=5']:>9.0%}{m['>10']:>9.0%}{m['held']:>7.0%}")


def main(sessions_per_cell=2, seed=7, tracker_settings=None):
    from multiprocessing import Pool
    jobs = [(label, noise, tone, seed * 1000 + i, tracker_settings or {})
            for label, noise in NOISE_LEVELS.items()
            for tone in SKIN_TONES for i in range(sessions_per_cell)]
    with Pool() as pool:
        done = pool.map(run_one, jobs)
    minutes = (120 - MIN_SEC) / 60

    print("Simulated 2-minute sessions, streamed like the live app (pooled over all sessions).")
    print("coverage = time a number is on screen; >10 off = share of shown time more than 10 BPM wrong\n")
    hdr = (f"{'light':<10}{'display logic':<16}{'coverage':>9}{'drops/min':>10}{'MAE':>7}"
           f"{'<=5 BPM':>9}{'>10 off':>9}{'held':>7}")
    print(hdr + "\n" + "-" * len(hdr))
    for label in NOISE_LEVELS:
        group = [d for d in done if d[0] == label]
        n_min = minutes * len(group)
        print_row(label, "v3 (old)", [r for d in group for r in d[2]], n_min)
        print_row(label, "v4 tracker", [r for d in group for r in d[3]], n_min)
        print()

    print("v4 tracker by simulated skin tone (all light levels):")
    for tone in SKIN_TONES:
        group = [d for d in done if d[1] == tone]
        print_row("", tone, [r for d in group for r in d[3]], minutes * len(group))


def vanish_job(seed, tracker_settings=None):
    """
    Pulse present for 60 s, then replaced by pulse-free data with the same
    light and camera noise (think: lights switched off, or a photo of a face).
    Returns, for v3 and v4: (linger seconds, share of 80-120 s showing a number).
    """
    with_pulse = make_session("lighter", 0.6, np.random.default_rng(seed))
    without = make_session("lighter", 0.6, np.random.default_rng(seed + 999), pulse=False, events=False)
    t, patches, bg, face, moving, true_bpm = with_pulse
    cut = t >= 60
    patches = {n: np.where(cut[:, None], without[1][n][:t.size], patches[n]) for n in patches}
    bg = np.where(cut[:, None], without[2][:t.size], bg)
    session = (t, patches, bg, face, moving, true_bpm)
    out = []
    for display, gap, drop in ((LegacyDisplay(), 0.0, False),
                               (HeartRateTracker(**(tracker_settings or {})), 1.0, True)):
        res = stream(session, display, gap, drop_motion=drop)
        after = [(now, bpm) for now, bpm, _, _ in res if now >= 60]
        first_gone = next((now for now, bpm in after if bpm is None), 120.0)
        noise_only = [bpm is not None for now, bpm in after if now >= 80]   # window is 100% noise by now
        out.append((first_gone - 60, float(np.mean(noise_only))))
    return out


def vanish(sessions=24):
    """
    Does the display let go when the pulse genuinely disappears?
      linger:        seconds from the pulse vanishing until the number first
                     disappears. Partly unavoidable: the 15 s window still
                     holds real pulse for a while after the cut.
      false reading: share of time a number is shown once the window is pure
                     noise. This is the false-positive rate of the display.
    """
    from multiprocessing import Pool
    with Pool() as pool:
        res = np.array(pool.map(vanish_job, range(sessions)))      # (sessions, 2 logics, 2 metrics)
    print(f"Pulse removed at t = 60 s ({sessions} sessions).")
    print(f"{'display logic':<14}{'linger median':>15}{'linger worst':>14}{'false reading':>15}")
    for k, name in enumerate(("v3 (old)", "v4 tracker")):
        linger, false = res[:, k, 0], res[:, k, 1]
        print(f"{name:<14}{np.median(linger):>13.1f} s{linger.max():>12.1f} s{false.mean():>15.1%}")


def calibrate(windows=600, seed=3):
    """
    How often does PURE NOISE (no pulse at all) pass each tracker rule?
    This is how the thresholds are justified rather than guessed.
      enter rule: SNR >= SNR_ENTER anywhere in the band
      keep rule:  SNR >= SNR_KEEP AND the peak lands within GATE_BPM of a track
                  (track placed at a random heart rate, peak picker given it as a prior)
    """
    rng = np.random.default_rng(seed)
    trk = HeartRateTracker()
    enter = keep = 0
    snrs = []
    for k in range(windows):
        t, patches, bg, *_ = make_session("medium", 1.1, rng, seconds=WINDOW_SEC + 0.5,
                                           pulse=False, events=False)
        track = rng.uniform(55, 100)
        r = analyze_heart_rate(t, patches, bg, method="auto", prev_bpm=track, patch_map=False)
        r_free = analyze_heart_rate(t, patches, bg, method="auto", patch_map=False)
        snrs.append(r_free["snr_db"])
        enter += r_free["snr_db"] >= trk.SNR_ENTER
        keep += (r["snr_db"] >= trk.SNR_KEEP) and abs(r["bpm"] - track) <= trk.GATE_BPM
    snrs = np.array(snrs)
    print(f"Pure-noise windows: {windows}")
    print(f"  SNR percentiles   50%: {np.percentile(snrs, 50):+.1f} dB   90%: {np.percentile(snrs, 90):+.1f} dB"
          f"   99%: {np.percentile(snrs, 99):+.1f} dB")
    print(f"  false 'enter' rate (SNR >= {trk.SNR_ENTER:+.1f} dB):                       {enter / windows:.1%}")
    print(f"  false 'keep' rate  (SNR >= {trk.SNR_KEEP:+.1f} dB and within {trk.GATE_BPM:.0f} BPM of track): {keep / windows:.1%}")
    print("Note: these are per-window rates. The tracker also needs a reading to persist")
    print(f"{trk.ACQUIRE_SEC:.1f} s to start and a jump to persist {trk.CONFIRM_SEC:.0f} s, and a weak window only")
    print(f"counts within {trk.REFRESH_SEC:.0f} s of a strong one, so false READINGS on screen are rarer.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true", help="measure false-accept rates on pure noise")
    ap.add_argument("--vanish", action="store_true", help="test how fast the number disappears when the pulse does")
    ap.add_argument("--sessions", type=int, default=2, help="sessions per skin tone per light level")
    args = ap.parse_args()
    if args.calibrate:
        calibrate()
    elif args.vanish:
        vanish()
    else:
        main(args.sessions)
