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
      python session_benchmark.py --start      (how fast and how right is the first number?)
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

WINDOW_SEC, MIN_SEC, UPDATE_EVERY, MOTION_HOLD = 15.0, 5.0, 0.5, 1.5
NOISE_LEVELS = {"ok light": 0.6, "dim": 1.1, "very dim": 1.6}   # camera noise (brightness levels)


def make_session(tone, noise, rng, seconds=120, pulse=True, events=True, fidget=0.0, nonlinear=0.3):
    """
    Simulated per-frame data for one sitting. Returns a dict:
      t, patches {name: (N,3)}, background (N,3), face (N,), moving (N,),
      pose (N,4) as FaceMesh would see it, true_bpm (N,)

    fidget: 0 = sitting very still, 1 = normal fidgeting (small nods, sways and
    tilts, in bursts, like talking or shifting in a chair). Small head motion
    changes how much light each patch catches, so each patch's brightness
    wobbles with the head. Deliberately NOT exactly the model the motion
    correction assumes: the effect is partly non-linear, the pose FaceMesh
    reports is jittery and one frame late.
    nonlinear: size of the second-order shading term (same sign on every patch,
    like the cosine fall-off of real shading), relative to the linear term.
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

    # Small continuous head motion: 4 pose signals (x, y, scale, roll), in bursts.
    envelope = np.clip(band_noise(t, 0.02, 0.2, 1.0, rng) + 0.3, 0, None)
    head = np.column_stack([band_noise(t, 0.15, 2.0, 1.0, rng) for _ in range(4)]) * envelope[:, None]
    observed = np.roll(head, 1, axis=0) + rng.normal(0, 0.15, head.shape)   # 1 frame late + jitter

    skin, strength = SKIN_TONES[tone]
    patches = {}
    for name in PATCHES:
        amp = 0.0012 * strength * (1.0 if patch_region(name) == "forehead" else 0.7)
        shade_coef = rng.normal(0, 0.0015 * fidget, 4)        # how this patch's shading follows the head
        spec_coef = rng.normal(0, 0.0006 * fidget, 4)         # and its shine
        shading = 1 + head @ shade_coef - nonlinear * (head ** 2) @ np.abs(shade_coef)
        patch_shine = shine + head @ spec_coef
        clean = light * (skin * shading[:, None]
                         + amp * np.linalg.norm(skin) * wave[:, None] * PULSE_DIRECTION
                         + patch_shine[:, None] * np.linalg.norm(skin) * np.ones(3) / np.sqrt(3))
        patches[name] = clean + rng.normal(0, noise, clean.shape)
    background = light * WALL + rng.normal(0, noise / 3, (t.size, 3))
    return {"t": t, "patches": patches, "background": background, "face": face,
            "moving": moving, "pose": observed, "true_bpm": true_bpm}


def stream(session, display, max_gap_sec, drop_motion=True, use_motion=True, min_sec=None):
    """Run a simulated session through the app's logic; adds the true BPM to each update."""
    s = session
    res = simulate_display(s["t"], s["patches"], s["background"], s["face"], s["moving"], display,
                           max_gap_sec, drop_motion=drop_motion, window_sec=WINDOW_SEC,
                           min_sec=MIN_SEC if min_sec is None else min_sec,
                           update_every=UPDATE_EVERY, motion_hold=MOTION_HOLD,
                           pose=s["pose"] if use_motion else None)
    return [(now, bpm, state, float(np.interp(now, s["t"], s["true_bpm"]))) for now, bpm, state in res]


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


LOGICS = ("v3 (old)", "v4", "v5")   # what is compared in the main table


def run_one(job):
    """One session through every display logic (runs in a worker process)."""
    label, noise, motion, tone, seed, tracker_settings = job
    rng = np.random.default_rng(seed)
    # Fidgeting sessions get a random amount of non-linearity (mild to harsh), so the
    # motion correction is not only tested on the case it is designed for.
    session = make_session(tone, noise, rng, fidget=1.5 if motion == "fidgeting" else 0.0,
                           nonlinear=rng.uniform(0.1, 0.3))
    v4_settings = dict(tracker_settings or {}, SHORT_WINDOW_EXTRA_DB=0.0)
    runs = {
        "v3 (old)": stream(session, LegacyDisplay(), 0.0, drop_motion=False, use_motion=False, min_sec=8.0),
        "v4": stream(session, HeartRateTracker(**v4_settings), 1.0, use_motion=False, min_sec=8.0),
        "v5": stream(session, HeartRateTracker(**(tracker_settings or {})), 1.0, use_motion=True, min_sec=5.0),
    }
    return label, motion, tone, runs


def first_number(results):
    return next((now for now, bpm, _, _ in results if bpm is not None), None)


def print_row(label, name, results, minutes, first=None):
    m = score([r for r in results if r[0] >= 8.0], minutes)      # coverage counted from 8 s for all
    first_txt = f"{first:.1f} s" if first is not None else "-"
    print(f"{label:<22}{name:<10}{first_txt:>8}{m['coverage']:>9.0%}{m['drops/min']:>10.1f}{m['MAE']:>7.1f}"
          f"{m['<=5']:>9.0%}{m['>10']:>9.0%}{m['held']:>7.0%}")


def main(sessions_per_cell=3, seed=7, tracker_settings=None):
    from multiprocessing import Pool
    jobs = [(label, noise, motion, tone, seed * 1000 + i, tracker_settings or {})
            for label, noise in NOISE_LEVELS.items() for motion in ("still", "fidgeting")
            for tone in SKIN_TONES for i in range(sessions_per_cell)]
    with Pool() as pool:
        done = pool.map(run_one, jobs)
    minutes = (120 - 8.0) / 60

    print("Simulated 2-minute sessions, streamed like the live app (pooled over all sessions).")
    print("v3 = original display, v4 = tracker, v5 = tracker + motion correction + early start")
    print("first = median seconds to the first number; coverage = time a number is on screen (from 8 s);")
    print(">10 off = share of shown time more than 10 BPM wrong (a confident wrong number)\n")
    hdr = (f"{'condition':<22}{'logic':<10}{'first':>8}{'coverage':>9}{'drops/min':>10}{'MAE':>7}"
           f"{'<=5 BPM':>9}{'>10 off':>9}{'held':>7}")
    print(hdr + "\n" + "-" * len(hdr))
    for label in NOISE_LEVELS:
        for motion in ("still", "fidgeting"):
            group = [d for d in done if d[0] == label and d[1] == motion]
            for name in LOGICS:
                firsts = [first_number(d[3][name]) for d in group]
                firsts = [f for f in firsts if f is not None]
                print_row(f"{label}, {motion}", name, [r for d in group for r in d[3][name]],
                          minutes * len(group), np.median(firsts) if firsts else None)
            print()

    print("v5 by simulated skin tone (all conditions):")
    for tone in SKIN_TONES:
        group = [d for d in done if d[2] == tone]
        print_row("", tone, [r for d in group for r in d[3]["v5"]], minutes * len(group))


def vanish_job(seed, tracker_settings=None):
    """
    Pulse present for 60 s, then replaced by pulse-free data with the same
    light and camera noise (think: lights switched off, or a photo of a face).
    Returns, for v3 and v4: (linger seconds, share of 80-120 s showing a number).
    """
    session = make_session("lighter", 0.6, np.random.default_rng(seed))
    without = make_session("lighter", 0.6, np.random.default_rng(seed + 999), pulse=False, events=False)
    cut = session["t"] >= 60
    session["patches"] = {n: np.where(cut[:, None], without["patches"][n], v)
                          for n, v in session["patches"].items()}
    session["background"] = np.where(cut[:, None], without["background"], session["background"])
    out = []
    for display, gap, drop in ((LegacyDisplay(), 0.0, False),
                               (HeartRateTracker(**(tracker_settings or {})), 1.0, True)):
        res = stream(session, display, gap, drop_motion=drop, min_sec=8.0)
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
    for k, name in enumerate(("v3 (old)", "v5 (current)")):
        linger, false = res[:, k, 0], res[:, k, 1]
        print(f"{name:<14}{np.median(linger):>13.1f} s{linger.max():>12.1f} s{false.mean():>15.1%}")


START_CONFIGS = {   # name: (start analysing after, extra dB for short windows)
    "wait 8 s (v4)": (8.0, 0.0),
    "start at 5 s, no extra bar": (5.0, 0.0),
    "start at 5 s + stricter bar": (5.0, 1.0),
}


def start_job(args):
    name, noise, pulse, seed = args
    min_sec, extra = START_CONFIGS[name]
    s = make_session("medium", noise, np.random.default_rng(seed), seconds=25, pulse=pulse, events=False)
    res = stream(s, HeartRateTracker(SHORT_WINDOW_EXTRA_DB=extra), 1.0, min_sec=min_sec)
    first = next((now for now, bpm, _, _ in res if bpm is not None), None)
    right = next((abs(bpm - truth) <= 5 for _, bpm, _, truth in res if bpm is not None), None)
    shown = np.mean([bpm is not None for now, bpm, _, _ in res if now <= 20])
    return name, noise, pulse, first, right, shown


def start(sessions=60):
    """
    How fast does the first number appear, and is it right?
    With a pulse: seconds until the first number, and whether that first
    number is within 5 BPM. Without a pulse: share of the first 20 s showing a
    (necessarily false) number.
    """
    from multiprocessing import Pool
    jobs = [(name, noise, pulse, 4000 + i) for name in START_CONFIGS for noise in (0.6, 1.1)
            for pulse in (True, False) for i in range(sessions)]
    with Pool() as pool:
        res = pool.map(start_job, jobs)
    print(f"Medium skin tone, sitting still, {sessions} sessions per cell.")
    print(f"{'':<38}{'first number (median s)':>25}{'first number right':>20}{'false, no pulse':>17}")
    for label, noise in (("ok light", 0.6), ("dim", 1.1)):
        for name in START_CONFIGS:
            with_p = [r for r in res if r[0] == name and r[1] == noise and r[2]]
            without = [r for r in res if r[0] == name and r[1] == noise and not r[2]]
            firsts = [r[3] if r[3] is not None else 25.0 for r in with_p]
            right = [r[4] for r in with_p if r[4] is not None]
            print(f"{label + ': ' + name:<38}{np.median(firsts):>23.1f} s{np.mean(right):>20.0%}"
                  f"{np.mean([r[5] for r in without]):>17.1%}")
        print()


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
        sess = make_session("medium", 1.1, rng, seconds=WINDOW_SEC + 0.5, pulse=False, events=False)
        t, patches, bg = sess["t"], sess["patches"], sess["background"]
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
    ap.add_argument("--start", action="store_true", help="how fast and how right is the first number?")
    ap.add_argument("--vanish", action="store_true", help="test how fast the number disappears when the pulse does")
    ap.add_argument("--sessions", type=int, default=3, help="sessions per skin tone per condition")
    args = ap.parse_args()
    if args.calibrate:
        calibrate()
    elif args.vanish:
        vanish()
    elif args.start:
        start()
    else:
        main(args.sessions)
