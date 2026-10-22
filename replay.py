"""
replay.py -- re-run the pipeline on a recorded session and compare versions.

Record in the app with "s" (saves average colours per frame, never images),
note your real heart rate from a watch / pulse oximeter, then:

    python replay.py recordings/session_YYYYMMDD_HHMMSS.csv --ref 72

Two tables, both on EXACTLY the same data:

1. Signal processing: slides a 15 s window across the recording (one estimate
   per second) and runs every pipeline configuration.
     coverage = % of windows confident enough to show a number (SNR >= 0 dB)
     MAE      = mean absolute error vs --ref, on the windows it showed

2. What the screen would have shown: streams the recording frame by frame
   through the old v3 display logic and the v4 tracker (vitals_stream.py),
   the same code the app runs.
     first number = seconds from the start of the recording to the first number
     coverage  = % of the time a number was on screen (from 8 s on)
     drops/min = how often the number vanished
     held      = share of shown time that was a held value

It also saves a plot next to the recording.

This is the honest way to claim "feature X helps": same input, one change,
measured against a reference. (A constant --ref assumes your heart rate was
steady while recording -- sit still and rest for a minute first. Phase 4 adds
proper time-aligned ground truth.)
"""

import argparse
import csv
import os
import warnings

import numpy as np

from face_regions import patch_region
from vitals_dsp import analyze_heart_rate
from vitals_stream import POSE_FIELDS, HeartRateTracker, LegacyDisplay, simulate_display

CONFIGS = {   # name: (method, use patches?, background?, harmonics?, motion correction?)
    "green, 3 regions": ("green", False, False, False, False),
    "POS, 3 regions (v2)": ("pos", False, False, False, False),
    "POS + patches": ("pos", True, False, False, False),
    "POS + patches + bg": ("pos", True, True, False, False),
    "POS + all": ("pos", True, True, True, False),
    "green + all": ("green", True, True, True, False),
    "auto (v3/v4)": ("auto", True, True, True, False),
    "green + motion corr": ("green", True, True, True, True),
    "auto + motion corr (v5)": ("auto", True, True, True, True),
}
SNR_MIN_DB = 0.0


def load_session(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    col = {name: i for i, name in enumerate(header)}
    times = np.array([float(r[col["time_s"]]) for r in data])
    names = sorted({h[:-2] for h in header if h.endswith("_r")})

    def series(name):
        out = np.full((len(data), 3), np.nan)
        for i, r in enumerate(data):
            vals = [r[col[f"{name}_{c}"]] for c in "rgb"]
            if all(vals):
                out[i] = [float(v) for v in vals]
        return out

    patches = {n: series(n) for n in names if n != "background"}
    background = series("background") if "background" in names else None

    # v4 recordings also store whether a face was found and whether the head
    # moved fast, per frame. Older recordings: assume face present, no motion.
    def flag(name, default):
        if name not in col:
            return np.full(len(data), default)
        return np.array([r[col[name]] == "1" for r in data])

    face = flag("face", True)
    moving = flag("moving", False)

    # v5 recordings also store head pose (for motion correction).
    # (v5 recordings have 4 pose signals; later ones add mouth opening.)
    pose = None
    fields = [k for k in POSE_FIELDS if f"pose_{k}" in col]
    if fields:
        pose = np.full((len(data), len(fields)), np.nan)
        for i, r in enumerate(data):
            vals = [r[col[f"pose_{k}"]] for k in fields]
            if all(vals):
                pose[i] = [float(v) for v in vals]
    return times, patches, background, face, moving, pose


def merge_regions(patches):
    """Average patches into 3 big regions (what v2 did). Missing patches are skipped per frame."""
    groups = {}
    for name, rgb in patches.items():
        groups.setdefault(patch_region(name), []).append(rgb)
    out = {}
    for region, arrays in groups.items():
        stack = np.stack(arrays)
        valid = ~np.isnan(stack).any(axis=2)
        with warnings.catch_warnings():               # frames with no patch at all -> NaN, expected
            warnings.simplefilter("ignore", RuntimeWarning)
            merged = np.nanmean(stack, axis=0)
        merged[valid.sum(axis=0) == 0] = np.nan
        out[region] = merged
    return out


def run_config(times, patches, background, pose, config, window, step):
    method, use_patches, use_bg, use_harm, use_motion = CONFIGS[config]
    traces = patches if use_patches else merge_regions(patches)
    results, prev = [], None
    start = times[0]
    while start + window <= times[-1]:
        sel = (times >= start) & (times < start + window)
        if sel.sum() > window * 15:             # need at least ~15 fps worth of frames
            r = analyze_heart_rate(times[sel], {n: v[sel] for n, v in traces.items()},
                                   background[sel] if background is not None else None,
                                   method=method, prev_bpm=prev,
                                   use_background=use_bg, use_harmonics=use_harm, patch_map=False,
                                   pose=pose[sel] if (use_motion and pose is not None) else None)
            if r is not None:
                shown = r["snr_db"] >= SNR_MIN_DB
                prev = r["bpm"] if shown else None
                results.append((start + window, r["bpm"], r["snr_db"], shown))
        start += step
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", help="CSV recorded with 's' in vitals_app.py")
    parser.add_argument("--ref", type=float, help="reference heart rate (BPM) from a watch / oximeter")
    parser.add_argument("--window", type=float, default=15.0, help="seconds per estimate (default 15)")
    parser.add_argument("--step", type=float, default=1.0, help="seconds between estimates (default 1)")
    args = parser.parse_args()

    times, patches, background, face, moving, pose = load_session(args.session)
    duration = times[-1] - times[0]
    print(f"{args.session}: {duration:.0f} s, {len(times)} frames "
          f"({len(times) / duration:.0f} fps), {len(patches)} patches, "
          f"background {'yes' if background is not None else 'no'}, "
          f"no-face frames {(~face).mean():.1%}, fast-motion frames {moving.mean():.1%}, "
          f"head pose {'yes' if pose is not None else 'no (recorded before v5)'}\n")
    print("1. SIGNAL PROCESSING (each 15 s window on its own)")

    all_results = {}
    print(f"{'configuration':<26}{'median SNR':>11}{'coverage':>10}{'median BPM':>12}"
          + (f"{'MAE':>8}{'<=5 BPM':>9}" if args.ref else ""))
    print("-" * (59 + (17 if args.ref else 0)))
    for config in CONFIGS:
        res = run_config(times, patches, background, pose, config, args.window, args.step)
        all_results[config] = res
        shown = np.array([b for _, b, _, s in res if s])
        coverage = len(shown) / max(len(res), 1)
        med_snr = np.median([snr for _, _, snr, _ in res]) if res else float("nan")
        line = (f"{config:<26}{med_snr:>+10.1f} {coverage:>10.0%}"
                f"{(np.median(shown) if len(shown) else float('nan')):>12.1f}")
        if args.ref:
            err = np.abs(shown - args.ref)
            line += (f"{err.mean():>8.1f}{(err <= 5).mean():>9.0%}" if len(shown) else f"{'--':>8}{'--':>9}")
        print(line)

    print("\n2. WHAT THE SCREEN WOULD HAVE SHOWN (frame-by-frame, auto method)")
    screen = {
        "v3 display (old)": simulate_display(times, patches, background, face, moving,
                                             LegacyDisplay(), max_gap_sec=0.0, drop_motion=False,
                                             min_sec=8.0),
        "v4 tracker": simulate_display(times, patches, background, face, moving,
                                       HeartRateTracker(SHORT_WINDOW_EXTRA_DB=0.0), min_sec=8.0),
        "v5 (+motion, early)": simulate_display(times, patches, background, face, moving,
                                                HeartRateTracker(), pose=pose),
    }
    print(f"{'display logic':<22}{'first number':>13}{'coverage':>10}{'drops/min':>11}{'held':>7}"
          + (f"{'MAE':>8}{'<=5 BPM':>9}{'MAE locked':>12}" if args.ref else ""))
    print("-" * (63 + (29 if args.ref else 0)))
    for name, res in screen.items():
        first = next((t - times[0] for t, b, _ in res if b is not None), None)
        res = [r for r in res if r[0] - times[0] >= 8.0]          # coverage counted from 8 s for all
        shown = [(b, s) for _, b, s in res if b is not None]
        flags = [b is not None for _, b, _ in res]
        drops = sum(1 for a, b in zip(flags, flags[1:]) if a and not b)
        held = np.mean([s == "holding" for _, s in shown]) if shown else 0.0
        line = (f"{name:<22}{(f'{first:.1f} s' if first is not None else 'never'):>13}"
                f"{len(shown) / max(len(res), 1):>10.0%}"
                f"{drops / max(duration / 60, 1e-9):>11.1f}{held:>7.0%}")
        if args.ref:
            if shown:
                err = np.abs(np.array([b for b, _ in shown]) - args.ref)
                locked = [abs(b - args.ref) for b, s in shown if s == "locked"]
                line += (f"{err.mean():>8.1f}{(err <= 5).mean():>9.0%}"
                         f"{(np.mean(locked) if locked else float('nan')):>12.1f}")
            else:
                line += f"{'--':>8}{'--':>9}{'--':>12}"
        print(line)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for config in ("POS, 3 regions (v2)", "auto (v3/v4)", "auto + motion corr (v5)"):
            pts = [(t, b) for t, b, _, s in all_results[config] if s]
            if pts:
                ax.plot(*zip(*pts), ".", label=f"window: {config}", markersize=4, alpha=0.6)
        # What v4 would have put on screen: solid where locked, faded where held.
        for state, alpha in (("locked", 1.0), ("holding", 0.35)):
            pts = [(t, b) for t, b, s in screen["v5 (+motion, early)"] if b is not None and s == state]
            if pts:
                ax.plot(*zip(*pts), "s", color="tab:red", markersize=3, alpha=alpha,
                        label=f"on screen (v5): {state}")
        if args.ref:
            ax.axhline(args.ref, color="k", linestyle="--", linewidth=1, label=f"reference {args.ref:.0f}")
        # Always show at least a 40 BPM range so tiny differences don't look dramatic
        values = [b for res in all_results.values() for _, b, _, s in res if s] + ([args.ref] if args.ref else [])
        if values:
            lo, hi = min(values), max(values)
            pad = max(0.0, (40 - (hi - lo)) / 2) + 3
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("time in recording (s)")
        ax.set_ylabel("estimated heart rate (BPM)")
        ax.set_title("Camera-based heart-rate estimates (only windows above the SNR threshold)")
        ax.legend()
        ax.grid(alpha=0.3)
        out = os.path.splitext(args.session)[0] + "_replay.png"
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        print(f"\nplot saved: {out}")
    except ImportError:
        print("\n(matplotlib not installed: skipping plot)")


if __name__ == "__main__":
    main()
