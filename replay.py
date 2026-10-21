"""
replay.py -- re-run the pipeline on a recorded session and compare versions.

Record in the app with "s" (saves average colours per frame, never images),
note your real heart rate from a watch / pulse oximeter, then:

    python replay.py recordings/session_YYYYMMDD_HHMMSS.csv --ref 72

It slides a window across the recording (one estimate per second), runs every
pipeline configuration on EXACTLY the same data, and prints:
  coverage = % of windows where the method was confident enough to show a number
  MAE      = mean absolute error vs --ref, on the windows it showed
It also saves a plot next to the recording.

This is the honest way to claim "feature X helps": same input, one change,
measured against a reference. (A constant --ref assumes your heart rate was
steady while recording -- sit still and rest for a minute first. Phase 4 adds
proper time-aligned ground truth.)
"""

import argparse
import csv
import os

import numpy as np

from face_regions import patch_region
from vitals_dsp import analyze_heart_rate

CONFIGS = {   # name: (method, use patches?, background?, harmonics?)
    "green, 3 regions": ("green", False, False, False),
    "POS, 3 regions (v2)": ("pos", False, False, False),
    "POS + patches": ("pos", True, False, False),
    "POS + patches + bg": ("pos", True, True, False),
    "POS + all": ("pos", True, True, True),
    "green + all": ("green", True, True, True),
    "auto (v3)": ("auto", True, True, True),
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
    return times, patches, background


def merge_regions(patches):
    """Average patches into 3 big regions (what v2 did). Missing patches are skipped per frame."""
    groups = {}
    for name, rgb in patches.items():
        groups.setdefault(patch_region(name), []).append(rgb)
    out = {}
    for region, arrays in groups.items():
        stack = np.stack(arrays)
        valid = ~np.isnan(stack).any(axis=2)
        with np.errstate(invalid="ignore"):
            merged = np.nanmean(stack, axis=0)
        merged[valid.sum(axis=0) == 0] = np.nan
        out[region] = merged
    return out


def run_config(times, patches, background, config, window, step):
    method, use_patches, use_bg, use_harm = CONFIGS[config]
    traces = patches if use_patches else merge_regions(patches)
    results, prev = [], None
    start = times[0]
    while start + window <= times[-1]:
        sel = (times >= start) & (times < start + window)
        if sel.sum() > window * 15:             # need at least ~15 fps worth of frames
            r = analyze_heart_rate(times[sel], {n: v[sel] for n, v in traces.items()},
                                   background[sel] if background is not None else None,
                                   method=method, prev_bpm=prev,
                                   use_background=use_bg, use_harmonics=use_harm)
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

    times, patches, background = load_session(args.session)
    print(f"{args.session}: {times[-1] - times[0]:.0f} s, {len(times)} frames "
          f"({len(times) / (times[-1] - times[0]):.0f} fps), {len(patches)} patches, "
          f"background {'yes' if background is not None else 'no'}\n")

    all_results = {}
    print(f"{'configuration':<22}{'coverage':>10}{'median BPM':>12}" + (f"{'MAE':>8}{'<=5 BPM':>9}" if args.ref else ""))
    print("-" * (44 + (17 if args.ref else 0)))
    for config in CONFIGS:
        res = run_config(times, patches, background, config, args.window, args.step)
        all_results[config] = res
        shown = np.array([b for _, b, _, s in res if s])
        coverage = len(shown) / max(len(res), 1)
        line = f"{config:<22}{coverage:>10.0%}{(np.median(shown) if len(shown) else float('nan')):>12.1f}"
        if args.ref:
            err = np.abs(shown - args.ref)
            line += (f"{err.mean():>8.1f}{(err <= 5).mean():>9.0%}" if len(shown) else f"{'--':>8}{'--':>9}")
        print(line)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for config in ("green, 3 regions", "POS, 3 regions (v2)", "auto (v3)"):
            pts = [(t, b) for t, b, _, s in all_results[config] if s]
            if pts:
                ax.plot(*zip(*pts), ".", label=config, markersize=5)
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
