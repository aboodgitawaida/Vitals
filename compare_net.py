"""
compare_net.py -- the trained network vs the hand-made pipeline, on one split of build_mcd_dataset.py output.

Runs from the saved per-frame features (the videos are not kept), on every
recording of the split:
  * every 15 s and 8 s window (step 3 s): heart rate from the network and from
    the hand-made pipeline, against the finger PPG over the same 15 s;
  * the screen, replayed second by second with the app's tracker for both:
    steady and settling numbers, and the time to the first number;
  * 95% confidence intervals by resampling PEOPLE (not windows), since windows
    of one person are not independent; and results by face brightness.

    python compare_net.py --data mcd_features --split validation
    python compare_net.py --data mcd_features --split test --out docs/validation/net_test   # once, at the end
"""

import argparse
import json
import os

import numpy as np

from build_mcd_dataset import load_features
from evaluate_video import read_ground_truth, reference_bpm
from face_layout import FACE_TRACE
from vitals_dsp import analyze_heart_rate
from vitals_stream import HeartRateTracker, simulate_display

WINDOWS = (15.0, 8.0)
STEP = 3.0
FIRST_SEC = 75.0          # time to first number: within the first 75 s of each recording
PIPELINES = {"network": "net", "hand-made": "classic"}


def run_recording(job):
    """One recording -> per-window estimates, replayed screens and brightness."""
    folder, tracker_kw = job
    f = load_features(folder)
    t_gt, ppg, _ = read_ground_truth(os.path.join(folder, "ground_truth.txt"))
    t = f["t"]
    face_rgb = np.nanmean(f["patches"][FACE_TRACE], axis=0)
    out = {"name": os.path.basename(folder), "brightness": float(0.299 * face_rgb[0] + 0.587 * face_rgb[1] + 0.114 * face_rgb[2]),
           "windows": {}, "screen": {}}
    for w in WINDOWS:
        rows = []
        for end in np.arange(15.0, t[-1], STEP):
            sel = (t > end - w) & (t <= end) & f["face"]
            ref = reference_bpm(t_gt, ppg, end)
            if ref is None or sel.sum() < w * 10:
                continue
            tr = {n: v[sel] for n, v in f["patches"].items()}
            est = {}
            for label, method in PIPELINES.items():
                r = analyze_heart_rate(t[sel], tr, f["background"][sel], patch_map=False, method=method)
                est[label] = None if r is None else r["bpm"]
                if label == "network":
                    est["confidence"] = None if r is None else r.get("confidence")
            rows.append({"end": float(end), "ref": ref, **est})
        out["windows"][str(w)] = rows
    for label, method in PIPELINES.items():
        kw = tracker_kw if method == "net" else {}
        shown = simulate_display(t, f["patches"], f["background"], f["face"], f["moving"],
                                 HeartRateTracker(**kw), pose=f["pose"], method=method, with_settling=True)
        rows = []
        for now, b, _, settling in shown:
            ref = reference_bpm(t_gt, ppg, max(now, 10.0), window=min(15.0, max(now, 10.0)))
            if ref is not None:
                rows.append((float(now), b, ref, bool(settling)))
        out["screen"][label] = rows
    return out


def person(name):
    return name.rsplit("_", 1)[0]


def bootstrap(recs, stat, n=1000, seed=0):
    """95% interval of stat(list of recordings), resampling people with replacement."""
    people = sorted({person(r["name"]) for r in recs})
    by_person = {p: [r for r in recs if person(r["name"]) == p] for p in people}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sample = [r for p in rng.choice(people, len(people)) for r in by_person[p]]
        v = stat(sample)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def window_errors(recs, w, label):
    return np.array([abs(row[label] - row["ref"]) for r in recs for row in r["windows"][str(w)] if row[label] is not None])


def screen_stats(recs, label):
    rows = [x for r in recs for x in r["screen"][label] if x[0] >= 8.0]
    steady = np.array([abs(b - ref) for _, b, ref, s in rows if b is not None and not s])
    settling = np.array([abs(b - ref) for _, b, ref, s in rows if b is not None and s])
    firsts = []
    for r in recs:
        shown = [now for now, b, _, _ in r["screen"][label] if b is not None and now <= FIRST_SEC]
        firsts.append(shown[0] if shown else np.inf)
    n = max(len(rows), 1)
    return {"steady_share": len(steady) / n, "settling_share": len(settling) / n,
            "steady_MAE": steady.mean() if len(steady) else np.nan,
            "steady_within5": (steady <= 5).mean() if len(steady) else np.nan,
            "settling_within5": (settling <= 5).mean() if len(settling) else np.nan,
            "first_median": float(np.median(firsts)), "first_by15": float(np.mean(np.array(firsts) <= 15)),
            "never": float(np.mean(np.isinf(firsts)))}


def report(recs):
    lines = [f"{len(recs)} recordings, {len({person(r['name']) for r in recs})} people. 95% intervals resample people.", ""]
    lines += ["| Window | Pipeline | MAE (BPM) [95% CI] | Within 5 BPM [95% CI] |", "| --- | --- | --- | --- |"]
    for w in WINDOWS:
        for label in PIPELINES:
            e = window_errors(recs, w, label)
            lo, hi = bootstrap(recs, lambda s, w=w, label=label: window_errors(s, w, label).mean())
            plo, phi = bootstrap(recs, lambda s, w=w, label=label: (window_errors(s, w, label) <= 5).mean())
            lines.append(f"| {w:.0f} s | {label} | {e.mean():.2f} [{lo:.2f}-{hi:.2f}] | {np.mean(e <= 5):.0%} [{plo:.0%}-{phi:.0%}] |")
    lines += ["", "| What the app shows | network | hand-made |", "| --- | --- | --- |"]
    st = {label: screen_stats(recs, label) for label in PIPELINES}
    for key, fmt in [("steady_share", "{:.0%}"), ("steady_within5", "{:.0%}"), ("steady_MAE", "{:.2f}"),
                     ("settling_share", "{:.0%}"), ("settling_within5", "{:.0%}"),
                     ("first_median", "{:.1f} s"), ("first_by15", "{:.0%}"), ("never", "{:.0%}")]:
        lines.append(f"| {key.replace('_', ' ')} | {fmt.format(st['network'][key])} | {fmt.format(st['hand-made'][key])} |")
    lines += ["", "| Face brightness | Recordings | network: within 5 BPM (15 s) | hand-made |", "| --- | --- | --- | --- |"]
    for lo, hi in [(0, 80), (80, 95), (95, 256)]:
        sub = [r for r in recs if lo <= r["brightness"] < hi]
        if sub:
            net_ok = np.mean(window_errors(sub, 15.0, "network") <= 5)
            hand_ok = np.mean(window_errors(sub, 15.0, "hand-made") <= 5)
            lines.append(f"| {lo}-{hi - 1 if hi < 256 else 255} | {len(sub)} | {net_ok:.0%} | {hand_ok:.0%} |")
    rows15 = [row for r in recs for row in r["windows"]["15.0"] if row["confidence"] is not None]
    conf = np.array([row["confidence"] for row in rows15])
    ok = np.array([abs(row["network"] - row["ref"]) <= 5 for row in rows15])
    lines += ["", "Network confidence vs how often it is right (15 s windows):", "",
              "| Confidence | Windows | Within 5 BPM |", "| --- | --- | --- |"]
    for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.any():
            lines.append(f"| {lo:.1f}-{min(hi, 1):.1f} | {m.sum()} | {ok[m].mean():.0%} |")
    return "\n".join(lines) + "\n", st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="mcd_features")
    ap.add_argument("--split", default="validation", choices=["validation", "test", "train"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--tracker", default="{}", help="HeartRateTracker overrides for the network, as JSON")
    ap.add_argument("--out", default=None, help="folder for summary.md and results.json")
    args = ap.parse_args()
    with open(os.path.join(args.data, "splits.json")) as f:
        people = json.load(f)[args.split]
    folders = [os.path.join(args.data, f"{p}_{s}") for p in people for s in ("before", "after")
               if os.path.exists(os.path.join(args.data, f"{p}_{s}", "features.npz"))]
    jobs = [(f, json.loads(args.tracker)) for f in folders]
    from multiprocessing import get_context
    with get_context("spawn").Pool(args.jobs) as pool:
        recs = pool.map(run_recording, jobs)
    text, _ = report(recs)
    print(text)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "summary.md"), "w") as f:
            f.write(text)
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(recs, f)


if __name__ == "__main__":
    main()
