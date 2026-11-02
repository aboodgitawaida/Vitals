"""
hard_conditions.py -- how the app does when the camera is not straight in front of you.

MCD-rPPG filmed every person with three cameras at once, each with its own
frame-synced finger PPG: front (what the network was trained on), left and
right (smaller, turned face, other light; the right one at 24 fps). Plus the
front video moved down so the chin is out of the picture, like a laptop camera
looking down at someone (build_mcd_dataset.py --transform low). Same people,
same split, so the conditions compare directly:

    python hard_conditions.py --split validation \
        --cond front=mcd_features --cond left=mcd_left --cond right=mcd_right --cond "chin cut off"=mcd_low

Per condition: how often a face is found, fast motion, start-overs per minute
(face lost longer than --gap seconds wipes the buffer, as in the app), what the
screen shows (network + tracker, replayed second by second), and every 15 s
window for the network and the hand-made pipeline.
"""

import argparse
import json
import os

import numpy as np

from build_mcd_dataset import load_features
from compare_net import FIRST_SEC, bootstrap, screen_stats
from evaluate_video import read_ground_truth, reference_bpm
from face_layout import FACE_SIZE_MIN, ROLL_MAX
from live_session import FACE_GAP_SEC, start_sec
from vitals_dsp import analyze_heart_rate
from vitals_stream import HeartRateTracker, simulate_display

STEP = 5.0


def restarts(t, face, gap):
    """Start-overs: the face missing for longer than gap while there was data (SampleBuffer.add_missing)."""
    n, last_seen = 0, None
    for now, ok in zip(t, face):
        if ok:
            last_seen = now
        elif last_seen is not None and now - last_seen > gap:
            n += 1
            last_seen = None
    return n


def run(job):
    folder, gap = job
    f = load_features(folder)
    t_gt, ppg, _ = read_ground_truth(os.path.join(folder, "ground_truth.txt"))
    t, face = f["t"], f["face"]
    pose = f["pose"][face]
    out = {"name": os.path.basename(folder), "minutes": (t[-1] - t[0]) / 60, "face": float(face.mean()),
           "moving": float(f["moving"][face].mean()) if face.any() else 0.0,
           "framing": float(np.mean((pose[:, 2] < FACE_SIZE_MIN) | (np.abs(pose[:, 3]) > ROLL_MAX))) if face.any() else 1.0,
           "restarts": restarts(t, face, gap), "windows": []}
    for end in np.arange(15.0, t[-1], STEP):
        sel = (t > end - 15) & (t <= end) & face & ~f["moving"]
        ref = reference_bpm(t_gt, ppg, end)
        if ref is None or sel.sum() < 150:
            continue
        row = {"ref": ref}
        for label, method in (("network", "net"), ("hand-made", "classic")):
            r = analyze_heart_rate(t[sel], {n: v[sel] for n, v in f["patches"].items()}, f["background"][sel],
                                   patch_map=False, method=method)
            row[label] = None if r is None else r["bpm"]
        out["windows"].append(row)
    shown = simulate_display(t, f["patches"], f["background"], face, f["moving"], HeartRateTracker(),
                             max_gap_sec=gap, pose=f["pose"], method="net", with_settling=True,
                             min_sec=start_sec("net"))
    rows = []
    for now, b, _, settling in shown:
        ref = reference_bpm(t_gt, ppg, max(now, 10.0), window=min(15.0, max(now, 10.0)))
        if ref is not None:
            rows.append((float(now), b, ref, bool(settling)))
    out["screen"] = {"network": rows}
    return out


def window_errors(recs, label):
    return np.array([abs(w[label] - w["ref"]) for r in recs for w in r["windows"] if w[label] is not None])


def report(results, gap):
    lines = [f"Start-over after a face loss longer than {gap:g} s. 95% intervals resample people.", "",
             "| Condition | Recordings | Face found | Fast motion | Too small / tilted | Start-overs per min | "
             "Steady number shown | Steady within 5 BPM | Steady MAE | First number (median) | "
             "15 s windows within 5 BPM: network [95% CI] | hand-made |",
             "| " + " | ".join(["---"] * 12) + " |"]
    for cond, recs in results.items():
        st = screen_stats(recs, "network")
        e_net, e_hand = window_errors(recs, "network"), window_errors(recs, "hand-made")
        lo, hi = bootstrap(recs, lambda s: (window_errors(s, "network") <= 5).mean())
        mins = sum(r["minutes"] for r in recs)
        lines.append(
            f"| {cond} | {len(recs)} | {np.mean([r['face'] for r in recs]):.0%} | {np.mean([r['moving'] for r in recs]):.0%} | "
            f"{np.mean([r['framing'] for r in recs]):.0%} | {sum(r['restarts'] for r in recs) / mins:.2f} | "
            f"{st['steady_share']:.0%} | {st['steady_within5']:.0%} | {st['steady_MAE']:.1f} | "
            f"{st['first_median']:.1f} s (never {st['never']:.0%}) | "
            f"{np.mean(e_net <= 5):.0%} [{lo:.0%}-{hi:.0%}] | {np.mean(e_hand <= 5):.0%} |")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cond", action="append", required=True, help="name=features folder (build_mcd_dataset.py output)")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--gap", type=float, action="append", help=f"face-loss bridge in seconds (default {FACE_GAP_SEC})")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-people", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the tables here (markdown)")
    args = ap.parse_args()
    from multiprocessing import get_context
    text = [f"# Harder conditions ({args.split} split)", "",
            f"`python hard_conditions.py --split {args.split} " + " ".join(f'--cond "{c}"' for c in args.cond) + "`", ""]
    for gap in args.gap or [FACE_GAP_SEC]:
        results = {}
        for spec in args.cond:
            name, folder = spec.split("=", 1)
            with open(os.path.join(folder, "splits.json")) as fh:
                people = json.load(fh)[args.split]
            if args.max_people:
                people = people[:args.max_people]
            folders = [os.path.join(folder, f"{p}_{s}") for p in people for s in ("before", "after")
                       if os.path.exists(os.path.join(folder, f"{p}_{s}", "features.npz"))]
            with get_context("spawn").Pool(args.jobs) as pool:
                results[name] = pool.map(run, [(fo, gap) for fo in folders])
        text += report(results, gap) + [""]
        print("\n".join(report(results, gap)), flush=True)
    text.append(f"First number: within the first {FIRST_SEC:.0f} s of each recording.")
    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
