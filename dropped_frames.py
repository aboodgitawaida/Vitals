"""
dropped_frames.py -- does the heart rate survive dropped camera frames? (validation split)

Browsers drop frames when the page is busy, and video conversions drop them on
a fixed pattern (30 -> 24 fps drops every 5th). With flickering room light
that can push a false rhythm into the heart-rate band (see vitals_dsp.deflicker).
This replays the saved features of every validation recording with frames
removed, and compares the network and the hand-made pipeline with and without
the flicker removal:

    python dropped_frames.py --data mcd_features

Every 15 s window, step 10 s. Frames are dropped before analysis; the
remaining frames keep their true times, as in the app.
"""

import argparse
import json
import os
import zlib

import numpy as np

import vitals_dsp
from build_mcd_dataset import load_features
from evaluate_video import read_ground_truth, reference_bpm

SCENARIOS = ("all frames", "30 -> 24 fps (every 5th dropped)", "30% dropped at random")


def keep(n, scenario, rng):
    if scenario == SCENARIOS[0]:
        return np.arange(n)
    if scenario == SCENARIOS[1]:
        return np.unique(np.round(np.arange(int(n * 24 / 29.92)) * 29.92 / 24).astype(int))
    return np.sort(rng.choice(n, int(n * 0.7), replace=False))


def run(folder):
    f = load_features(folder)
    t_gt, ppg, _ = read_ground_truth(os.path.join(folder, "ground_truth.txt"))
    rng = np.random.default_rng(zlib.crc32(os.path.basename(folder).encode()))
    rows = []
    for scenario in SCENARIOS:
        idx = keep(len(f["t"]), scenario, rng)
        t = f["t"][idx]
        for end in np.arange(15.0, t[-1], 10.0):
            sel = (t > end - 15) & (t <= end) & f["face"][idx]
            ref = reference_bpm(t_gt, ppg, end)
            if ref is None or sel.sum() < 150:
                continue
            tr = {n: v[idx][sel] for n, v in f["patches"].items()}
            for fix in (False, True):
                vitals_dsp.DEFLICKER = fix
                for method in ("net", "classic"):
                    r = vitals_dsp.analyze_heart_rate(t[sel], tr, f["background"][idx][sel], patch_map=False, method=method)
                    if r is not None:
                        rows.append((scenario, fix, method, abs(r["bpm"] - ref)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="mcd_features")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    with open(os.path.join(args.data, "splits.json")) as fh:
        people = json.load(fh)[args.split]
    folders = [os.path.join(args.data, f"{p}_{s}") for p in people for s in ("before", "after")
               if os.path.exists(os.path.join(args.data, f"{p}_{s}", "features.npz"))]
    from multiprocessing import get_context
    with get_context("spawn").Pool(args.jobs) as pool:
        rows = [r for rec in pool.map(run, folders) for r in rec]
    print(f"{len(folders)} recordings ({args.split}), 15 s windows\n")
    print("| Frames | Pipeline | without flicker removal: MAE / within 5 BPM | with: MAE / within 5 BPM |")
    print("| --- | --- | --- | --- |")
    for scenario in SCENARIOS:
        for method, label in (("net", "network"), ("classic", "hand-made")):
            cells = []
            for fix in (False, True):
                e = np.array([r[3] for r in rows if r[0] == scenario and r[1] == fix and r[2] == method])
                cells.append(f"{e.mean():.2f} / {np.mean(e <= 5):.0%}")
            print(f"| {scenario} | {label} | {cells[0]} | {cells[1]} |")


if __name__ == "__main__":
    main()
