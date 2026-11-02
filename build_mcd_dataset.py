"""
build_mcd_dataset.py -- per-frame features for most of MCD-rPPG, for training and testing vitals_net.

For every recording: download the frontal-webcam video and its synchronised PPG,
extract the per-frame features exactly as the app measures them
(evaluate_video.extract_features: whole face, 16 patches, wall, head pose),
keep those (~1.3 MB) plus the ground truth, and delete the video. Several
recordings are processed at once; an interrupted run resumes where it stopped.

People are split once, at random with a fixed seed, and the split is saved:
    test        100 people   the 16 earlier test people (seed 1) + 84 more; used once, at the end
    validation   40 people   the 15 earlier development people (seed 0) + 25 more
    train       400 people   the rest of a random draw

    python build_mcd_dataset.py --out mcd_features --jobs 4          # ~5 h on 4 cores, ~1.5 GB kept
    python build_mcd_dataset.py --out mcd_features --only validation   # one split

Output: <out>/splits.json and <out>/<patient>_<before|after>/{features.npz, ground_truth.txt}
"""

import argparse
import csv
import json
import os
import random
import shutil
import tempfile
import time

import numpy as np

from fetch_mcd_rppg import CAMERA, choose, download, write_ground_truth

SPLIT_SEED = 2026
N_TEST, N_VAL, N_TRAIN = 100, 40, 400


def complete_patients(rows):
    """Patients with a frontal-webcam recording both before and after exercise."""
    steps = {}
    for r in rows:
        if r["camera"] == CAMERA:
            steps.setdefault(r["patient_id"], set()).add(r["step"])
    return sorted(p for p, s in steps.items() if {"before", "after"} <= s)


def make_splits(rows):
    """Fixed split by person. Keeps the earlier dev (seed 0) and test (seed 1) people where they were."""
    people = complete_patients(rows)
    old_dev = sorted({r["patient_id"] for r in choose(rows, 15, 0)})
    old_test = sorted({r["patient_id"] for r in choose(rows, 16, 1)})
    rest = [p for p in people if p not in old_dev and p not in old_test]
    random.Random(SPLIT_SEED).shuffle(rest)
    test = old_test + rest[:N_TEST - len(old_test)]
    rest = rest[N_TEST - len(old_test):]
    val = old_dev + rest[:N_VAL - len(old_dev)]
    train = rest[N_VAL - len(old_dev):N_VAL - len(old_dev) + N_TRAIN]
    return {"seed": SPLIT_SEED, "test": sorted(test), "validation": sorted(val), "train": sorted(train)}


def build_one(job):
    """Download one recording, extract features, delete the video. Returns (name, status)."""
    row, out, local_video = job
    name = f"{row['patient_id']}_{row['step']}"
    folder = os.path.join(out, name)
    feats_path = os.path.join(folder, "features.npz")
    if os.path.exists(feats_path):
        return name, "done already"
    os.makedirs(folder, exist_ok=True)
    from evaluate_video import extract_features      # imported here: workers load MediaPipe themselves
    tmp = tempfile.mkdtemp(prefix="mcd_")
    try:
        t0 = time.time()
        sync = os.path.join(tmp, "ppg_sync.txt")
        download(row["ppg_sync"], sync)
        with open(sync) as f:
            write_ground_truth(f.read(), float(row["pulse"] or "nan"), os.path.join(folder, "ground_truth.txt"))
        video = local_video
        if video is None:
            video = os.path.join(tmp, "vid.avi")
            download(row["video"], video)
        f = extract_features(video, cache_dir=None)
        np.savez_compressed(feats_path + ".part.npz", t=f["t"], background=f["background"], face=f["face"],
                            moving=f["moving"], pose=f["pose"], **{f"p_{n}": v for n, v in f["patches"].items()})
        os.replace(feats_path + ".part.npz", feats_path)
        return name, f"{time.time() - t0:.0f} s"
    except Exception as e:                            # one bad recording must not stop a 5-hour run
        return name, f"FAILED: {type(e).__name__}: {e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_features(folder):
    """features.npz -> the same dict evaluate_video.extract_features returns."""
    d = np.load(os.path.join(folder, "features.npz"))
    patches = {k[2:]: d[k] for k in d.files if k.startswith("p_")}
    return {"t": d["t"], "patches": patches, "background": d["background"], "face": d["face"],
            "moving": d["moving"], "pose": d["pose"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="mcd_features")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", choices=["validation", "train", "test"], action="append",
                    help="build only these splits (default: validation, train, test, in that order)")
    ap.add_argument("--local", action="append", default=[],
                    help="folders that already hold <patient>_<step>/vid.avi (skips those downloads)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    db = os.path.join(args.out, "db.csv")
    download("db.csv", db)
    with open(db, newline="") as f:
        rows = list(csv.DictReader(f))
    splits_path = os.path.join(args.out, "splits.json")
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            splits = json.load(f)
    else:
        splits = make_splits(rows)
        with open(splits_path, "w") as f:
            json.dump(splits, f, indent=1)
    print({k: len(v) for k, v in splits.items() if k != "seed"})

    frontal = {(r["patient_id"], r["step"]): r for r in rows if r["camera"] == CAMERA}
    jobs = []
    for split in args.only or ["validation", "train", "test"]:
        for p in splits[split]:
            for step in ("before", "after"):
                name = f"{p}_{step}"
                local = next((os.path.join(d, name, "vid.avi") for d in args.local
                              if os.path.exists(os.path.join(d, name, "vid.avi"))), None)
                jobs.append((frontal[(p, step)], args.out, local))
    print(f"{len(jobs)} recordings", flush=True)

    from multiprocessing import get_context
    t0 = time.time()
    with get_context("spawn").Pool(args.jobs) as pool:
        for i, (name, status) in enumerate(pool.imap_unordered(build_one, jobs), 1):
            eta = (time.time() - t0) / i * (len(jobs) - i) / 60
            print(f"{i}/{len(jobs)} {name}: {status}  (~{eta:.0f} min left)", flush=True)


if __name__ == "__main__":
    main()
