"""
build_mcd_dataset.py -- per-frame features for most of MCD-rPPG, for training and testing vitals_net.

For every recording: download the frontal-webcam video and its synchronised PPG,
extract the per-frame features exactly as the app measures them
(evaluate_video.extract_features: whole face, 16 patches, wall, head pose),
keep those (~1.3 MB) plus the ground truth, and delete the video. Several
recordings are processed at once; an interrupted run resumes where it stopped.
Each worker needs ~3 GB (MediaPipe); if one is killed for lack of memory, its
recording is lost and the run never finishes: stop it and run it again.

People are split once, at random with a fixed seed, and the split is saved:
    test        100 people   the 16 earlier test people (seed 1) + 84 more; used once, at the end
    validation   40 people   the 15 earlier development people (seed 0) + 25 more
    train       400 people   the rest of a random draw

    python build_mcd_dataset.py --out mcd_features --jobs 4          # ~5 h on 4 cores, ~1.5 GB kept
    python build_mcd_dataset.py --out mcd_features --only validation   # one split

Harder conditions (hard_conditions.py), same people and splits:
    python build_mcd_dataset.py --out mcd_left --camera USBVideo --splits mcd_features/splits.json --only validation
    python build_mcd_dataset.py --out mcd_low --transform low --front mcd_features --only validation
The dataset filmed everyone with three cameras at once (front, left, right), each with
its own frame-synced PPG. "--transform low" moves the frontal video down so the face
sits at the bottom edge with the chin out of the picture (a laptop camera looked
down at); --front gives the frontal features, whose head pose sets how far.

Frame times come from the true frame rate in the dataset's frame timestamps
(fetch_mcd_rppg.true_fps), not from the video header, which is wrong for about a
quarter of the videos. Folders built before that fix are corrected in place,
without redoing the face tracking:
    python build_mcd_dataset.py --out mcd_features --retime

Output: <out>/splits.json and <out>/<patient>_<before|after>/{features.npz, ground_truth.txt, fps.txt}
"""

import argparse
import csv
import json
import os
import random
import shutil
import tempfile
import time

import subprocess

import numpy as np

from fetch_mcd_rppg import CAMERA, choose, download, fetch_text, read_fps, save_fps, true_fps, write_ground_truth

CAMERAS = ("FullHDwebcam", "USBVideo", "IriunWebcam")    # front, left, right
LOW_NOSE_Y = 0.93          # --transform low: the nose ends up this far down the picture

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
    row, out, local_video, shift = job
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
        fps = true_fps(fetch_text(row["meta"]))
        write_ground_truth(fetch_text(row["ppg_sync"]), float(row["pulse"] or "nan"),
                           os.path.join(folder, "ground_truth.txt"), fps)
        video = local_video
        if video is None:
            video = os.path.join(tmp, "vid.avi")
            download(row["video"], video)
        if shift:                                    # --transform low: face at the bottom edge
            moved = os.path.join(tmp, "low.avi")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", video, "-vf",
                            f"pad=iw:ih+{shift}:0:{shift}:color=gray,crop=iw:ih-{shift}:0:0",
                            "-c:v", "ffv1", moved], check=True)
            video = moved
        f = extract_features(video, cache_dir=None, fps=fps)
        np.savez_compressed(feats_path + ".part.npz", t=f["t"], background=f["background"], face=f["face"],
                            moving=f["moving"], pose=f["pose"], **{f"p_{n}": v for n, v in f["patches"].items()})
        os.replace(feats_path + ".part.npz", feats_path)
        save_fps(folder, fps)
        return name, f"{time.time() - t0:.0f} s"
    except Exception as e:                            # one bad recording must not stop a 5-hour run
        return name, f"FAILED: {type(e).__name__}: {e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def retime(job):
    """A folder built from the video header's frame rate -> true frame times. Returns (name, status)."""
    row, folder = job
    name = os.path.basename(folder)
    feats_path = os.path.join(folder, "features.npz")
    if not os.path.exists(feats_path):
        return name, "no features"
    if read_fps(folder):
        return name, "done already"
    try:
        fps = true_fps(fetch_text(row["meta"]))
        sync = fetch_text(row["ppg_sync"])
        d = dict(np.load(feats_path))
        old = 1 / np.median(np.diff(d["t"]))
        d["t"] = np.arange(len(d["t"])) / fps
        write_ground_truth(sync, float(row["pulse"] or "nan"), os.path.join(folder, "ground_truth.txt"), fps)
        np.savez_compressed(feats_path + ".part.npz", **d)
        os.replace(feats_path + ".part.npz", feats_path)
        save_fps(folder, fps)                       # last: marks the folder as done
        return name, f"header {old:.2f} fps -> true {fps:.2f} fps"
    except Exception as e:
        return name, f"FAILED: {type(e).__name__}: {e}"


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
    ap.add_argument("--camera", choices=CAMERAS, default=CAMERA, help="front (default), left (USBVideo), right (IriunWebcam)")
    ap.add_argument("--splits", default=None, help="reuse this splits.json (same people in each split)")
    ap.add_argument("--transform", choices=["low"], default=None, help="low: face at the bottom edge, chin cut off")
    ap.add_argument("--front", default=None, help="frontal features folder (head pose for --transform low)")
    ap.add_argument("--retime", action="store_true", help="fix frame times in folders already built (no face tracking)")
    args = ap.parse_args()
    if args.transform and (args.camera != CAMERA or not args.front):
        ap.error("--transform low needs the front camera and --front")

    os.makedirs(args.out, exist_ok=True)
    db = os.path.join(args.out, "db.csv")
    download("db.csv", db)
    with open(db, newline="") as f:
        rows = list(csv.DictReader(f))
    splits_path = os.path.join(args.out, "splits.json")
    if args.splits and not os.path.exists(splits_path):
        shutil.copy(args.splits, splits_path)
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            splits = json.load(f)
    else:
        splits = make_splits(rows)
        with open(splits_path, "w") as f:
            json.dump(splits, f, indent=1)
    print({k: len(v) for k, v in splits.items() if k != "seed"})

    frontal = {(r["patient_id"], r["step"]): r for r in rows if r["camera"] == args.camera}
    if args.retime:
        from concurrent.futures import ThreadPoolExecutor   # downloads only: threads are enough
        names = sorted(n for n in os.listdir(args.out) if os.path.isdir(os.path.join(args.out, n)))
        jobs = [(frontal[tuple(n.split("_"))], os.path.join(args.out, n)) for n in names
                if tuple(n.split("_")) in frontal]
        with ThreadPoolExecutor(16) as ex:
            for i, (name, status) in enumerate(ex.map(retime, jobs), 1):
                print(f"{i}/{len(jobs)} {name}: {status}", flush=True)
        return
    jobs = []
    for split in args.only or ["validation", "train", "test"]:
        for p in splits[split]:
            for step in ("before", "after"):
                name = f"{p}_{step}"
                local = next((os.path.join(d, name, "vid.avi") for d in args.local
                              if os.path.exists(os.path.join(d, name, "vid.avi"))), None)
                if args.camera != CAMERA:
                    local = None                             # local copies are frontal videos
                shift = 0
                if args.transform == "low":
                    nose_y = np.nanmedian(load_features(os.path.join(args.front, name))["pose"][:, 1])
                    shift = max(0, int(round((LOW_NOSE_Y - nose_y) * 480 / 2)) * 2)   # 640x480 webcam; even
                if (p, step) in frontal:
                    jobs.append((frontal[(p, step)], args.out, local, shift))
    print(f"{len(jobs)} recordings", flush=True)

    from multiprocessing import get_context
    t0 = time.time()
    with get_context("spawn").Pool(args.jobs) as pool:
        for i, (name, status) in enumerate(pool.imap_unordered(build_one, jobs), 1):
            eta = (time.time() - t0) / i * (len(jobs) - i) / 60
            print(f"{i}/{len(jobs)} {name}: {status}  (~{eta:.0f} min left)", flush=True)


if __name__ == "__main__":
    main()
