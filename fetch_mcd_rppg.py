"""
fetch_mcd_rppg.py -- download part of the public MCD-rPPG dataset in the layout evaluate_video.py reads.

MCD-rPPG (Egorov et al., ACM Multimedia 2025, CC BY 4.0) has 600 people filmed
for 3 minutes at rest and again after exercise, by three cameras, with a
finger PPG sensor recorded at the same time. This script takes the frontal
webcam (640x480, 30 fps) and the PPG signal already synchronised to its frames.

Subjects are drawn at random with a fixed seed, so the sample is not hand-picked
and anyone can reproduce it:

    python fetch_mcd_rppg.py --subjects 15 --out mcd_rppg      # ~2.4 GB
    python evaluate_video.py mcd_rppg --jobs 4

Output, one folder per recording (UBFC-rPPG layout):
    mcd_rppg/<patient>_<before|after>/vid.avi
    mcd_rppg/<patient>_<before|after>/ground_truth.txt   PPG per frame, HR from the dataset, time (s)

Dataset: https://huggingface.co/datasets/kyegorov/mcd_rppg
Paper:   https://arxiv.org/abs/2508.17924
"""

import argparse
import csv
import os
import random
import time
import urllib.error
import urllib.request

BASE = "https://huggingface.co/datasets/kyegorov/mcd_rppg/resolve/main/"
CAMERA = "FullHDwebcam"          # the frontal webcam
FPS = 29.9                       # frame rate stored in the dataset's video files


def download(path, dest, tries=6):
    """Fetch BASE + path to dest, waiting and retrying when the Hub rate-limits anonymous users."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    headers = {"User-Agent": "vitals-evaluation"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(BASE + path, headers=headers), timeout=120) as r:
                data = r.read()
            with open(dest + ".part", "wb") as f:
                f.write(data)
            os.replace(dest + ".part", dest)
            return
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"   rate-limited by the Hub, waiting {wait} s (set HF_TOKEN to avoid this)", flush=True)
            time.sleep(wait)


def write_ground_truth(ppg_sync_text, hr, path):
    """ppg_sync rows are 'PPG value, sync offset' for each video frame -> UBFC 3-line ground truth."""
    ppg = [float(line.split()[0]) for line in ppg_sync_text.splitlines() if line.strip()]
    times = [i / FPS for i in range(len(ppg))]
    with open(path, "w") as f:
        for row in (ppg, [hr] * len(ppg), times):
            f.write(" ".join(f"{x:.6g}" for x in row) + "\n")


def choose(rows, n, seed):
    """n patients at random (fixed seed) that have a frontal webcam recording before and after exercise."""
    by_patient = {}
    for r in rows:
        if r["camera"] == CAMERA:
            by_patient.setdefault(r["patient_id"], {})[r["step"]] = r
    complete = sorted(p for p, steps in by_patient.items() if {"before", "after"} <= steps.keys())
    picked = random.Random(seed).sample(complete, min(n, len(complete)))
    return [by_patient[p][s] for p in sorted(picked) for s in ("before", "after")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", type=int, default=15, help="people to fetch (2 recordings each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="mcd_rppg")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    db = os.path.join(args.out, "db.csv")
    download("db.csv", db)
    with open(db, newline="") as f:
        rows = list(csv.DictReader(f))
    chosen = choose(rows, args.subjects, args.seed)
    print(f"{len(chosen)} recordings from {len(chosen) // 2} people (seed {args.seed})")
    for r in chosen:
        folder = os.path.join(args.out, f"{r['patient_id']}_{r['step']}")
        os.makedirs(folder, exist_ok=True)
        print(f" {os.path.basename(folder)}", flush=True)
        download(r["video"], os.path.join(folder, "vid.avi"))
        sync = os.path.join(folder, "ppg_sync.txt")
        download(r["ppg_sync"], sync)
        with open(sync) as f:
            write_ground_truth(f.read(), float(r["pulse"] or "nan"), os.path.join(folder, "ground_truth.txt"))


if __name__ == "__main__":
    main()
