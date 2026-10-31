"""
evaluate_video.py -- measure accuracy on videos with a known heart rate.

This is how camera-based heart rate is validated in research: run the method
on videos recorded together with a contact sensor (pulse oximeter / ECG) and
compare. The same per-frame code as the live app is used (frame_features.py),
so this measures the real pipeline, not a separate "evaluation version".

Inputs (any mix):
  * a folder in UBFC-rPPG layout: vid.avi (or .mp4) + ground_truth.txt
    (3 lines: pulse-oximeter waveform, heart rate, time in seconds).
    UBFC-rPPG ("DATASET_2", Bobbia et al. 2019) is a free research dataset
    of 42 people; request access from its authors.
  * a dataset folder containing many such subject folders
  * a folder made by make_test_video.py (same layout, known heart rate)
  * a plain video file plus --ref-bpm 72 (e.g. your own recording + a watch)

    python evaluate_video.py path/to/UBFC/DATASET_2
    python evaluate_video.py test_videos/selfie_72
    python evaluate_video.py my_video.mp4 --ref-bpm 68
    python evaluate_video.py mcd_rppg --jobs 4       # MCD-rPPG sample (fetch_mcd_rppg.py), 4 videos at a time

Reference heart rate per window: the spectral peak of the pulse-oximeter
waveform over the SAME 15 s window (standard practice; the oximeter's own HR
column is averaged and lagged differently).

Reports:
  1. per-window error of each method on every window (comparable to papers):
     MAE, RMSE, Pearson r, share within 5 BPM
  2. what the app would have shown (tracker on): coverage + the same errors
  3. Bland-Altman plot (bias and 95% limits of agreement) + a markdown table
Feature extraction is cached (eval_cache/), so re-runs are fast.
Privacy: only per-frame average colours are cached, never images.
"""

import argparse
import glob
import hashlib
import os

import cv2
import numpy as np

from face_regions import TRACES
from frame_features import FeatureExtractor
from vitals_dsp import HR_HIGH, HR_LOW, analyze_heart_rate, bandpass, pick_peak, power_spectrum, resample_uniform
from vitals_stream import HeartRateTracker, simulate_display

WINDOW_SEC, STEP_SEC = 15.0, 1.0
CACHE_VERSION = 2          # bump when the per-frame features change (2: whole-face colour added)
VIDEO_EXT = (".avi", ".mp4", ".mov", ".mkv", ".m4v", ".webm")


# --------------------------------------------------------------------------
# Finding videos and reading ground truth
# --------------------------------------------------------------------------

def find_subjects(paths):
    """Expand inputs into (name, video path, ground-truth path or None)."""
    out = []
    for p in paths:
        if os.path.isfile(p):
            out.append((os.path.splitext(os.path.basename(p))[0], p, None))
            continue
        videos = [f for f in sorted(glob.glob(os.path.join(p, "*"))) if f.lower().endswith(VIDEO_EXT)]
        gt = os.path.join(p, "ground_truth.txt")
        if videos:
            out.append((os.path.basename(os.path.normpath(p)), videos[0], gt if os.path.exists(gt) else None))
        else:                                       # a dataset folder: one level down
            for sub in sorted(glob.glob(os.path.join(p, "*"))):
                if os.path.isdir(sub):
                    out += find_subjects([sub])
    return out


def read_ground_truth(path):
    """UBFC-rPPG DATASET_2 ground_truth.txt -> (times s, ppg waveform, hr)."""
    with open(path) as f:
        lines = [list(map(float, line.split())) for line in f if line.strip()]
    ppg, hr, times = (np.array(x) for x in lines[:3])
    if times[-1] > 1000:                            # some copies store milliseconds
        times = times / 1000.0
    return times - times[0], ppg, hr


def reference_bpm(times, ppg, t_end, window=WINDOW_SEC):
    """Heart rate from the contact sensor over [t_end - window, t_end]: its spectral peak."""
    sel = (times >= t_end - window) & (times <= t_end)
    if sel.sum() < 30:
        return None
    _, u = resample_uniform(times[sel], ppg[sel])
    f, p = power_spectrum(bandpass(u - u.mean(), HR_LOW, HR_HIGH))
    return 60.0 * pick_peak(f, p, HR_LOW, HR_HIGH)


# --------------------------------------------------------------------------
# Video -> per-frame features (same code as the live app)
# --------------------------------------------------------------------------

def make_face_mesh():
    import mediapipe as mp                          # imported here so tests can run without it
    return mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False,
                                           min_detection_confidence=0.5, min_tracking_confidence=0.5)


def extract_features(video_path, face_mesh=None, cache_dir="eval_cache", process_width=640):
    """
    Returns dict: t (N,), patches {trace name: (N,3)} (the 16 patches and the whole face),
    background (N,3), face (N,), moving (N,), pose (N,5).
    Times come from the video's frame rate, not the wall clock.
    """
    cache = None
    if cache_dir:
        key = hashlib.sha1(f"{os.path.abspath(video_path)}:{os.path.getsize(video_path)}:{CACHE_VERSION}"
                           .encode()).hexdigest()[:16]
        cache = os.path.join(cache_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_{key}.npz")
    if cache and os.path.exists(cache):
        d = np.load(cache)
        return {"t": d["t"], "patches": {n: d[f"p_{n}"] for n in TRACES}, "background": d["background"],
                "face": d["face"], "moving": d["moving"], "pose": d["pose"]}

    face_mesh = face_mesh or make_face_mesh()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fx = FeatureExtractor()
    t, face, moving, bg, pose = [], [], [], [], []
    patches = {n: [] for n in TRACES}
    missing = np.full(3, np.nan)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        scale = process_width / frame.shape[1]
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        res = face_mesh.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        ff = fx.process(frame, res.multi_face_landmarks[0] if res.multi_face_landmarks else None)
        t.append(i / fps)
        face.append(ff.face)
        moving.append(ff.fast_motion)
        bg.append(missing if ff.bg_rgb is None else ff.bg_rgb)
        pose.append(np.full(5, np.nan) if ff.pose is None else ff.pose)
        for n in TRACES:
            c = ff.colors.get(n) if ff.face else None
            patches[n].append(missing if c is None else c)
        i += 1
    cap.release()
    out = {"t": np.array(t), "patches": {n: np.array(v) for n, v in patches.items()},
           "background": np.array(bg), "face": np.array(face), "moving": np.array(moving), "pose": np.array(pose)}
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache, t=out["t"], background=out["background"], face=out["face"],
                            moving=out["moving"], pose=out["pose"], **{f"p_{n}": v for n, v in out["patches"].items()})
    return out


# --------------------------------------------------------------------------
# Scoring one subject
# --------------------------------------------------------------------------

WINDOW_METHODS = {          # name: analyze_heart_rate settings
    "green": dict(method="green"),
    "POS": dict(method="pos"),
    "CHROM": dict(method="chrom"),
    "full pipeline (auto)": dict(method="auto"),
}


def evaluate_subject(feats, ref_fn):
    """
    ref_fn(t_end) -> reference BPM for the window ending at t_end, or None.
    Returns {"windows": {method: [(t, est, ref)]}, "screen": [(t, shown or None, ref)]}.
    """
    t = feats["t"]
    windows = {m: [] for m in WINDOW_METHODS}
    end = t[0] + WINDOW_SEC
    while end <= t[-1]:
        sel = (t > end - WINDOW_SEC) & (t <= end) & feats["face"]
        ref = ref_fn(end)
        if ref is not None and sel.sum() > WINDOW_SEC * 10:
            for name, kw in WINDOW_METHODS.items():
                r = analyze_heart_rate(t[sel], {n: v[sel] for n, v in feats["patches"].items()},
                                       feats["background"][sel], pose=feats["pose"][sel],
                                       patch_map=False, **kw)
                windows[name].append((end, None if r is None else r["bpm"], ref))
        end += STEP_SEC

    shown = simulate_display(t, feats["patches"], feats["background"], feats["face"], feats["moving"],
                             HeartRateTracker(), pose=feats["pose"])
    screen = [(now, b, ref_fn(now)) for now, b, _ in shown if now - t[0] >= 8.0]
    screen = [s for s in screen if s[2] is not None]
    return {"windows": windows, "screen": screen}


def score_video(job):
    """(name, video, ground truth or None, constant reference or None) -> (name, evaluate_subject result).
    A top-level function so --jobs can run it in worker processes."""
    name, video, gt, ref_bpm = job
    if gt:
        gt_t, gt_ppg, _ = read_ground_truth(gt)
        ref_fn = lambda te: reference_bpm(gt_t, gt_ppg, te)
    else:
        ref_fn = lambda te: ref_bpm
    return name, evaluate_subject(extract_features(video), ref_fn)


def score_all(jobs, workers):
    """Yield (name, result) in input order, using several processes if workers > 1 (FaceMesh is slow)."""
    if workers <= 1:
        for job in jobs:
            print(f"{job[0]}: analysing {job[1]} ...", flush=True)
            yield score_video(job)
        return
    from multiprocessing import get_context
    with get_context("spawn").Pool(workers) as pool:
        yield from pool.imap(score_video, jobs)


def metrics(pairs):
    """pairs: [(estimate or None, reference)] -> dict of standard rPPG metrics."""
    pairs = [(e, r) for e, r in pairs if e is not None]
    if len(pairs) < 2:
        return {"n": len(pairs), "MAE": np.nan, "RMSE": np.nan, "r": np.nan, "<=5": np.nan}
    e, r = np.array(pairs).T
    err = e - r
    pearson = np.corrcoef(e, r)[0, 1] if e.std() > 0 and r.std() > 0 else np.nan
    return {"n": len(e), "MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()), "r": pearson,
            "<=5": np.mean(np.abs(err) <= 5), "bias": err.mean(), "sd": err.std()}


def bland_altman(pairs, path, title):
    """Difference vs mean of (estimate, reference), with bias and 95% limits of agreement."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e, r = np.array([(a, b) for a, b in pairs if a is not None]).T
    mean, diff = (e + r) / 2, e - r
    bias, sd = diff.mean(), diff.std()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(mean, diff, s=8, alpha=0.5)
    for y, ls, lab in ((bias, "-", f"bias {bias:+.1f}"), (bias + 1.96 * sd, "--", f"+1.96 SD {bias + 1.96 * sd:+.1f}"),
                       (bias - 1.96 * sd, "--", f"-1.96 SD {bias - 1.96 * sd:+.1f}")):
        ax.axhline(y, color="k", linestyle=ls, linewidth=1, label=lab)
    ax.set_xlabel("mean of camera and reference (BPM)")
    ax.set_ylabel("camera - reference (BPM)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="videos, subject folders or a dataset folder")
    ap.add_argument("--ref-bpm", type=float, help="constant reference heart rate for plain videos")
    ap.add_argument("--out", default="eval_results", help="folder for the plot and tables")
    ap.add_argument("--jobs", type=int, default=1, help="videos to analyse at the same time (processes)")
    args = ap.parse_args()

    subjects = find_subjects(args.paths)
    if not subjects:
        raise SystemExit("No videos found.")
    jobs = []
    for name, video, gt in subjects:
        if gt or args.ref_bpm:
            jobs.append((name, video, gt, args.ref_bpm))
        else:
            print(f"skip {name}: no ground_truth.txt and no --ref-bpm")
    os.makedirs(args.out, exist_ok=True)
    all_windows = {m: [] for m in WINDOW_METHODS}
    all_screen, rows, groups = [], [], {}
    for name, res in score_all(jobs, args.jobs):
        m_all = metrics([(e, r) for _, e, r in res["windows"]["full pipeline (auto)"]])
        m_scr = metrics([(e, r) for _, e, r in res["screen"]])
        cov = np.mean([e is not None for _, e, _ in res["screen"]]) if res["screen"] else np.nan
        print(f"{name}: windows MAE {m_all['MAE']:.1f} BPM | on screen: coverage {cov:.0%}, "
              f"MAE {m_scr['MAE']:.1f} BPM", flush=True)
        rows.append((name, m_all, m_scr, cov))
        for m in WINDOW_METHODS:
            all_windows[m] += [(e, r) for _, e, r in res["windows"][m]]
        all_screen += [(e, r) for _, e, r in res["screen"]]
        if "_" in name:                             # e.g. 1020_before / 1020_after (MCD-rPPG)
            groups.setdefault(name.rsplit("_", 1)[1], []).extend((e, r) for _, e, r in res["screen"])

    if not rows:
        return
    print(f"\n{len(rows)} video(s). 1) Every 15 s window, every method (comparable to published results):")
    print(f"{'method':<24}{'windows':>8}{'MAE':>7}{'RMSE':>7}{'r':>7}{'<=5 BPM':>9}")
    md = ["| Method | Windows | MAE (BPM) | RMSE (BPM) | Pearson r | Within 5 BPM |", "| --- | --- | --- | --- | --- | --- |"]
    for m in WINDOW_METHODS:
        x = metrics(all_windows[m])
        print(f"{m:<24}{x['n']:>8}{x['MAE']:>7.2f}{x['RMSE']:>7.2f}{x['r']:>7.2f}{x['<=5']:>9.0%}")
        md.append(f"| {m} | {x['n']} | {x['MAE']:.2f} | {x['RMSE']:.2f} | {x['r']:.2f} | {x['<=5']:.0%} |")
    x = metrics(all_screen)
    cov = np.mean([e is not None for e, _ in all_screen])
    print(f"\n2) What the app would have shown: coverage {cov:.0%}, MAE {x['MAE']:.2f}, RMSE {x['RMSE']:.2f}, "
          f"r {x['r']:.2f}, within 5 BPM {x['<=5']:.0%}")
    md.append(f"| app display (tracker on) | coverage {cov:.0%} | {x['MAE']:.2f} | {x['RMSE']:.2f} | {x['r']:.2f} | {x['<=5']:.0%} |")
    if len(groups) > 1:                             # e.g. at rest vs after exercise
        print("\n3) What the app showed, by group (folder name after the last _):")
        md += ["", "| Group | Coverage | Mean reference (BPM) | MAE (BPM) | RMSE (BPM) | Within 5 BPM |",
               "| --- | --- | --- | --- | --- | --- |"]
        for g, pairs in sorted(groups.items()):
            x = metrics(pairs)
            gcov = np.mean([e is not None for e, _ in pairs])
            ref = np.mean([r for _, r in pairs])
            print(f"   {g:<10} coverage {gcov:.0%}, mean reference {ref:.0f} BPM, MAE {x['MAE']:.2f}, "
                  f"RMSE {x['RMSE']:.2f}, within 5 BPM {x['<=5']:.0%}")
            md.append(f"| {g} | {gcov:.0%} | {ref:.0f} | {x['MAE']:.2f} | {x['RMSE']:.2f} | {x['<=5']:.0%} |")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(args.out, "per_video.csv"), "w") as f:
        f.write("video,window_MAE,window_RMSE,screen_coverage,screen_MAE\n")
        for name, a, s, c in rows:
            f.write(f"{name},{a['MAE']:.3f},{a['RMSE']:.3f},{c:.3f},{s['MAE']:.3f}\n")
    if len([e for e, _ in all_screen if e is not None]) >= 2:
        bland_altman(all_screen, os.path.join(args.out, "bland_altman.png"),
                     "Camera vs reference heart rate (what the app shows)")
    print(f"\nSaved {args.out}/summary.md, per_video.csv, bland_altman.png")


if __name__ == "__main__":
    main()
