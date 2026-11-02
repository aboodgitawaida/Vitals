"""
tune_tracker.py -- choose the tracker's confidence thresholds for the network, on the validation split.

Runs the app's replay once per recording with the network, from MIN_SEC_FLOOR
seconds of data on, and keeps every update the tracker would see. Tracker
settings are then swept over those saved updates in seconds (the network does
not use the tracker's hint, so its outputs do not depend on the settings).

    python tune_tracker.py --data mcd_features                        # validation split
    python tune_tracker.py --data mcd_features --weights other.npz

Prints, for each setting: share of time a steady number is shown and how often
it is within 5 BPM, the same for settling numbers, and time to the first number.
"""

import argparse
import itertools
import json
import os
import pickle

import numpy as np

from build_mcd_dataset import load_features
from compare_net import screen_stats
from evaluate_video import read_ground_truth, reference_bpm
from vitals_stream import HeartRateTracker, Reading, simulate_display

MIN_SEC_FLOOR = 6.0


class Recorder:
    """Stands in for the tracker during the replay and keeps what it is given."""

    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.append(("reset",))

    def update(self, now, bpm=None, snr_db=None, usable=True, window_sec=None, agreement=None, confidence=None):
        self.calls.append(("update", now, bpm, snr_db, usable, window_sec, agreement, confidence))
        return Reading(None, "acquiring", snr_db, None)


def record(job):
    folder, weights = job
    if weights:
        import vitals_net
        vitals_net._net = vitals_net.PulseNet(weights)
    f = load_features(folder)
    t_gt, ppg, _ = read_ground_truth(os.path.join(folder, "ground_truth.txt"))
    rec = Recorder()
    shown = simulate_display(f["t"], f["patches"], f["background"], f["face"], f["moving"], rec,
                             pose=f["pose"], method="net", min_sec=MIN_SEC_FLOOR)
    refs = [reference_bpm(t_gt, ppg, max(now, 10.0), window=min(15.0, max(now, 10.0))) for now, _, _ in shown]
    return {"name": os.path.basename(folder), "calls": rec.calls, "refs": refs}


def replay(rec, min_sec, **kw):
    """One recording's saved updates through a tracker with these settings -> compare_net screen rows."""
    tr = HeartRateTracker(**kw)
    rows, i = [], 0
    for c in rec["calls"]:
        if c[0] == "reset":
            tr.reset()
            continue
        _, now, bpm, snr, usable, span, agreement, conf = c
        r = tr.update(now) if span is None or span < min_sec else tr.update(now, bpm, snr, usable=usable, window_sec=span,
                                                             agreement=agreement, confidence=conf)
        ref = rec["refs"][i]
        i += 1
        if ref is not None:
            rows.append((now, r.bpm, ref, bool(r.bpm is not None and tr.settling(now))))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="mcd_features")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--weights", default=None, help="network weights (default: vitals_net_weights.npz)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--cache", default=None, help="save / reuse the recorded updates here (.pkl)")
    args = ap.parse_args()
    if args.cache and os.path.exists(args.cache):
        with open(args.cache, "rb") as f:
            recs = pickle.load(f)
    else:
        with open(os.path.join(args.data, "splits.json")) as f:
            people = json.load(f)[args.split]
        folders = [os.path.join(args.data, f"{p}_{s}") for p in people for s in ("before", "after")
                   if os.path.exists(os.path.join(args.data, f"{p}_{s}", "features.npz"))]
        from multiprocessing import get_context
        with get_context("spawn").Pool(args.jobs) as pool:
            recs = pool.map(record, [(f, args.weights) for f in folders])
        if args.cache:
            with open(args.cache, "wb") as f:
                pickle.dump(recs, f)

    print("| min s | enter | keep | confirm | steady shown | steady within 5 | steady MAE | settling shown | "
          "settling within 5 | first (median) | first by 15 s | never |")
    print("| " + " | ".join(["---"] * 12) + " |")
    grid = itertools.product([6.0, 7.0, 8.0], [0.3, 0.4, 0.5, 0.6], [0.2, 0.3], [0.6, 0.7, 0.8])
    for min_sec, enter, keep, confirm in grid:
        if keep >= enter:
            continue
        kw = {"CONF_ENTER": enter, "CONF_KEEP": keep, "CONF_CONFIRM": confirm}
        s = screen_stats([{"screen": {"x": replay(r, min_sec, **kw)}} for r in recs], "x")
        print(f"| {min_sec:.0f} | {enter} | {keep} | {confirm} | {s['steady_share']:.0%} | {s['steady_within5']:.0%} | "
              f"{s['steady_MAE']:.2f} | {s['settling_share']:.0%} | {s['settling_within5']:.0%} | "
              f"{s['first_median']:.1f} s | {s['first_by15']:.0%} | {s['never']:.0%} |", flush=True)
    s = screen_stats([{"screen": {"x": replay(r, 8.0, CONF_ENTER=np.inf)}} for r in recs], "x")
    print(f"(sanity: a tracker that never accepts shows nothing: never = {s['never']:.0%})")


if __name__ == "__main__":
    main()
