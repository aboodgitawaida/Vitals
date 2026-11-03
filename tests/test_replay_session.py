"""replay_session.py reads a saved web session and reproduces what the browser showed."""

import gzip
import json

import numpy as np

import replay_session
from face_layout import PATCHES
from live_session import LiveSession
from session_benchmark import make_session

STATS = {"luminance": {"forehead": 150.0, "left cheek": 140.0, "right cheek": 145.0}, "clipped_fraction": 0.0,
         "framing": {"outside": 0.0, "size": 0.15, "roll": 0.0}}


def browser_session(seconds=20, seed=0):
    """
    A session file as web/app.js writes it, with the screen produced the way web/worker.js does:
    frames in uneven batches (analysis on the last frame of each), a method switch, and a
    3.5 s face loss that makes the session start over.
    """
    rng = np.random.default_rng(seed)
    s = make_session("lighter", 0.4, rng, seconds=seconds, events=False)
    frames = []
    for i, now in enumerate(s["t"]):
        lost = 12.0 < now < 15.5
        frames.append({"t": float(now), "face": not lost,
                       "colors": None if lost else {n: s["patches"][n][i].tolist() for n in PATCHES},
                       "bg": None if lost else s["background"][i].tolist(),
                       "pose": None if lost else s["pose"][i].tolist(),
                       "fast": bool(s["moving"][i]), "small": False, "stats": None if lost else STATS, "fps": 30.0})
    sizes, left = [], len(frames)
    while left:
        sizes.append(min(left, int(rng.integers(1, 25))))
        left -= sizes[-1]
    events = [{"before_batch": 5, "t": 1.0, "action": "method", "value": "classic", "why": None}]
    worker, screen, pos = LiveSession(), [], 0
    for b, n in enumerate(sizes):
        for ev in events:
            if ev["before_batch"] == b:
                worker.set_method(ev["value"])
        for k, f in enumerate(frames[pos:pos + n]):
            worker.frame(f["t"], f["face"], f["colors"], f["bg"], f["pose"], f["fast"], f["small"], f["stats"],
                         f["fps"], analyze=k == n - 1)
        pos += n
        info = worker.info()
        screen.append({"batch": b, "bpm": info["bpm"], "state": info["state"]})
    return {"format": "vitals-session", "format_version": 1, "frames": frames, "batches": sizes,
            "events": events, "screen": screen, "reference_bpm": float(s["true_bpm"][-1])}


def test_saved_session_round_trip_reproduces_the_screen(tmp_path):
    doc = browser_session()
    path = tmp_path / "vitals_session.json.gz"
    path.write_bytes(gzip.compress(json.dumps(doc).encode()))
    loaded = replay_session.load(str(path))
    records, screens, resets = replay_session.replay(loaded)
    match, n = replay_session.screen_match(screens, loaded["screen"])
    assert n == len(doc["batches"]) and match == 1.0
    assert len(records) == len(doc["frames"])
    assert [why for _, why in resets] == ["face lost for 3.0 s"]
    text = "\n".join(replay_session.report(loaded, records, screens, resets, loaded["reference_bpm"]))
    assert "Started over 1 time(s):" in text and "face lost" in text


def test_windows_compare_both_pipelines():
    doc = browser_session(seconds=32, seed=1)
    rows = replay_session.windows(doc, None)
    done = [r for r in rows if "network" in r]
    assert done and all(r["hand-made"] is not None for r in done)
    true = doc["reference_bpm"]
    assert np.median([abs(r["hand-made"] - true) for r in done]) < 5
