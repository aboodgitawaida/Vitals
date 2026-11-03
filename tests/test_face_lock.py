"""frame_features.FaceLock: with two people in view, measure one and keep following them."""

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

import face_layout
from face_regions import background_color
from frame_features import FaceLock
from live_session import FACE_GAP_SEC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 1280, 480


def face(cx, cy, width=160.0, seed=0):
    """A face-shaped cloud of landmarks, `width` pixels across, centred at (cx, cy)."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.5, 0.5, (478, 2)) * [width, width * 1.3]
    pts[0], pts[1] = (-width / 2, -width * 0.65), (width / 2, width * 0.65)     # exact extent
    return (pts + [cx, cy]).astype(np.float32)


def sequence():
    """Two people; the one on the right is bigger; they swap order in the list; the right one leaves and comes back;
    a tap moves to the left one."""
    frames = []
    for k in range(30):                                    # right person bigger and nearer the middle
        a, b = face(380 + k, 240, 150, 1), face(800 - k, 240, 190, 2)
        frames.append({"faces": [a, b] if k % 2 else [b, a], "t": k / 30})
    for k in range(30, 60):                                # right person leaves (1 s): no measurement
        frames.append({"faces": [face(410, 240, 150, 1)], "t": k / 30})
    for k in range(60, 90):                                # and comes back within 3 s: same person again
        frames.append({"faces": [face(410, 240, 150, 1), face(770, 240, 190, 2)], "t": k / 30})
    frames.append({"faces": [face(410, 240, 150, 1), face(770, 240, 190, 2)], "t": 3.0, "tap": (400, 250)})
    for k in range(91, 100):
        frames.append({"faces": [face(770, 240, 190, 2), face(410, 240, 150, 1)], "t": k / 30})
    for k in range(100, 110):                              # the measured (left) person leaves...
        frames.append({"faces": [face(770, 240, 190, 2)], "t": k / 30})
    frames.append({"faces": [face(770, 240, 190, 2)], "t": 110 / 30, "tap": (770, 240)})   # ...the other is tapped
    return frames


def run_py(frames):
    lock, out = FaceLock(), []
    for f in frames:
        if f.get("tap"):
            lock.request(*f["tap"])
        i = lock.pick(f["faces"], f["t"], W, H)
        out.append({"index": i, "switched": lock.switched, "count": lock.count})
    return out


def centre_x(frames, out, k):
    return None if out[k]["index"] is None else float(frames[k]["faces"][out[k]["index"]][:, 0].mean())


def test_follows_one_person_and_switches_on_tap():
    frames = sequence()
    out = run_py(frames)
    xs = [centre_x(frames, out, k) for k in range(len(frames))]
    assert all(x > 600 for x in xs[:30])                    # the bigger, more central face, whatever the order
    assert all(x is None for x in xs[30:60])                # gone: not swapped for the other person
    assert all(x > 600 for x in xs[60:90])                  # back: the same person
    assert not any(o["switched"] for o in out[:90])
    assert out[90]["switched"] and xs[90] < 600             # tap on the left face
    assert all(x < 600 for x in xs[91:100]) and not any(o["switched"] for o in out[91:100])
    assert all(x is None for x in xs[100:110])              # the measured person left: nothing measured
    assert out[110]["switched"] and xs[110] > 600           # tapping the one still there starts over
    assert out[0]["count"] == 2 and out[45]["count"] == 1


def test_picks_again_after_the_person_is_gone_long_enough():
    lock = FaceLock()
    assert lock.pick([face(800, 240, 190)], 0.0, W, H) == 0
    assert lock.pick([face(300, 240, 150)], 1.0, W, H) is None
    assert lock.pick([face(300, 240, 150)], 1.0 + face_layout.FACE_LOCK_SEC + 0.1, W, H) == 0
    assert lock.switched


def test_tap_while_the_measured_person_is_away_starts_over():
    """You step out; your dad, sitting about where you were, is tapped: his data must not join yours."""
    lock = FaceLock()
    assert lock.pick([face(640, 240, 160), face(1000, 240, 160)], 0.0, W, H) == 0
    assert lock.pick([face(1000, 240, 160)], 0.5, W, H) is None          # you left
    lock.request(1000, 240)
    assert lock.pick([face(1000, 240, 160)], 1.0, W, H) == 0 and lock.switched
    lock.request(1000, 240)                                                # tapping the measured face again
    assert lock.pick([face(1000, 240, 160)], 1.1, W, H) == 0 and not lock.switched


def test_lock_time_matches_the_session_start_over():
    assert face_layout.FACE_LOCK_SEC == FACE_GAP_SEC


def test_wall_boxes_avoid_the_other_person():
    frame = np.full((H, W, 3), 128, np.uint8)
    me = face(640, 300, 160)
    rgb, boxes = background_color(frame, me)
    assert len(boxes) == 2
    dad = face(160, 160, 160)                               # sits where the left wall box is
    rgb, boxes = background_color(frame, me, [dad])
    assert len(boxes) == 1 and boxes[0][0] > W / 2


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_browser_face_lock_matches_python():
    frames = sequence()
    payload = {"cfg": face_layout.layout_config(), "frames": [
        {"faces": [f.tolist() for f in fr["faces"]], "t": fr["t"], "w": W, "h": H, "tap": fr.get("tap")}
        for fr in frames]}
    res = subprocess.run(["node", os.path.join(ROOT, "tests", "run_face_lock.mjs")], input=json.dumps(payload),
                         capture_output=True, text=True, check=True)
    assert json.loads(res.stdout) == run_py(frames)
