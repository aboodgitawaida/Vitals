"""live_session.LiveSession (the app logic the browser version runs) behaves like the app."""

import json

import numpy as np

from face_layout import PATCHES
from live_session import MIN_SEC, LiveSession
from session_benchmark import make_session

STATS = {"luminance": {"forehead": 150.0, "left cheek": 140.0, "right cheek": 145.0}, "clipped_fraction": 0.0}


def play(session, s, until=None):
    """Feed a simulated session frame by frame, the way the browser does."""
    infos = []
    for i, now in enumerate(s["t"]):
        if until is not None and now > until:
            break
        colors = {n: s["patches"][n][i] for n in PATCHES}
        session.frame(now, bool(s["face"][i]), colors, s["background"][i], s["pose"][i],
                      fast_motion=bool(s["moving"][i]), stats=STATS, fps=30.0)
        infos.append((now, session.info()))
    return infos


def test_finds_the_simulated_heart_rate():
    s = make_session("lighter", 0.4, np.random.default_rng(3), seconds=40, events=False)
    infos = play(LiveSession(), s)
    now, info = infos[-1]
    true = s["true_bpm"][-1]
    assert info["state"] == "locked"
    assert abs(info["bpm"] - true) < 5, (info["bpm"], true)
    assert info["level"] in ("good", "fair")
    assert info["agreement"] >= 2
    assert len(info["spectrum"][0]) == len(info["spectrum"][1]) > 10
    json.dumps(info, allow_nan=False)            # sent to the browser as JSON


def test_collecting_then_first_reading():
    s = make_session("lighter", 0.4, np.random.default_rng(4), seconds=12, events=False)
    infos = play(LiveSession(), s)
    early = [i for t, i in infos if t < MIN_SEC - 0.5]
    assert all(i["bpm"] is None and i["status"].startswith("Hold still") for i in early)
    assert any(i["bpm"] is not None for t, i in infos)


def test_long_face_loss_starts_over():
    s = make_session("lighter", 0.4, np.random.default_rng(5), seconds=20, events=False)
    session = LiveSession()
    play(session, s)
    t = s["t"][-1]
    for k in range(1, 45):                    # 1.5 s without a face
        session.frame(t + k / 30, False)
    info = session.info()
    assert info["status"] == "No face detected"
    assert info["bpm"] is None and info["collected_sec"] == 0.0


def test_motion_pauses_and_says_so():
    s = make_session("lighter", 0.4, np.random.default_rng(6), seconds=20, events=False)
    s["moving"][(s["t"] > 15) & (s["t"] < 15.5)] = True
    infos = play(LiveSession(), s, until=16.0)
    assert infos[-1][1]["status"] == "Movement detected - hold still"


def test_skipping_analysis_keeps_collecting():
    """analyze=False (worker busy) still buffers the frame, so nothing is lost."""
    s = make_session("lighter", 0.4, np.random.default_rng(7), seconds=8, events=False)
    session = LiveSession()
    for i, now in enumerate(s["t"]):
        colors = {n: s["patches"][n][i] for n in PATCHES}
        ran = session.frame(now, True, colors, s["background"][i], s["pose"][i], stats=STATS,
                            fps=30.0, analyze=False)
        assert not ran
    assert session.info()["collected_sec"] > 7
    assert session.analysis is None


def test_method_and_background_switches():
    session = LiveSession()
    session.set_method("green")
    session.set_background(False)
    info = session.info()
    assert info["method"] == "green" and info["use_background"] is False
    try:
        session.set_method("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("bad method accepted")
