"""The rolling buffer and the display tracker follow their documented rules."""

import numpy as np

from vitals_stream import HeartRateTracker, LegacyDisplay, SampleBuffer, simulate_display

NAMES = ["a", "b"]


def frame(v=100.0):
    return {n: np.array([v, v, v]) for n in NAMES}, np.array([50.0, 50.0, 50.0])


# ---------------- SampleBuffer ----------------

def test_buffer_keeps_only_the_window():
    buf = SampleBuffer(NAMES, window_sec=2.0)
    for i in range(100):
        buf.add(i / 30, *frame())
    assert buf.span() <= 2.0 + 1e-9
    t, traces, bg = buf.arrays()
    assert len(t) == len(traces["a"]) == len(bg)


def test_buffer_bridges_short_face_loss():
    buf = SampleBuffer(NAMES, max_gap_sec=1.0)
    for i in range(30):
        buf.add(i / 30, *frame())
    assert buf.add_missing(1.2)                    # 0.23 s after the last face: keep going
    _, traces, _ = buf.arrays()
    assert np.isnan(traces["a"][-1]).all()
    assert buf.span() > 1.0


def test_buffer_resets_after_long_face_loss():
    buf = SampleBuffer(NAMES, max_gap_sec=1.0)
    buf.add(0.0, *frame())
    t = 0.0
    while buf.add_missing(t := t + 1 / 30):
        pass
    assert t > 1.0 and len(buf.times) == 0


def test_missing_frames_do_not_count_as_seeing_a_face():
    buf = SampleBuffer(NAMES, max_gap_sec=1.0)
    buf.add(0.0, *frame())
    for k in range(1, 30):                         # 0.97 s of missing frames
        assert buf.add_missing(k / 30)
    assert not buf.add_missing(1.1)                # still measured from t=0, not from the last NaN


def test_unusable_frames_are_gaps_but_keep_the_face():
    buf = SampleBuffer(NAMES, max_gap_sec=1.0)
    buf.add(0.0, *frame())
    buf.add_unusable(0.9)
    assert buf.add_missing(1.5)                    # face was seen at 0.9 s
    _, traces, _ = buf.arrays()
    assert np.isnan(traces["a"][1]).all()


# ---------------- HeartRateTracker ----------------

def feed(trk, values, start=0.0, step=0.5, usable=True):
    """values: list of (bpm, snr). Returns the readings."""
    out = []
    for k, (bpm, snr) in enumerate(values):
        out.append(trk.update(start + k * step, bpm, snr, usable))
    return out


def test_first_reading_needs_confirmation():
    trk = HeartRateTracker()
    r = feed(trk, [(70, 5)] * 5)
    shown = [x.bpm is not None for x in r]
    assert shown == [False, False, False, True, True]   # appears once it has persisted 1.5 s
    assert r[-1].state == "locked"


def test_one_strong_noise_window_never_shows():
    trk = HeartRateTracker()
    r = feed(trk, [(70, 5), (120, 5), (95, 5), (60, 5), (140, 5)])
    assert all(x.bpm is None for x in r)


def test_weak_window_near_track_is_kept_far_one_is_not():
    trk = HeartRateTracker()
    feed(trk, [(70, 5)] * 4)
    near = trk.update(2.0, 72, -1.0)
    assert near.state == "locked"
    far = trk.update(2.5, 100, -1.0)
    assert far.state == "holding" and far.bpm is not None and abs(far.bpm - 70) < 3


def test_weak_windows_alone_cannot_keep_a_reading_alive():
    trk = HeartRateTracker()
    feed(trk, [(70, 5)] * 4)                          # last strong window at t = 1.5
    r = feed(trk, [(70, -1.0)] * 30, start=2.0)       # weak but agreeing, forever
    last_shown = max(x for x, rd in zip(np.arange(2.0, 17.0, 0.5), r) if rd.bpm is not None)
    assert last_shown <= 1.5 + trk.REFRESH_SEC + trk.HOLD_SEC + 1e-9
    assert r[-1].state == "lost"


def test_hold_then_lost():
    trk = HeartRateTracker()
    feed(trk, [(70, 5)] * 4)
    r = feed(trk, [(None, None)] * 10, start=2.0)
    states = [x.state for x in r]
    assert states[0] == "holding" and states[-1] == "lost"
    assert all(x.bpm is not None for x in r if x.state == "holding")
    assert all(x.bpm is None for x in r if x.state == "lost")


def test_motion_windows_are_not_trusted():
    trk = HeartRateTracker()
    feed(trk, [(70, 5)] * 4)
    r = trk.update(2.0, 110, 10.0, usable=False)
    assert r.state == "holding" and abs(r.bpm - 70) < 1


def test_real_jump_is_followed_after_confirmation():
    trk = HeartRateTracker()
    feed(trk, [(70, 5)] * 6)
    r = feed(trk, [(95, 5)] * 8, start=3.0)
    # held at ~70 while the jump is being confirmed, then switches
    assert abs(r[0].bpm - 70) < 1
    assert abs(r[-1].bpm - 95) < 1
    switch_time = 3.0 + 0.5 * next(i for i, x in enumerate(r) if abs(x.bpm - 95) < 1)
    assert switch_time - 3.0 >= trk.CONFIRM_SEC


def test_display_is_median_smoothed():
    trk = HeartRateTracker()
    r = feed(trk, [(70, 5), (70, 5), (70, 5), (70, 5), (78, 5), (70, 5)])
    assert r[4].bpm == 70


def test_legacy_blanks_on_any_weak_window():
    leg = LegacyDisplay()
    assert leg.update(0, 70, 5).bpm == 70
    assert leg.update(0.5, 70, -0.1).bpm is None


# ---------------- end to end ----------------

def test_simulated_session_locks_near_truth():
    from session_benchmark import make_session
    s = make_session("lighter", 0.6, np.random.default_rng(11), seconds=40)
    t, true_bpm = s["t"], s["true_bpm"]
    res = simulate_display(t, s["patches"], s["background"], s["face"], s["moving"], HeartRateTracker(),
                           pose=s["pose"])
    shown = [(now, b) for now, b, s in res if b is not None]
    assert len(shown) > 0.4 * len(res)
    err = [abs(b - np.interp(now, t, true_bpm)) for now, b in shown]
    assert np.median(err) < 3.0


def test_short_windows_need_a_higher_snr():
    trk = HeartRateTracker()
    assert trk.extra_db(5.0) == trk.SHORT_WINDOW_EXTRA_DB
    assert trk.extra_db(15.0) == 0.0 and trk.extra_db(None) == 0.0
    # 0.5 dB is enough for a full window but not for a 5 s one
    full = feed(HeartRateTracker(), [(70, 0.5)] * 5)
    t5 = HeartRateTracker()
    short = [t5.update(k * 0.5, 70, 0.5, window_sec=5.0) for k in range(5)]
    assert full[-1].bpm is not None and short[-1].bpm is None


def test_buffer_stores_pose():
    buf = SampleBuffer(NAMES)
    assert buf.pose_array() is None
    buf.add(0.0, *frame(), pose=[0.5, 0.5, 0.3, 0.0])
    buf.add(0.1, *frame())
    p = buf.pose_array()
    assert p.shape == (2, 4) and np.isnan(p[1]).all()


def test_unusable_and_disagreeing_windows_are_ignored():
    """Motion (usable=False) and a lone method (agreement 1/3) must not start or feed a reading."""
    for kwargs in (dict(usable=False), dict(agreement=1)):
        tr = HeartRateTracker()
        for k in range(20):
            reading = tr.update(10.0 + 0.5 * k, 72.0, 8.0, window_sec=15.0, **kwargs)
        assert reading.bpm is None, kwargs
    tr = HeartRateTracker()
    for k in range(20):
        reading = tr.update(10.0 + 0.5 * k, 72.0, 8.0, window_sec=15.0, agreement=2)
    assert reading.bpm == 72.0
