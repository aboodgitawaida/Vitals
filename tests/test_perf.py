"""Background analysis gives the same answers as inline, and stale results are thrown away."""

import time

import numpy as np

from perf import BackgroundAnalyzer, StageTimer
from synthetic_benchmark import make_trial


def _job():
    t, patches, bg, truth = make_trial("steady light", "lighter", np.random.default_rng(1))
    return dict(times=t, patch_traces=patches, background=bg, patch_map=False), truth


def _wait(an, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        done = an.poll()
        if done is not None:
            return done
        time.sleep(0.01)
    raise TimeoutError


def test_stage_timer_smooths():
    timer = StageTimer(smoothing=0.5)
    timer.add("x", 10)
    timer.add("x", 20)
    assert timer.get("x") == 15 and timer.get("missing") is None


def test_background_matches_inline():
    kwargs, truth = _job()
    inline = BackgroundAnalyzer(background=False)
    assert inline.submit("meta", **kwargs)
    meta, a, _ = _wait(inline)
    bg = BackgroundAnalyzer(background=True)
    try:
        assert bg.submit("meta2", **kwargs)
        assert not bg.submit("again", **kwargs)          # one job at a time
        meta2, b, ms = _wait(bg)
    finally:
        bg.close()
    assert meta == "meta" and meta2 == "meta2" and ms > 0
    assert a["bpm"] == b["bpm"] and abs(a["bpm"] - truth) < 3


def test_invalidate_discards_result_in_flight():
    kwargs, _ = _job()
    an = BackgroundAnalyzer(background=False)
    an.submit("old", **kwargs)
    an.invalidate()
    assert an.poll() is None                              # stale result dropped
    assert an.submit("new", **kwargs)
    assert an.poll()[0] == "new"
