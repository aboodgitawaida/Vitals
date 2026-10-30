"""signal_lite (numpy-only, used in the browser) matches scipy.signal for what vitals_dsp uses."""

import numpy as np
import pytest
from scipy import signal

import signal_lite


@pytest.mark.parametrize("order,wn,btype", [
    (3, [0.7 / 15, 4.0 / 15], "band"),     # vitals_dsp.bandpass
    (3, 5.0 / 15, "high"),                 # vitals_dsp.noise_level
    (2, [0.1 / 15, 0.5 / 15], "band"),
    (4, 0.3, "low"),
])
def test_butter_matches_scipy(order, wn, btype):
    b1, a1 = signal.butter(order, wn, btype=btype)
    b2, a2 = signal_lite.butter(order, wn, btype=btype)
    assert np.allclose(b1, b2, rtol=1e-9, atol=1e-14)
    assert np.allclose(a1, a2, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("n", [40, 151, 450])
@pytest.mark.parametrize("shape", [(), (3,), (16,)])
def test_filtfilt_matches_scipy(n, shape):
    rng = np.random.default_rng(n)
    x = rng.normal(size=(n,) + shape).cumsum(axis=0)
    for b, a in (signal.butter(3, [0.7 / 15, 4.0 / 15], btype="band"), signal.butter(3, 5.0 / 15, btype="high")):
        want = signal.filtfilt(b, a, x, axis=0)
        got = signal_lite.filtfilt(b, a, x, axis=0)
        assert got.shape == want.shape
        assert np.allclose(got, want, rtol=1e-7, atol=1e-9 * np.abs(want).max())


def test_filtfilt_too_short_raises_like_scipy():
    b, a = signal.butter(3, [0.1, 0.4], btype="band")
    with pytest.raises(ValueError):
        signal_lite.filtfilt(b, a, np.zeros(10))


def test_detrend_matches_scipy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300) + np.linspace(0, 5, 300)
    assert np.allclose(signal_lite.detrend(x), signal.detrend(x), atol=1e-10)
    m = rng.normal(size=(200, 3)).cumsum(axis=0)
    assert np.allclose(signal_lite.detrend(m, axis=0), signal.detrend(m, axis=0), atol=1e-10)


def test_live_session_same_without_scipy(monkeypatch):
    """The whole pipeline gives the same reading with signal_lite as with scipy."""
    import vitals_dsp
    from face_layout import PATCHES
    from live_session import LiveSession
    from session_benchmark import make_session

    s = make_session("medium", 0.6, np.random.default_rng(11), seconds=25, fidget=1.0)
    stats = {"luminance": {"forehead": 150.0}, "clipped_fraction": 0.0}

    def run():
        session, out = LiveSession(), []
        for i, now in enumerate(s["t"]):
            colors = {n: s["patches"][n][i] for n in PATCHES}
            if session.frame(now, bool(s["face"][i]), colors, s["background"][i], s["pose"][i],
                             fast_motion=bool(s["moving"][i]), stats=stats, fps=30.0):
                a = session.analysis
                out.append((a and a["bpm"], a and a["snr_db"], session.reading.state))
        return out

    with_scipy = run()
    for name in ("butter", "filtfilt", "detrend"):
        monkeypatch.setattr(vitals_dsp, name, getattr(signal_lite, name))
    without = run()
    assert len(with_scipy) == len(without) > 20
    assert sum(b is not None for b, _, _ in with_scipy) > 20
    for (b1, s1, st1), (b2, s2, st2) in zip(with_scipy, without):
        assert st1 == st2
        assert (b1 is None) == (b2 is None)
        if b1 is not None:
            assert b1 == pytest.approx(b2, abs=1e-6) and s1 == pytest.approx(s2, abs=1e-6)
