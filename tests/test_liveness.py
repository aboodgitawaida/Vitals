"""Liveness check: randomness, the flash test, the fairness rule and the full flow."""

import numpy as np

from face_regions import PATCHES
from liveness import (CHALLENGE_SEC, PALETTE, SEGMENT_SEC, FlashResult, LivenessSession, ProgressivePulse,
                      PulseResult, colour_at, flash_check, make_challenge, verdict)
from session_benchmark import make_session


def test_challenge_is_random_and_well_formed():
    a = make_challenge(np.random.default_rng(1))
    b = make_challenge(np.random.default_rng(2))
    assert [s[2] for s in a] != [s[2] for s in b]                       # unpredictable
    for (s0, e0, c0), (s1, e1, c1) in zip(a, a[1:]):
        assert c0 != c1 and abs(e0 - s1) < 1e-9                         # every boundary is a real change
    assert all(SEGMENT_SEC[0] - 1e-9 <= e - s <= SEGMENT_SEC[1] + 1e-9 for s, e, _ in a[:-1])
    assert a[-1][1] == CHALLENGE_SEC
    assert colour_at(a, 0.0) == PALETTE[a[0][2]] and colour_at(a, CHALLENGE_SEC + 1) is None


def _flash(reacts, glint_only=False, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 7.0, 1 / 30)
    ch = make_challenge(rng)
    shown = np.array([colour_at(ch, x - 0.5) or (255, 255, 255) for x in t], float)
    seen = np.array([colour_at(ch, x - 0.5 - 0.1) or (255, 255, 255) for x in t], float) / 255
    patches = {}
    for i, n in enumerate(PATCHES):
        base = np.array([180.0, 120.0, 95.0]) * np.ones((t.size, 1))
        if reacts and (not glint_only or i < 2):
            base = base * (1 + 0.3 * seen) if not glint_only else base + 10 * seen
        patches[n] = base + rng.normal(0, 0.5, base.shape)
    wall = np.array([185.0, 180.0, 170.0]) * (1 + 0.02 * seen * reacts) + rng.normal(0, 0.2, (t.size, 3))
    return flash_check(t, shown, patches, wall)


def test_flash_passes_for_reacting_skin():
    r = _flash(reacts=True)
    assert r.verdict == "pass" and r.r > 0.8 and 50 <= r.delay_ms <= 150 and r.coverage > 0.9


def test_flash_fails_for_video_that_ignores_the_screen():
    assert _flash(reacts=False).verdict == "fail"


def test_flash_fails_for_a_glint_in_one_spot():
    r = _flash(reacts=True, glint_only=True)
    assert r.verdict != "pass" and r.coverage < 0.3


def test_weak_pulse_never_means_not_live():
    """Fairness rule: weak or absent pulse can block LIVE, but never produces NOT LIVE."""
    flash_ok = FlashResult("pass", 0.9, 100, 1.0, 0.1)
    for pv in ("weak", "absent"):
        assert verdict(PulseResult(pv, None, None), flash_ok) == "NOT CONFIRMED"
    assert verdict(PulseResult("pass", 70, 5), flash_ok) == "LIVE"
    assert verdict(PulseResult("pass", 70, 5), FlashResult("fail", 0.1, 0, 0, 0)) == "NOT LIVE"


def test_progressive_pulse_rules():
    strong, ok, weak = PulseResult("pass", 70, 4.0), PulseResult("pass", 70, 0.5), PulseResult("weak", 70, -0.5)
    p = ProgressivePulse()
    assert p.add(10, strong) is strong and p.confirmed_at == 10          # strong: confirm at once
    p = ProgressivePulse()
    assert p.add(10, ok) is None                                         # one ordinary pass is not enough
    assert p.add(15, ok) is ok and p.confirmed_at == 15                  # two in a row
    p = ProgressivePulse()
    for look in ProgressivePulse.LOOKS[:-1]:
        assert p.add(look, weak) is None
    final = p.add(30, ok)                                                # passed once, at the very end
    assert final.verdict == "weak" and p.confirmed_at is None


def _run_session(reacts, seed=3):
    s = make_session("lighter", 0.4, np.random.default_rng(seed), seconds=40, events=False)
    sess = LivenessSession(0.0, PATCHES, rng=np.random.default_rng(seed))
    shown = (255, 255, 255)
    history = [np.ones(3)]
    for i, now in enumerate(s["t"]):
        seen = history[max(0, len(history) - 3)]
        gain = (1 + 0.3 * seen) if reacts else np.ones(3)
        colors = {n: s["patches"][n][i] * gain for n in PATCHES}
        if not sess.frame(now, colors, s["background"][i], s["pose"][i], emitted=shown):
            break
        shown = sess.screen_colour(now)
        history.append(np.asarray(shown, float) / 255)
    return sess


def test_full_flow_live_face():
    sess = _run_session(reacts=True)
    assert sess.result == "LIVE" and sess.pulse_confirmed_at == 10


def test_full_flow_replayed_video():
    sess = _run_session(reacts=False)
    assert sess.result == "NOT LIVE"


def test_face_lost_stops_the_check():
    sess = LivenessSession(0.0, PATCHES)
    sess.frame(0.0, {n: np.array([150.0, 110, 90]) for n in PATCHES}, np.array([180.0, 180, 170]))
    t = 0.0
    while sess.frame(t := t + 1 / 30, None, None):
        pass
    assert sess.result == "NOT CONFIRMED" and "face lost" in sess.evidence()[0]
