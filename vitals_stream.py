"""
vitals_stream.py -- everything that happens BETWEEN "one analysis of a 15 s
window" and "the number on screen".

Two pieces, both free of camera code so the app, replay.py and the benchmarks
all run exactly the same logic:

  SampleBuffer       the rolling 15 s of per-patch colours. Survives short face
                     dropouts instead of throwing everything away.
  HeartRateTracker   decides what to SHOW. A small state machine, the same idea
                     a pulse oximeter uses: lock on, ride through short weak
                     patches, and only give up after a few seconds.

Why this matters: in v3, any single weak window (SNR < 0 dB) or a single frame
without a face blanked the reading and wiped its history. With a real-world
SNR hovering around 0 dB, the number flickered on and off even when the
underlying estimate was fine.

States shown to the user (never hidden -- a held value is labelled as held):
  acquiring  not enough data yet, or no reading has been confirmed
  locked     the latest window produced a reading we accepted
  holding    recent windows were weak or the head moved; showing the last
             accepted value for at most HOLD_SEC, clearly marked
  lost       weak for too long; showing "--" until we re-acquire
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from vitals_dsp import analyze_heart_rate

# Per-frame signals from FaceMesh used for motion correction (see vitals_dsp.remove_nuisance):
# where the head is (x, y), how close (scale), its tilt (roll), and how open the mouth is
# (mouth: talking moves the cheeks in a 1-4 Hz rhythm, right in the heart-rate band).
POSE_FIELDS = ("x", "y", "scale", "roll", "mouth")


# --------------------------------------------------------------------------
# Rolling sample buffer
# --------------------------------------------------------------------------

class SampleBuffer:
    """
    The last `window_sec` seconds of per-patch mean RGB plus the background.

    A frame with no face is stored as a row of NaN instead of resetting.
    vitals_dsp fills short NaN gaps by interpolation, so blinking or a hand
    passing your chin costs a few frames, not 15 seconds of data. Only a long
    absence (`max_gap_sec`) clears the buffer, because by then you have
    probably moved and the old samples no longer match the new position.
    """

    def __init__(self, patch_names, window_sec=15.0, max_gap_sec=1.0):
        self.patch_names = list(patch_names)
        self.window_sec = window_sec
        self.max_gap_sec = max_gap_sec
        self.reset()

    def reset(self):
        self.times = deque()
        self.traces = {n: deque() for n in self.patch_names}
        self.background = deque()
        self.pose = deque()
        self.last_face_time = None

    def add(self, t, colors, background_rgb, pose=None):
        """One frame with a face. colors: {patch: RGB or None}; pose: POSE_FIELDS values or None."""
        missing = np.full(3, np.nan)
        self.times.append(t)
        for n in self.patch_names:
            c = colors.get(n)
            self.traces[n].append(missing if c is None else np.asarray(c, dtype=float))
        self.background.append(missing if background_rgb is None else np.asarray(background_rgb, dtype=float))
        self.pose.append(None if pose is None else np.asarray(pose, dtype=float))   # None = no pose this frame
        self.last_face_time = t
        self._trim(t)

    def add_unusable(self, t):
        """
        One frame where the face IS visible but the head is moving fast. Stored
        as NaN like a missing frame, so the motion never enters the spectrum.
        Without this, a 1 s head turn spoils every window for the next 15 s.
        """
        self.add(t, {}, None)

    def add_missing(self, t):
        """
        One frame with no face. Returns False (and resets) if the face has been
        gone longer than max_gap_sec, True if we are still riding it out.
        """
        last_seen = self.last_face_time
        if last_seen is None or t - last_seen > self.max_gap_sec:
            self.reset()
            return False
        self.add(t, {}, None)
        self.last_face_time = last_seen      # add() moved it; a NaN frame is not a face
        return True

    def _trim(self, now):
        while self.times and now - self.times[0] > self.window_sec:
            self.times.popleft()
            self.background.popleft()
            self.pose.popleft()
            for n in self.patch_names:
                self.traces[n].popleft()

    def span(self):
        return self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0

    def arrays(self):
        """(times, {patch: (N, 3)}, background (N, 3)) ready for analyze_heart_rate."""
        return (np.array(self.times),
                {n: np.array(self.traces[n]) for n in self.patch_names},
                np.array(self.background))

    def pose_array(self):
        """(N, len(POSE_FIELDS)) head pose, or None if no frame in the window had one."""
        # Width comes from the poses actually stored (older recordings have 4, newer 5),
        # so frames without a pose become NaN rows of the same width.
        width = next((len(p) for p in self.pose if p is not None), None)
        if width is None:
            return None
        empty = np.full(width, np.nan)
        p = np.array([empty if x is None else x for x in self.pose])
        return None if np.isnan(p).all() else p


# --------------------------------------------------------------------------
# What to show: the tracker
# --------------------------------------------------------------------------

@dataclass
class Reading:
    bpm: float | None        # value to display, or None for "--"
    state: str               # acquiring / locked / holding / lost
    snr_db: float | None     # SNR of the latest window (for the UI)
    prior_bpm: float | None  # what the peak picker should gently favour next time


class HeartRateTracker:
    """
    Turns a stream of per-window estimates (bpm, snr) into a steady display.

    The rules, and why each one is there:

    1. Two-level SNR threshold (hysteresis).
       To START showing a number, a window must reach SNR_ENTER (0 dB).
       Once locked, a weaker window (down to SNR_KEEP) is still accepted IF its
       peak is within GATE_BPM of the current reading. A weak peak that agrees
       with the last few seconds is far more believable than a weak peak
       somewhere random: the gate covers only ~10% of the 42-240 BPM band.
       `python session_benchmark.py --calibrate` measures how often pure
       noise passes each rule instead of assuming it.

    2. Jumps must be confirmed.
       Heart rate can change, but not 30 BPM in half a second. A strong reading
       far from the current track is a "candidate"; we switch only if it
       repeats (within CONFIRM_BPM) for CONFIRM_SEC. A single noise or harmonic
       peak (your 62-vs-78 BPM problem) can no longer grab the display.

    3. Weak windows can't keep a reading alive forever.
       A SNR_KEEP window only counts if a strong (SNR_ENTER) window was seen in
       the last REFRESH_SEC. So when the pulse truly disappears (lights off,
       face turned away), the reading lets go within a bounded time instead of
       being propped up by noise that happens to land near the track.

    4. Hold, then let go.
       No acceptable window (weak, or head moving)? Keep showing the last value
       as "holding" for up to HOLD_SEC, then show "--". The tracking prior is
       kept a bit longer (PRIOR_SEC) so re-acquiring lands on the same track.

    5. Confirm the first reading.
       From acquiring/lost, a strong reading must repeat for ACQUIRE_SEC before
       a number appears. Why 1.5 s and not "two windows in a row"? Windows
       0.5 s apart share 14.5 of their 15 seconds of data, so two in a row are
       nearly the same measurement and agree even when it is noise. 1.5 s adds
       a little genuinely new data; session_benchmark.py showed the 0.5 s
       version letting a noise peak through on dim, darker-skin sessions.

    6. Methods must agree.
       analyze_heart_rate also finds the peak with POS, CHROM and green
       separately. If no second method agrees (within 5 BPM), the estimate is
       ignored, like a window with motion. Measured on the synthetic scenarios:
       windows that passed the SNR bar but had only 1/3 agreement were more
       than 10 BPM wrong 67% of the time; with 2/3 or 3/3, essentially never.

    7. Start early, but with a stricter bar.
       The app starts analysing after 5 s instead of waiting for a full 15 s
       window. A short window has coarser frequency resolution, and pure noise
       reaches a given SNR more easily (measured: at 5 s, noise passes 0 dB
       ~4% of the time vs ~0% at 15 s). So for windows shorter than
       FULL_BAR_SEC both thresholds are raised by up to SHORT_WINDOW_EXTRA_DB.

    The displayed value is the median of the last few accepted estimates.

    Settings were chosen with session_benchmark.py (coverage, "confidently
    wrong" time, and how fast the number disappears when the pulse does) on
    SIMULATED data. They get re-checked on real recordings in Phase 4.
    """

    SNR_ENTER = 0.0      # dB, start showing a number
    SNR_KEEP = -1.5      # dB, keep showing it if the peak agrees with the track
    GATE_BPM = 10.0      # "agrees with the track" = within this many BPM
    CONFIRM_BPM = 5.0    # two estimates "agree" if within this
    CONFIRM_SEC = 2.0    # a jump must persist this long to be believed
    ACQUIRE_SEC = 1.5    # a first reading must persist this long before it is shown
    REFRESH_SEC = 4.0    # a weak window only counts if a strong one was seen this recently
    HOLD_SEC = 3.0       # show a held value at most this long
    PRIOR_SEC = 10.0     # keep favouring the old track this long after the last accept
    SMOOTH_N = 5         # median of this many accepted estimates
    MIN_AGREEMENT = 2             # of 3 methods (POS, CHROM, green) must agree on the peak
    SHORT_WINDOW_EXTRA_DB = 1.0   # extra SNR needed for the shortest (5 s) windows
    SHORT_WINDOW_SEC = 5.0        # window length that gets the full extra
    FULL_BAR_SEC = 8.0            # from this window length on, no extra

    def __init__(self, **overrides):
        for k, v in overrides.items():
            if not hasattr(self, k):
                raise AttributeError(f"unknown tracker setting {k}")
            setattr(self, k, v)
        self.reset()

    def reset(self):
        self.track = None            # current smoothed value
        self.last_accept = None      # time of the last accepted estimate
        self.last_strong = None      # time of the last accepted estimate with SNR >= SNR_ENTER
        self.history = deque(maxlen=self.SMOOTH_N)
        self.candidate = None        # (first seen, last seen, [estimates]) of an unconfirmed value
        self.last_snr = None

    def _start_track(self, now, estimates):
        """Begin a new track, seeded with the agreeing estimates that confirmed it."""
        self.history.clear()
        self.history.extend(estimates)
        self.track = float(np.median(self.history))
        self.last_accept = self.last_strong = now
        self.candidate = None

    def _accept(self, now, bpm, strong):
        self.history.append(bpm)
        self.track = float(np.median(self.history))
        self.last_accept = now
        if strong:
            self.last_strong = now
        self.candidate = None

    def _consider_candidate(self, now, bpm, need_sec):
        """
        Track a not-yet-believed value. Returns True once it has repeated
        (each time within CONFIRM_BPM of the last, with no gap > 1.5 s) for
        need_sec.
        """
        c = self.candidate
        if c is None or abs(bpm - c[2][-1]) > self.CONFIRM_BPM or now - c[1] > 1.5:
            self.candidate = (now, now, [bpm])
            return False
        self.candidate = (c[0], now, c[2] + [bpm])
        return now - c[0] >= need_sec

    def extra_db(self, window_sec):
        """How much stricter both SNR thresholds are for a window this short."""
        if window_sec is None:
            return 0.0
        span = self.FULL_BAR_SEC - self.SHORT_WINDOW_SEC
        frac = np.clip((self.FULL_BAR_SEC - window_sec) / span, 0.0, 1.0)
        return float(self.SHORT_WINDOW_EXTRA_DB * frac)

    def update(self, now, bpm=None, snr_db=None, usable=True, window_sec=None, agreement=None):
        """
        One new window. bpm/snr_db from analyze_heart_rate (None if no face
        data); usable=False when the head moved (estimate not trusted);
        window_sec = how many seconds of data the estimate used;
        agreement = how many of the 3 methods agree with bpm (None = unknown).
        """
        self.last_snr = snr_db
        if agreement is not None and agreement < self.MIN_AGREEMENT:
            usable = False                      # one method alone: not trusted
        has_track = self.track is not None and now - self.last_accept <= self.HOLD_SEC
        if bpm is not None and snr_db is not None and usable:
            snr_db = snr_db - self.extra_db(window_sec)   # short window: same rules, higher bar
            if has_track:
                near = abs(bpm - self.track) <= self.GATE_BPM
                strong = snr_db >= self.SNR_ENTER
                refreshed = now - self.last_strong <= self.REFRESH_SEC
                if near and (strong or (snr_db >= self.SNR_KEEP and refreshed)):
                    self._accept(now, bpm, strong)
                elif not near and snr_db >= self.SNR_ENTER:
                    # Possible real change: believe it only if it persists.
                    if self._consider_candidate(now, bpm, self.CONFIRM_SEC):
                        self._start_track(now, self.candidate[2])
            elif snr_db >= self.SNR_ENTER:
                # No current track: a strong reading must persist ACQUIRE_SEC.
                # (No shortcut for "same value as before the gap": the peak
                # picker is already biased toward the old value, so agreeing
                # with it is weak evidence. Benchmarked: the shortcut raised
                # false readings on pure noise ~10x.)
                if self._consider_candidate(now, bpm, self.ACQUIRE_SEC):
                    self._start_track(now, self.candidate[2])

        return self.reading(now)

    def prior(self, now=None):
        if self.track is None or self.last_accept is None:
            return None
        if now is not None and now - self.last_accept > self.PRIOR_SEC:
            return None
        return self.track

    def reading(self, now):
        if self.track is None:
            return Reading(None, "acquiring", self.last_snr, None)
        age = now - self.last_accept
        prior = self.prior(now)
        if age < 1e-9:
            return Reading(self.track, "locked", self.last_snr, prior)
        if age <= self.HOLD_SEC:
            return Reading(self.track, "holding", self.last_snr, prior)
        return Reading(None, "lost", self.last_snr, prior)


class LegacyDisplay:
    """
    The v3 display logic, kept only so replay.py / session_benchmark.py can
    compare old vs new on exactly the same data. Any weak or moving window
    blanks the number and wipes the median history.
    """

    def __init__(self, snr_min_db=0.0, smooth_n=5):
        self.snr_min_db = snr_min_db
        self.history = deque(maxlen=smooth_n)
        self.bpm = None

    def reset(self):
        self.history.clear()
        self.bpm = None

    def update(self, now, bpm=None, snr_db=None, usable=True, window_sec=None, agreement=None):
        if bpm is None or snr_db is None or not usable or snr_db < self.snr_min_db:
            self.reset()
            return Reading(None, "lost", snr_db, None)
        self.history.append(bpm)
        self.bpm = float(np.median(self.history))
        return Reading(self.bpm, "locked", snr_db, self.bpm)


# --------------------------------------------------------------------------
# Offline: run a whole recording through buffer -> analysis -> display
# --------------------------------------------------------------------------

def simulate_display(times, patches, background, face, moving, display, max_gap_sec=1.0,
                     drop_motion=True, window_sec=15.0, min_sec=5.0, update_every=0.5,
                     motion_hold=1.5, method="auto", pose=None):
    """
    Replays per-frame data in the same order as vitals_app.py and returns
    [(time, bpm shown or None, state), ...] once per update.

    Used by replay.py (your recordings) and session_benchmark.py (simulated
    sessions), so both measure the logic the app actually runs.
      face:   (N,) bool, False = FaceMesh found no face in that frame
      moving: (N,) bool, True = fast head motion in that frame
      pose:   (N, K) head pose per frame (POSE_FIELDS), or None (no motion correction)
    """
    names = list(patches)
    buf = SampleBuffer(names, window_sec, max_gap_sec)
    last_update = last_motion = -1e9
    prior, out = None, []
    for i, now in enumerate(times):
        if not face[i]:
            if not buf.add_missing(now):
                display.reset()
        elif moving[i] and drop_motion:
            buf.add_unusable(now)
        else:
            buf.add(now, {n: patches[n][i] for n in names},
                    None if background is None else background[i],
                    None if pose is None else pose[i])
        if moving[i]:
            last_motion = now
        if now - last_update < update_every:
            continue
        last_update = now
        if buf.span() < min_sec:
            reading = display.update(now)
        else:
            t, traces, bg = buf.arrays()
            r = analyze_heart_rate(t, traces, None if background is None else bg, method=method,
                                   prev_bpm=prior, patch_map=False, pose=buf.pose_array())
            reading = display.update(now, r and r["bpm"], r and r["snr_db"],
                                     usable=now - last_motion >= motion_hold, window_sec=buf.span(),
                                     agreement=r and r.get("agreement"))
        prior = reading.prior_bpm
        out.append((float(now), reading.bpm, reading.state))
    return out
