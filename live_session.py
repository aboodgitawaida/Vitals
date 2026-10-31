"""
live_session.py -- the live app's heart-rate logic, without camera or UI.

Takes per-frame features (skin-patch colours, wall colour, head pose, motion
flags) and decides, frame by frame, what the screen should say: the number,
the tracker state, the lighting advice and the status line. It follows the
same steps as the main loop of vitals_app.py and uses the same settings
(vitals_app imports them from here).

The browser version (web/) runs this file unchanged in Pyodide. The browser
only does what needs a camera: finding the face and measuring the colours.
"""

import math
from collections import deque

import numpy as np

from face_layout import REGIONS, TRACES, assess_lighting, patch_region
from vitals_dsp import GREEN_MARGIN_DB, HR_HIGH, HR_LOW, analyze_heart_rate
from vitals_stream import HeartRateTracker, SampleBuffer

# ---------------- Settings you can tune (shared with vitals_app.py) ----------------
WINDOW_SEC = 15          # seconds of history per estimate (15 s -> ~4 BPM frequency resolution)
MIN_SEC = 5              # first estimate after 5 s (the tracker asks more of short windows)
UPDATE_EVERY = 0.5       # seconds between estimates
SNR_GOOD_DB = 3.0        # at or above: "good signal" (show/hide thresholds live in HeartRateTracker)
FACE_GAP_SEC = 1.0       # face missing longer than this -> start over (shorter gaps are bridged)
TALKING_STD = 0.015      # lip gap (fraction of face width) varying more than this over 1 s = talking
MOTION_HOLD = 1.5        # seconds to pause readings after big movement
METHODS = ["auto", "pos", "green"]
TREND_SEC = 60           # length of the trend chart


def region_summary(analysis):
    """Best patch SNR and total weight share per region, for the dashboard."""
    summary = {r: (None, 0.0) for r in REGIONS}
    if not analysis:
        return summary
    for name, snr in analysis["patch_snr"].items():
        if snr is None:
            continue
        region = patch_region(name)
        best, share = summary[region]
        summary[region] = (snr if best is None else max(best, snr),
                           share + analysis["patch_weight"].get(name, 0.0))
    return summary


def _num(x):
    """float, or None for None / NaN (JSON has no NaN)."""
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


class LiveSession:
    def __init__(self, use_motion=True):
        self.use_motion = use_motion
        self.method = "auto"
        self.use_background = True
        self.buffer = SampleBuffer(TRACES, WINDOW_SEC, FACE_GAP_SEC)    # rolling face and patch colours
        self.tracker = HeartRateTracker()                               # decides what number to show
        self.trend = deque()                     # (time, bpm or None) for the trend chart
        self.talking_level = deque(maxlen=30)    # mouth opening over the last ~1 s
        self.lighting = ("none", "Light: waiting for face")
        self.last_motion = self.last_small_motion = -99.0
        self.status = ""
        self.reset()

    # ---------------- controls ----------------

    def reset(self):
        """Start over (long face loss, light mode switched: old samples no longer match)."""
        self.buffer.reset()
        self.tracker.reset()
        self.analysis = None
        self.reading = self.tracker.reading(0.0)
        self.last_update = -1e9

    def set_method(self, method):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method = method
        self.tracker.reset()

    def set_background(self, on):
        self.use_background = bool(on)
        self.tracker.reset()

    # ---------------- per frame ----------------

    def frame(self, now, face, colors=None, bg_rgb=None, pose=None, fast_motion=False,
              small_motion=False, stats=None, fps=0.0, analyze=True):
        """
        One camera frame. colors: {patch: RGB or None}; stats: lighting numbers from
        the colour measurement. analyze=False skips the (slow) analysis on this frame
        even if one is due, like the desktop app does while its worker is busy.
        Returns True if an analysis ran.
        """
        ran = False
        if not face:
            # Brief loss (blink, hand passing): keep the data and ride it out.
            if self.buffer.add_missing(now):
                self.status = "Face lost - hold still"
                self.reading = self.tracker.reading(now)
            else:
                self.reset()
                self.status = "No face detected"
            return ran

        if fast_motion:
            self.last_motion = now
        if small_motion:
            self.last_small_motion = now
        if pose is not None:
            self.talking_level.append(pose[4])
        # Fast-motion frames are stored as gaps (filled by interpolation), so a
        # head turn doesn't poison the next 15 s of windows.
        if fast_motion:
            self.buffer.add_unusable(now)
        else:
            self.buffer.add(now, colors, bg_rgb, pose)

        have = self.buffer.span()
        moving = now - self.last_motion < MOTION_HOLD
        if have < MIN_SEC:
            self.lighting = assess_lighting(stats, fps)
            self.reading = self.tracker.update(now)
        elif analyze and now - self.last_update >= UPDATE_EVERY:
            self.lighting = assess_lighting(stats, fps)
            # If the head moved at all during this window, make green prove itself (see vitals_dsp).
            margin = GREEN_MARGIN_DB + (3.0 if now - self.last_small_motion < WINDOW_SEC else 0.0)
            times, traces, background = self.buffer.arrays()
            a = analyze_heart_rate(
                times, traces, background, method=self.method, prev_bpm=self.reading.prior_bpm,
                use_background=self.use_background, green_margin_db=margin,
                pose=self.buffer.pose_array(), use_motion=self.use_motion)
            self.analysis = a
            self.reading = self.tracker.update(now, a and a["bpm"], a and a["snr_db"], usable=not moving,
                                               window_sec=have, agreement=a and a["agreement"])
            self.trend.append((now, self.reading.bpm))
            self.last_update = now
            ran = True
        while self.trend and now - self.trend[0][0] > TREND_SEC:
            self.trend.popleft()

        if have < MIN_SEC:
            self.status = f"Hold still... collecting {have:.0f}/{MIN_SEC} s"
        elif moving:
            self.status = "Movement detected - hold still"
        elif len(self.talking_level) > 10 and np.std(self.talking_level) > TALKING_STD:
            self.status = "Talking detected - stay quiet for the best reading"
        elif self.reading.state == "holding":
            self.status = "Weak signal - holding last reading"
        elif self.reading.bpm is None:
            self.status = self.lighting[1] if self.lighting[0] != "good" else "Weak signal - keep still"
        else:
            self.status = ""
        return ran

    # ---------------- what to show ----------------

    def level(self):
        """Colour of the reading: good / fair / none."""
        a, r = self.analysis, self.reading
        if r.bpm is None:
            return "none"
        if r.state == "holding":
            return "fair"
        # "Good" needs a clean spectrum AND all three methods agreeing.
        good = a is not None and a["snr_db"] >= SNR_GOOD_DB and a["agreement"] == 3
        return "good" if good else "fair"

    def info(self):
        """Everything the dashboard draws, as plain JSON-friendly values."""
        a, r = self.analysis, self.reading
        out = {
            "bpm": _num(r.bpm), "state": r.state, "level": self.level(), "status": self.status,
            "lighting": list(self.lighting), "method": self.method,
            "use_background": self.use_background, "use_motion": self.use_motion,
            "collected_sec": float(self.buffer.span()), "min_sec": MIN_SEC,
            "trend": [[float(t), _num(b)] for t, b in self.trend],
            "regions": {k: [_num(s), float(w)] for k, (s, w) in region_summary(a).items()},
            "snr_db": None, "method_used": None, "agreement": None, "peak_bpm": None,
            "motion_patches": None, "patch_snr": {}, "spectrum": None, "pulse": None, "source": None,
        }
        if a is not None:
            band = (a["freqs"] >= HR_LOW - 0.2) & (a["freqs"] <= HR_HIGH + 0.2)
            out.update({
                "snr_db": _num(a["snr_db"]), "method_used": a["method_used"],
                "agreement": int(a["agreement"]), "peak_bpm": _num(a["bpm"]),
                "motion_patches": int(a["motion_patches"]), "source": a["source"],
                "patch_snr": {k: _num(v) for k, v in a["patch_snr"].items()},
                # Every 4th bin is plenty for a chart and keeps the message small.
                "spectrum": [(a["freqs"][band][::4] * 60).tolist(), a["power"][band][::4].tolist()],
                "pulse": a["pulse"][-300:].tolist(),
            })
        return out
