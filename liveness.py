"""
liveness.py -- "is there a live person in front of this screen right now?"

A research PROTOTYPE, not a security product. It combines two independent
checks, each covering the other's blind spot:

1. PULSE CHECK (catches photos and masks)
   A real face has blood pumping under the skin, so its colour pulses at the
   heart rate -- and it is the SAME heart rate on the forehead and on both
   cheeks. A photo or mask has no pulse; a fake that only partly pulses tends
   to disagree between regions.
   Blind spot: a pre-recorded video of a real person DOES contain a pulse.

2. FLASH CHALLENGE (catches pre-recorded and injected video)
   The screen flashes a RANDOM sequence of colours (random order, random
   timing, decided only when the check starts). Real skin a few tens of
   centimetres from the screen catches that light, so its tint follows the
   sequence within a fraction of a second. A recording cannot know a sequence
   that did not exist when it was made; a video fed in as a virtual camera
   never sees the screen at all; a phone held up to the camera only glints in
   one or two spots.
   How the response is measured, and why:
   * TINT, not brightness: we track (R-B)/(R+G+B) and (G-(R+B)/2)/(R+G+B).
     The camera's auto-exposure fights brightness changes within a second,
     but leaves the tint mostly alone.
   * SKIN MINUS WALL: auto white-balance shifts the tint of the whole image.
     Subtracting the background wall cancels that; a face near the screen
     catches far more of the flash than the wall behind it.
   * DELAY: screen + camera take ~50-300 ms. We search delays 0-0.5 s.
   * COVERAGE: most skin patches must respond, not just one spot.
   Blind spot: a real-time face-swap of a real person sitting in front of the
   screen. Their real face DOES reflect the flash and DOES have a pulse, and
   face-swap software often carries both through. That is an open research
   problem; we report it as a known limitation rather than pretend otherwise.

Verdict:
  LIVE            both checks pass
  NOT LIVE        the face did not react to the flashes: strong physical evidence
                  of a replayed or injected video
  NOT CONFIRMED   anything weaker, including "no pulse found". Photos and masks
                  end up here, never at LIVE.
Why a weak pulse never gives NOT LIVE: melanin absorbs light, so darker skin
gives a weaker pulse signal, and in dim light a real but weak pulse looks just
like no pulse. The first version of this check turned "weak" into "fake" and
rejected every simulated darker-skinned person in normal light. Never
accepting a weak result is safe; accusing a real person of being fake is not.

Re-score a recorded attempt (colours only, never images):
    python liveness.py recordings/liveness_YYYYMMDD_HHMMSS
"""

import argparse
import csv
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d

from vitals_dsp import FS, HR_HIGH, HR_LOW, _fill_missing, analyze_heart_rate, resample_uniform, snr_db

# Challenge colours (RGB). Saturated colours give the biggest change in tint.
PALETTE = {"red": (255, 40, 40), "blue": (40, 60, 255), "green": (40, 255, 60), "white": (255, 255, 255)}
SEGMENT_SEC = (0.35, 0.8)     # each colour lasts a random time in this range
CHALLENGE_SEC = 6.0
MAX_DELAY_SEC = 0.5           # screen + camera delay we are willing to believe

# Flash thresholds (simulation-calibrated; re-check on real attempts with this file's CLI)
FLASH_R_PASS = 0.6            # correlation between the sequence and the skin's tint
FLASH_R_FAIL = 0.3
FLASH_COVER_PASS = 0.6        # share of skin patches that individually follow the sequence
FLASH_GAIN_MIN = 0.01         # skin tint change per unit of screen tint change

# Pulse thresholds
PULSE_SNR_PASS = 0.0          # dB, overall. Pure noise reaches this in well under 1% of windows.
PULSE_SNR_ABSENT = -1.0       # below this (and no region support): "no pulse found"
REGION_SUPPORT_SNR = -3.0     # a region "supports" the heart rate if its own SNR AT that rate is above this
REGION_SUPPORT_MIN = 2        # of 3 regions


# --------------------------------------------------------------------------
# The challenge
# --------------------------------------------------------------------------

def make_challenge(rng, duration=CHALLENGE_SEC):
    """
    Random sequence of (start s, end s, colour name). Pass a fresh, unpredictable
    rng (the app seeds it from the operating system's randomness).
    Never the same colour twice in a row, so every boundary is a real change.
    """
    names = list(PALETTE)
    segments, t, prev = [], 0.0, None
    while t < duration:
        name = rng.choice([n for n in names if n != prev])
        length = rng.uniform(*SEGMENT_SEC)
        segments.append((t, min(t + length, duration), str(name)))
        t += length
        prev = name
    return segments


def colour_at(challenge, t):
    """RGB of the challenge at t seconds after it started (None outside it)."""
    for start, end, name in challenge:
        if start <= t < end:
            return PALETTE[name]
    return None


# --------------------------------------------------------------------------
# Measuring the response to the flashes
# --------------------------------------------------------------------------

def tint(rgb):
    """(N, 3) RGB -> (N, 2): red-vs-blue and green-vs-rest, independent of brightness."""
    rgb = np.asarray(rgb, dtype=float)
    total = rgb.sum(axis=1) + 1e-9
    return np.column_stack([(rgb[:, 0] - rgb[:, 2]) / total,
                            (rgb[:, 1] - 0.5 * (rgb[:, 0] + rgb[:, 2])) / total])


def _detrend(x, fs=FS, sec=1.5):
    """Remove slow drift (white-balance creeping, someone switching a lamp) but keep the flashes."""
    return x - uniform_filter1d(x, size=max(3, int(sec * fs)), axis=0, mode="nearest")


def _corr(a, b):
    a = (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-12)
    b = (b - b.mean(axis=0)) / (b.std(axis=0) + 1e-12)
    return float((a * b).mean())


@dataclass
class FlashResult:
    verdict: str                 # pass / fail / unclear
    r: float                     # how well the skin's tint follows the sequence (0..1)
    delay_ms: float              # best-fitting screen+camera delay
    coverage: float              # share of patches that follow it on their own
    gain: float                  # skin tint change per unit of screen tint change
    reason: str = ""


def flash_check(times, emitted_rgb, patch_traces, background):
    """
    times:        (N,) seconds
    emitted_rgb:  (N, 3) colour the app put on screen at each frame
    patch_traces: {patch: (N, 3) mean RGB, NaN = missing}
    background:   (N, 3) wall RGB, or None
    """
    t = np.asarray(times, dtype=float)
    em = _fill_missing(t, np.asarray(emitted_rgb, dtype=float))
    if em is None or t[-1] - t[0] < 3.0:
        return FlashResult("unclear", 0, 0, 0, 0, "challenge too short")
    _, em_u = resample_uniform(t, em)
    e = _detrend(tint(em_u))

    wall = None
    if background is not None:
        bg = _fill_missing(t, np.asarray(background, dtype=float))
        if bg is not None:
            wall = tint(resample_uniform(t, bg)[1])

    responses = []
    for rgb in patch_traces.values():
        rgb = _fill_missing(t, np.asarray(rgb, dtype=float))
        if rgb is None:
            continue
        skin = tint(resample_uniform(t, rgb)[1])
        if wall is not None:
            skin = skin - wall                  # cancels white-balance shifts of the whole image
        responses.append(_detrend(skin))
    if len(responses) < 3:
        return FlashResult("unclear", 0, 0, 0, 0, "face not visible during the flashes")

    pooled = np.mean(responses, axis=0)
    n = min(len(e), len(pooled))
    e, pooled, responses = e[:n], pooled[:n], [r[:n] for r in responses]

    # Try every plausible delay; keep the best fit.
    best_r, best_lag = -1.0, 0
    for lag in range(int(MAX_DELAY_SEC * FS) + 1):
        r = _corr(e[:n - lag], pooled[lag:])
        if r > best_r:
            best_r, best_lag = r, lag
    lag = best_lag
    e_l = e[:n - lag]
    coverage = float(np.mean([_corr(e_l, resp[lag:]) > FLASH_R_FAIL for resp in responses]))
    gain = float(np.sum(e_l * pooled[lag:]) / (np.sum(e_l ** 2) + 1e-12))
    delay_ms = 1000.0 * lag / FS

    if best_r >= FLASH_R_PASS and coverage >= FLASH_COVER_PASS and gain >= FLASH_GAIN_MIN:
        verdict, reason = "pass", "skin follows the random flashes"
    elif best_r < FLASH_R_FAIL or gain < FLASH_GAIN_MIN / 3:
        verdict, reason = "fail", "skin does not respond to the flashes"
    elif coverage < FLASH_COVER_PASS / 2:
        verdict, reason = "fail", "only a small spot responds (like a glossy screen)"
    else:
        verdict, reason = "unclear", "weak response - move closer to the screen or dim the room"
    return FlashResult(verdict, max(best_r, 0.0), delay_ms, coverage, gain, reason)


# --------------------------------------------------------------------------
# Pulse check
# --------------------------------------------------------------------------

@dataclass
class PulseResult:
    verdict: str                 # pass / weak / absent (never "fail": see module notes)
    bpm: float | None
    snr_db: float | None
    regions: dict = field(default_factory=dict)   # region -> (own peak bpm, SNR at the overall rate)
    agreeing: int = 0                             # regions supporting the overall heart rate
    reason: str = ""


def pulse_check(times, patch_traces, background=None, pose=None):
    """Same heart rate on the forehead and the cheeks? Uses the ~15 s before the flashes."""
    from face_regions import patch_region
    t = np.asarray(times, dtype=float)
    if len(t) < 2 or t[-1] - t[0] < 5.0:
        return PulseResult("unclear", None, None, reason="not enough data before the check")
    overall = analyze_heart_rate(t, patch_traces, background, pose=pose, patch_map=False)
    if overall is None:
        return PulseResult("unclear", None, None, reason="no usable skin")
    bpm, snr = overall["bpm"], overall["snr_db"]

    # Region support: does each region show pulse energy AT the overall heart rate?
    # (Asking each region to find its own peak is too strict: with 4-8 patches a region
    # is noisy, and real pulses failed that test 40% of the time in dim light.)
    regions = {}
    for region in ("forehead", "left cheek", "right cheek"):
        subset = {n: v for n, v in patch_traces.items() if patch_region(n) == region}
        r = analyze_heart_rate(t, subset, background, pose=pose, patch_map=False) if subset else None
        if r is None:
            regions[region] = (None, None)
            continue
        regions[region] = (r["bpm"], snr_db(r["freqs"], r["power"], bpm / 60.0, HR_LOW, HR_HIGH))
    supporting = sum(1 for _, s in regions.values() if s is not None and s >= REGION_SUPPORT_SNR)

    if snr >= PULSE_SNR_PASS and supporting >= REGION_SUPPORT_MIN:
        verdict, reason = "pass", f"pulse at {bpm:.0f} BPM, seen in {supporting}/3 face regions"
    elif snr < PULSE_SNR_ABSENT and supporting < REGION_SUPPORT_MIN:
        verdict, reason = "absent", "no pulse found - if you are real: more light, move closer, hold still"
    else:
        verdict, reason = "weak", "weak pulse - more light or hold stiller"
    return PulseResult(verdict, bpm, snr, regions, supporting, reason)


class ProgressivePulse:
    """
    Looks for the pulse at 10, 15, 20, 25 and 30 s and stops as soon as it is sure.

    Why progressive: a strong pulse is obvious in 10 s, but a weak one (darker
    skin, dim room) needs more data -- doubling the data adds ~3 dB of SNR.
    Making everyone wait 30 s would be annoying; giving up at 15 s would fail
    the people who most need the extra time.

    Why the two rules: every extra look is another chance for noise to pass by
    luck (5 looks: 0.7% false confirmations on pure noise, vs 0.3% for one).
      * SNR >= STRONG_SNR_DB on one look: pure noise essentially never gets
        there, so confirm immediately.
      * otherwise two looks in a row must pass (measured: 0 of 300 no-pulse
        trials got through).
    """

    LOOKS = (10.0, 15.0, 20.0, 25.0, 30.0)
    STRONG_SNR_DB = 3.0

    def __init__(self):
        self.results = []
        self.final = None             # PulseResult once decided
        self.confirmed_at = None

    def next_look(self):
        return None if self.final is not None else self.LOOKS[len(self.results)]

    def add(self, elapsed, result):
        """Record the look at `elapsed` seconds; returns the final result once decided."""
        self.results.append(result)
        last_two = self.results[-2:]
        if result.verdict == "pass" and result.snr_db >= self.STRONG_SNR_DB:
            self.final, self.confirmed_at = result, elapsed
        elif len(last_two) == 2 and all(r.verdict == "pass" for r in last_two):
            self.final, self.confirmed_at = result, elapsed
        elif len(self.results) == len(self.LOOKS):
            if result.verdict == "pass":            # passed only once, at the very end: not enough
                result = PulseResult("weak", result.bpm, result.snr_db, result.regions, result.agreeing,
                                     "pulse seen only once - try again with more light")
            self.final = result
        return self.final


def progressive_pulse_check(times, patch_traces, background=None, pose=None):
    """Offline version of ProgressivePulse over a recording of up to 30 s."""
    t = np.asarray(times, dtype=float)
    prog = ProgressivePulse()
    while prog.next_look() is not None:
        look = prog.next_look()
        sel = t - t[0] <= look
        if t[-1] - t[0] < look - 0.5:              # recording ended: decide with what we have
            sel = np.ones_like(t, dtype=bool)
        res = pulse_check(t[sel], {n: np.asarray(v)[sel] for n, v in patch_traces.items()},
                          None if background is None else np.asarray(background)[sel],
                          None if pose is None else np.asarray(pose)[sel])
        if prog.add(look, res) is None and not sel.all():
            continue
        if prog.final is None:                     # ran out of data
            prog.final = res
        break
    return prog.final, prog.confirmed_at


# --------------------------------------------------------------------------
# Combined verdict
# --------------------------------------------------------------------------

def verdict(pulse, flash):
    """LIVE if both pass; NOT LIVE only on flash failure (strong evidence); else NOT CONFIRMED."""
    if pulse.verdict == "pass" and flash.verdict == "pass":
        return "LIVE"
    if flash.verdict == "fail":
        return "NOT LIVE"
    return "NOT CONFIRMED"


def summary_lines(pulse, flash):
    """Short evidence lines for the dashboard."""
    p = (f"Pulse: {pulse.bpm:.0f} BPM, SNR {pulse.snr_db:+.1f} dB, {pulse.agreeing}/3 regions"
         if pulse.bpm is not None else "Pulse: none found")
    f = (f"Flash: follows sequence r={flash.r:.2f}, delay {flash.delay_ms:.0f} ms, "
         f"{flash.coverage:.0%} of skin responds")
    why = flash.reason if flash.verdict == "fail" else (pulse.reason if pulse.verdict != "pass" else flash.reason)
    return [f"{p}  [{pulse.verdict}]", f"{f}  [{flash.verdict}]", f"Why: {why}"]


# --------------------------------------------------------------------------
# The live check, step by step (driven by the app once per camera frame)
# --------------------------------------------------------------------------

class LivenessSession:
    """
    pulse  -> screen white (extra light on the face), look for the pulse for 10-30 s
    flash  -> 0.6 s white lead-in, the random colour challenge, 0.6 s tail
              (the tail catches the screen + camera delay of the last colours)
    result -> verdict and evidence

    The app calls frame() every camera frame and draws screen_colour(now).
    No camera or drawing code here, so the whole flow is testable.
    """

    LEAD_SEC, TAIL_SEC = 0.6, 0.6
    FACE_LOST_SEC = 2.0
    WHITE = (255, 255, 255)

    def __init__(self, now, patch_names, rng=None):
        from vitals_stream import SampleBuffer
        self.rng = rng if rng is not None else np.random.default_rng()   # OS randomness: unpredictable
        self.patch_names = list(patch_names)
        self.phase = "pulse"
        self.started = now
        self.pulse_buffer = SampleBuffer(self.patch_names, window_sec=31.0, max_gap_sec=self.FACE_LOST_SEC)
        self.progress = ProgressivePulse()
        self.pulse = self.pulse_confirmed_at = None
        self.pulse_data = None
        self.flash_start = None
        self.challenge = None
        self.flash_rows = []          # (time, emitted RGB, {patch: RGB}, background RGB)
        self.flash = None
        self.result = None
        self.message = ""
        self.last_face = now

    # ---- what to show
    def screen_colour(self, now):
        """RGB the whole screen should show right now."""
        if self.phase == "flash":
            return colour_at(self.challenge, now - self.flash_start - self.LEAD_SEC) or self.WHITE
        return self.WHITE

    def progress_text(self, now):
        if self.phase == "pulse":
            look = self.progress.next_look() or 30.0
            return (f"Step 1/2: hold still, finding your pulse... {now - self.started:4.1f} s "
                    f"(up to {ProgressivePulse.LOOKS[-1]:.0f} s)")
        if self.phase == "flash":
            left = self.LEAD_SEC + CHALLENGE_SEC + self.TAIL_SEC - (now - self.flash_start)
            return f"Step 2/2: keep looking at the screen... {max(left, 0):3.1f} s"
        return self.result or ""

    # ---- feeding frames
    def frame(self, now, colors, background_rgb, pose=None, emitted=None):
        """
        One camera frame. colors=None when no face was found.
        emitted: the RGB that was on screen for this frame (defaults to screen_colour(now)).
        Returns True while the check is still running.
        """
        if self.phase == "result":
            return False
        if colors is None:
            if now - self.last_face > self.FACE_LOST_SEC:
                self._finish_aborted("face lost - stay in front of the camera")
                return False
        else:
            self.last_face = now

        if self.phase == "pulse":
            if colors is None:
                self.pulse_buffer.add_missing(now)
            else:
                self.pulse_buffer.add(now, colors, background_rgb, pose)
            look = self.progress.next_look()
            if look is not None and now - self.started >= look:
                t, traces, bg = self.pulse_buffer.arrays()
                pose_arr = self.pulse_buffer.pose_array()
                if self.progress.add(look, pulse_check(t, traces, bg, pose_arr)) is not None:
                    self.pulse, self.pulse_confirmed_at = self.progress.final, self.progress.confirmed_at
                    self.pulse_data = (t, traces, bg, pose_arr)
                    self.phase, self.flash_start = "flash", now
                    self.challenge = make_challenge(self.rng)
            return True

        # flash phase
        shown = emitted if emitted is not None else self.screen_colour(now)
        missing = np.full(3, np.nan)
        cols = {n: (missing if colors is None or colors.get(n) is None else np.asarray(colors[n], float))
                for n in self.patch_names}
        self.flash_rows.append((now, np.asarray(shown, float), cols,
                                missing if background_rgb is None else np.asarray(background_rgb, float)))
        if now - self.flash_start >= self.LEAD_SEC + CHALLENGE_SEC + self.TAIL_SEC:
            self.flash = flash_check(*self.flash_data())
            self.result = verdict(self.pulse, self.flash)
            self.phase = "result"
            return False
        return True

    def flash_data(self):
        t = np.array([r[0] for r in self.flash_rows])
        emitted = np.array([r[1] for r in self.flash_rows])
        traces = {n: np.array([r[2][n] for r in self.flash_rows]) for n in self.patch_names}
        bg = np.array([r[3] for r in self.flash_rows])
        return t, emitted, traces, bg

    def _finish_aborted(self, why):
        self.phase, self.result, self.message = "result", "NOT CONFIRMED", why

    def evidence(self):
        if self.pulse is None or self.flash is None:
            return [f"Check stopped: {self.message}"]
        lines = summary_lines(self.pulse, self.flash)
        if self.pulse_confirmed_at:
            lines[0] += f" (after {self.pulse_confirmed_at:.0f} s)"
        return lines

    def save(self, prefix):
        """Save colours only (never images) so the attempt can be re-scored with this file's CLI."""
        if self.pulse_data is None or not self.flash_rows:
            return False
        save_attempt(prefix, self.patch_names, self.pulse_data, self.flash_data())
        return True


# --------------------------------------------------------------------------
# Saving and re-scoring attempts (colours only, never images)
# --------------------------------------------------------------------------

def save_attempt(prefix, patch_names, pulse_data, flash_data):
    """
    pulse_data: (times, {patch: (N,3)}, background (N,3), pose (N,4) or None)
    flash_data: (times, emitted (N,3), {patch: (N,3)}, background (N,3))
    """
    def write(path, times, extra_cols, extra, traces, bg):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["time_s"] + extra_cols
            for n in list(patch_names) + ["background"]:
                header += [f"{n}_r", f"{n}_g", f"{n}_b"]
            w.writerow(header)
            for i, t in enumerate(times):
                row = [f"{t:.4f}"] + [("" if np.isnan(v) else f"{v:.6f}") for v in extra[i]]
                for arr in [traces[n] for n in patch_names] + [bg]:
                    row += ["" if np.isnan(v) else f"{v:.4f}" for v in arr[i]]
                w.writerow(row)

    t, traces, bg, pose = pulse_data
    pose = pose if pose is not None else np.full((len(t), 5), np.nan)
    from vitals_stream import POSE_FIELDS
    pose_cols = [f"pose_{k}" for k in POSE_FIELDS][:pose.shape[1]]
    write(prefix + "_pulse.csv", t, pose_cols, pose, traces, bg)
    t, emitted, traces, bg = flash_data
    write(prefix + "_flash.csv", t, ["emit_r", "emit_g", "emit_b"], emitted, traces, bg)


def load_attempt(prefix):
    def read(path, extra_cols):
        with open(path) as f:
            rows = list(csv.reader(f))
        col = {k: i for i, k in enumerate(rows[0])}
        data = rows[1:]
        num = lambda r, k: float(r[col[k]]) if r[col[k]] != "" else np.nan
        t = np.array([num(r, "time_s") for r in data])
        extra = np.array([[num(r, k) for k in extra_cols] for r in data])
        names = sorted({h[:-2] for h in rows[0] if h.endswith("_r") and not h.startswith(("emit", "pose"))})
        series = {n: np.array([[num(r, f"{n}_{c}") for c in "rgb"] for r in data]) for n in names}
        bg = series.pop("background", None)
        return t, extra, series, bg

    from vitals_stream import POSE_FIELDS
    with open(prefix + "_pulse.csv") as f:
        header = next(csv.reader(f))
    t, pose, traces, bg = read(prefix + "_pulse.csv", [f"pose_{k}" for k in POSE_FIELDS if f"pose_{k}" in header])
    pulse = (t, traces, bg, None if np.isnan(pose).all() else pose)
    t, emitted, traces, bg = read(prefix + "_flash.csv", ["emit_r", "emit_g", "emit_b"])
    return pulse, (t, emitted, traces, bg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="recordings/liveness_YYYYMMDD_HHMMSS (without _pulse.csv / _flash.csv)")
    args = ap.parse_args()
    (pt, ptr, pbg, ppose), (ft, fem, ftr, fbg) = load_attempt(args.prefix)
    pulse, when = progressive_pulse_check(pt, ptr, pbg, ppose)
    flash = flash_check(ft, fem, ftr, fbg)
    print(verdict(pulse, flash) + (f"   (pulse confirmed after {when:.0f} s)" if when else ""))
    for line in summary_lines(pulse, flash):
        print("  " + line)
    for region, (b, s) in pulse.regions.items():
        print(f"    {region:<12} " + (f"own peak {b:5.1f} BPM, SNR at overall rate {s:+.1f} dB"
                                       if b is not None else "no data"))
    print(f"    flash gain {flash.gain:.3f} (pass needs >= {FLASH_GAIN_MIN})")


if __name__ == "__main__":
    main()
