"""
vitals_dsp.py -- the signal-processing core of "Vitals from Video".

No camera code in here on purpose: functions take arrays of samples and return
numbers, so everything is testable with synthetic data (synthetic_benchmark.py)
and with recorded sessions (replay.py).

THE BIG IDEA (rPPG = remote photoplethysmography)
Each heartbeat pushes extra blood into the skin. Blood absorbs light (green
most strongly), so skin colour changes very slightly with every beat -- far
less than one brightness level out of 255. Average the colour over a patch of
skin each frame and you get a tiny wobble at your heart rate, buried under
much bigger changes from lighting, the camera and movement. Everything below
is about digging that wobble out.

v3 PIPELINE
  for each of 16 small skin patches:
      mean RGB per frame -> even 30 Hz -> remove room-light changes seen on the
      background wall -> normalise -> measure noise above 5 Hz
  inverse-noise-weighted average of all patches -> POS projection
  -> bandpass 0.7-4 Hz -> spectrum -> harmonic-aware peak with tracking
  -> BPM + SNR
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt, detrend

FS = 30.0                     # everything is resampled to 30 samples/second
HR_LOW, HR_HIGH = 0.7, 4.0    # Hz. 42-240 BPM: the range a human heart rate can plausibly be in


# --------------------------------------------------------------------------
# Basic building blocks
# --------------------------------------------------------------------------

def resample_uniform(times, values, fs=FS):
    """
    Webcams don't deliver frames at perfectly even intervals, and our loop
    speed varies too. Filters and FFTs assume evenly spaced samples, so we
    linearly interpolate onto an even time grid first.
    Works for 1-D values (N,) or multi-channel values (N, C).
    """
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    t_u = np.arange(t[0], t[-1], 1.0 / fs)
    if v.ndim == 1:
        return t_u, np.interp(t_u, t, v)
    return t_u, np.column_stack([np.interp(t_u, t, v[:, c]) for c in range(v.shape[1])])


def bandpass(x, low_hz, high_hz, fs=FS, order=3):
    """
    Keep only frequencies between low_hz and high_hz (Butterworth filter).
    Slow lighting drift (below the band) and sensor noise (above it) get removed.
    filtfilt runs the filter forwards then backwards, which cancels the delay a
    normal filter adds. Works along axis 0, so (N,) or (N, C) both work.
    """
    nyquist = fs / 2.0
    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")
    return filtfilt(b, a, x, axis=0)


def power_spectrum(x, fs=FS, nfft=8192):
    """
    FFT = "how much of each frequency is in this signal?"
    * Hann window fades the edges of the window so its hard cut-offs don't
      smear power across all frequencies ("spectral leakage").
    * Zero-padding (nfft > len(x)) gives a finely spaced frequency axis so the
      peak is easy to locate. It does NOT add real resolution: with a T-second
      window, frequencies closer than ~1/T Hz can't be separated
      (10 s -> 6 BPM, 15 s -> 4 BPM).
    """
    windowed = x * np.hanning(len(x))
    power = np.abs(np.fft.rfft(windowed, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freqs, power


# --------------------------------------------------------------------------
# Removing room-light changes using the background as a reference
# --------------------------------------------------------------------------

def remove_background(rgb, background_band):
    """
    Illumination rectification using a non-skin reference
    (idea from Li et al., CVPR 2014).

    The wall behind you has no pulse, but it IS lit by the same room light.
    So anything the wall's colour does in the heart-rate band (lamp flicker
    aliasing with the camera, auto-exposure steps, screen glow) is noise. For
    each skin colour channel we find the best least-squares fit of the
    wall's three channels to the skin channel and subtract it.

    Why it's safe: the pulse is not in the wall, so the fit can only remove
    what the two genuinely share (plus a tiny bit of chance correlation).
    Why POS doesn't already handle this: POS cancels changes that scale all
    channels equally (pure brightness). Lights that change COLOUR (warm/cool
    flicker, a screen showing a video) are not cancelled by POS -- but they
    show up on the wall too.

    rgb:             (N, 3) evenly sampled skin colour
    background_band: (N, 3) wall colour, normalised and band-passed
    """
    mean = rgb.mean(axis=0)
    normalized = rgb / mean - 1.0
    coef, *_ = np.linalg.lstsq(background_band, bandpass(normalized, HR_LOW, HR_HIGH), rcond=None)
    cleaned = normalized - background_band @ coef
    return mean * (1.0 + cleaned)


# --------------------------------------------------------------------------
# Turning RGB into a pulse signal
# --------------------------------------------------------------------------

def green_pulse(rgb):
    """
    Baseline method (Verkruysse et al., 2008): just use the green channel.
    Simple, but ANY change in light intensity shows up in green exactly like
    a pulse would. Kept so you can press "m" and compare against POS live.
    """
    g = rgb[:, 1]
    return detrend(g / g.mean() - 1.0)


def pos_pulse(rgb, fs=FS, win_sec=1.6):
    """
    POS = "Plane-Orthogonal-to-Skin" (Wang, den Brinker, Stuijk & de Haan,
    IEEE Trans. Biomedical Engineering, 2017).

    On short overlapping windows (~1.6 s):
    1. Divide each channel by its own mean. Now "the light got 2% brighter"
       raises R, G and B all by 2%: movement along the direction [1, 1, 1].
    2. Project onto two directions that are both perpendicular to [1, 1, 1]:
       [0, 1, -1] and [-2, 1, 1]. Pure brightness changes vanish. The pulse
       survives because blood changes the channels by DIFFERENT amounts.
    3. Combine the two projections, S1 + alpha * S2 with alpha = std(S1)/std(S2).
       Specular shine from movement appears in the two with opposite sign
       after scaling, so it largely cancels while the pulse adds up.
    Windows are overlap-added into one signal.

    Vectorised: all windows are processed at once with sliding_window_view
    (a view, no copying), so this is fast enough to run on ~16 patches
    twice a second.
    """
    n = len(rgb)
    win = int(round(win_sec * fs))
    windows = sliding_window_view(rgb, win, axis=0)          # (n - win + 1, 3, win)
    normalized = windows / (windows.mean(axis=2, keepdims=True) + 1e-9)   # step 1
    r, g, b = normalized[:, 0], normalized[:, 1], normalized[:, 2]
    s1 = g - b                                                # step 2: [0, 1, -1]
    s2 = -2 * r + g + b                                       #         [-2, 1, 1]
    alpha = s1.std(axis=1) / (s2.std(axis=1) + 1e-9)          # step 3
    p = s1 + alpha[:, None] * s2
    p -= p.mean(axis=1, keepdims=True)
    h = np.zeros(n)
    for k in range(win):                                      # overlap-add
        h[k:k + len(p)] += p[:, k]
    return h


def pulse_signal(rgb_uniform, method="pos", fs=FS):
    """RGB trace (evenly sampled) -> band-limited pulse waveform."""
    raw = pos_pulse(rgb_uniform, fs) if method == "pos" else green_pulse(rgb_uniform)
    return bandpass(raw, HR_LOW, HR_HIGH, fs)


# --------------------------------------------------------------------------
# Reading the heart rate out of the spectrum
# --------------------------------------------------------------------------

def pick_peak(freqs, power, low, high, prior_hz=None, prior_width_hz=0.25, harmonics=False):
    """
    Find the most likely heart-rate frequency in [low, high].

    harmonics: a real pulse wave is not a pure sine (sharp upstroke, slower
    fall), so it also has energy at 2x its frequency. Random noise peaks
    usually don't. Scoring each frequency as P(f) + 0.5 * P(2f) favours
    candidates that have that "fingerprint" -- useful when two peaks are
    nearly equal in height.

    prior_hz (tracking): heart rate can't jump 40 BPM in half a second. If we
    had a confident reading, gently favour peaks near it. The 0.3 floor lets
    a much stronger peak elsewhere still win, so it can't get stuck.
    """
    band = (freqs >= low) & (freqs <= high)
    f, score = freqs[band], power[band].copy()
    if harmonics:
        score = score + 0.5 * np.interp(2 * f, freqs, power)
    if prior_hz is not None:
        score = score * (0.3 + 0.7 * np.exp(-0.5 * ((f - prior_hz) / prior_width_hz) ** 2))
    return float(f[np.argmax(score)])


def snr_db(freqs, power, f0, low, high, width_hz=0.1):
    """
    Signal-to-noise ratio in decibels, in the style of de Haan & Jeanne (2013):
    power near the peak (+/-0.1 Hz) and its 2nd harmonic (+/-0.2 Hz), divided
    by all other power in the heart-rate band.
      > 0 dB : more power in the "pulse" bins than everywhere else combined
      < 0 dB : the peak barely stands out from the noise
    Used to REFUSE to show a number when the signal is junk.
    """
    band = (freqs >= low) & (freqs <= high)
    f, p = freqs[band], power[band]
    is_signal = (np.abs(f - f0) <= width_hz) | (np.abs(f - 2 * f0) <= 2 * width_hz)
    return float(10 * np.log10(p[is_signal].sum() / (p[~is_signal].sum() + 1e-12)))


def _fill_missing(t, rgb, max_missing=0.2):
    """
    Frames where a patch had too few skin pixels are stored as NaN.
    Fill short gaps by interpolation; give up on the patch if >20% is missing.
    """
    valid = ~np.isnan(rgb).any(axis=1)
    if valid.sum() < 10 or valid.mean() < 1.0 - max_missing:
        return None
    if valid.all():
        return rgb
    return np.column_stack([np.interp(t, t[valid], rgb[valid, c]) for c in range(3)])


def noise_level(normalized, fs=FS, cutoff_hz=5.0):
    """
    How noisy is this patch? Measured ABOVE 5 Hz, where no heartbeat can be
    (max ~4 Hz), so whatever is there is camera noise. Camera noise is roughly
    flat across frequencies, so this also tells us how much noise sits inside
    the heart-rate band.

    Key point: this measurement does not look at the pulse peak at all. Scoring
    patches by their own peak SNR and keeping the "best" would favour patches
    whose random noise happens to look like a peak (selection bias).
    """
    b, a = butter(3, cutoff_hz / (fs / 2), btype="high")
    return float(filtfilt(b, a, normalized, axis=0).var(axis=0).mean())


def analyze_heart_rate(times, patch_traces, background=None, method="auto", prev_bpm=None,
                       use_background=True, use_harmonics=True, green_margin_db=0.0, fs=FS):
    """
    Full pipeline for one update.

    times:          (N,) timestamps in seconds
    patch_traces:   {patch name: (N, 3) mean RGB per frame, NaN rows = missing}
    background:     (N, 3) mean RGB of the background wall, or None
    method:         "auto" (try both, keep cleaner), "pos" or "green"
    prev_bpm:       last confident reading, for tracking (or None)
    green_margin_db: in auto mode, green must beat POS by this many dB to be
                    chosen. The app raises it to 3 dB when the head moved during
                    the window, because motion creates clean-looking false peaks
                    in green that POS cancels.

    Why many small patches instead of 3 big regions? With light from one
    side, part of a cheek is lit and part is in shadow. Shadowed skin
    reflects less light, so relative to its brightness it is much noisier.
    A big region averages good skin and bad skin equally.

    How patches are combined (inverse-variance weighting):
      1. Normalise each patch's colour by its own mean (so a dark patch and a
         bright patch are on the same scale).
      2. Measure each patch's noise above 5 Hz (noise_level).
      3. Average the normalised colours with weight = 1 / noise variance.
         This is the statistically optimal way to average independent noisy
         measurements of the same thing: quiet patches count more.
      4. Run POS ONCE on that pooled colour signal.
    (An earlier version ran POS per patch and kept the top patches by SNR.
    The synthetic ablation showed that was worse -- see synthetic_benchmark.py.)

    Per-patch SNR is still computed, but only to draw the SNR map.
    Returns a dict, or None if no patch had usable data.
    """
    t = np.asarray(times, dtype=float)

    background_band = None
    if use_background and background is not None:
        bg = _fill_missing(t, np.asarray(background, dtype=float))
        if bg is not None:
            _, bg_u = resample_uniform(t, bg, fs)
            background_band = bandpass(bg_u / bg_u.mean(axis=0) - 1.0, HR_LOW, HR_HIGH, fs)

    pooled, total_weight, patch_snr, patch_weight = None, 0.0, {}, {}
    for name, rgb in patch_traces.items():
        rgb = _fill_missing(t, np.asarray(rgb, dtype=float))
        if rgb is None:
            patch_snr[name] = None
            continue
        _, rgb_u = resample_uniform(t, rgb, fs)
        if background_band is not None:
            rgb_u = remove_background(rgb_u, background_band)
        normalized = rgb_u / rgb_u.mean(axis=0)                # step 1
        weight = 1.0 / (noise_level(normalized, fs) + 1e-12)   # step 2
        pooled = weight * normalized if pooled is None else pooled + weight * normalized   # step 3
        total_weight += weight
        patch_weight[name] = weight

        # Per-patch SNR, for the on-screen SNR map only
        map_method = "green" if method == "auto" else method
        f, p = power_spectrum(pulse_signal(rgb_u, map_method, fs), fs)
        patch_snr[name] = snr_db(f, p, pick_peak(f, p, HR_LOW, HR_HIGH), HR_LOW, HR_HIGH)

    if pooled is None:
        return None

    # Step 4. In "auto" mode, try both methods and keep the cleaner spectrum.
    # Why: POS cancels flickering / changing light, but it leans on the blue
    # channel, which skin reflects weakly -- in a dim room blue is mostly sensor
    # noise, and POS mixes that noise in. Green carries the strongest pulse.
    # So: steady but dim light -> green tends to win; flicker/motion -> POS wins.
    # The synthetic ablation showed each method failing where the other succeeds.
    prior_hz = prev_bpm / 60.0 if prev_bpm else None
    candidates = []
    for m in (["pos", "green"] if method == "auto" else [method]):
        fused = pulse_signal(pooled / total_weight, m, fs)
        fused = fused / (fused.std() + 1e-12)
        freqs, power = power_spectrum(fused, fs)
        f0 = pick_peak(freqs, power, HR_LOW, HR_HIGH, prior_hz, harmonics=use_harmonics)
        snr = snr_db(freqs, power, f0, HR_LOW, HR_HIGH)
        bias = -green_margin_db if m == "green" else 0.0      # POS is the default; green must earn it
        candidates.append((snr + bias, snr, m, f0, fused, freqs, power))
    _, snr, used, f0, fused, freqs, power = max(candidates, key=lambda c: c[0])

    return {
        "bpm": f0 * 60.0,
        "snr_db": snr,
        "method_used": used,
        "pulse": fused,
        "freqs": freqs,
        "power": power,
        "patch_snr": patch_snr,
        "patch_weight": {n: w / total_weight for n, w in patch_weight.items()},
        "background_used": background_band is not None,
    }


# --------------------------------------------------------------------------
# v1 helper, kept so vitals_phase1.py still runs as a baseline.
# Phase 2 (breathing) will reuse this with a 0.1-0.5 Hz band.
# --------------------------------------------------------------------------

def estimate_rate(times, values, low_hz, high_hz, fs=FS, order=3):
    """Single 1-D signal -> (rate per minute, quality 0..1, filtered signal)."""
    _, v_u = resample_uniform(times, values, fs)
    filtered = bandpass(detrend(v_u), low_hz, high_hz, fs, order)
    freqs, power = power_spectrum(filtered, fs)
    f0 = pick_peak(freqs, power, low_hz, high_hz)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    f_band, p_band = freqs[band], power[band]
    near = np.abs(f_band - f0) < 0.15
    if 2 * f0 <= high_hz:
        near |= np.abs(f_band - 2 * f0) < 0.15
    return f0 * 60.0, float(p_band[near].sum() / p_band.sum()), filtered
