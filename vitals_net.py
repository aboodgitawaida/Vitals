"""
vitals_net.py -- a small trained network that turns per-frame skin colours into a pulse and a heart rate.

Why a network: on real recordings the hand-written pipeline (whole-face POS) was
often confidently wrong. Strong periodic components in the video locked the
estimate, and no hand-written check (SNR, agreement between methods or face
regions, stability) told those cases apart. Telling them apart is pattern
recognition, which a network can learn from examples (MCD-rPPG: hundreds of
people with a finger PPG sensor).

Input: the colours the app already measures every frame (whole face, 16 skin
patches, the background wall), i.e. a "spatio-temporal map" in rPPG terms
(as in RhythmNet), plus the POS, CHROM and green signals of the face as
hand-made hints. Output:
  * a pulse waveform (for the dashboard), and
  * a probability for every heart rate from 40 to 180 BPM. Its peak is the
    estimate; the probability within 5 BPM of the peak is the confidence.

Network, 74k weights (hidden width 48):
  1x1 colour mixing (a learned, many-channel POS) -> 5 residual blocks of dilated
  temporal convolutions -> waveform head (1x1)
                        -> spectral head: power spectrum of every feature channel,
                           mapped onto the 40-180 BPM grid, mixed across channels
                           and neighbouring bins -> softmax

This file is numpy only, so it runs unchanged in the browser (Pyodide).
train_net.py trains the same network in PyTorch and exports the weights to
vitals_net_weights.npz; tests/test_net.py checks that both give the same output.
The input preparation below is used by training too, so the two cannot drift.
"""

import json
import os

import numpy as np

from face_layout import FACE_TRACE, PATCHES
from vitals_dsp import FS, HR_HIGH, HR_LOW, _fill_missing, bandpass, chrom_pulse, green_pulse, pos_pulse, resample_uniform

TRACES = [FACE_TRACE] + PATCHES + ["background"]   # 18 colour traces, in this order
N_TRACES = len(TRACES)
N_IN = 3 * N_TRACES + 3 + N_TRACES                # RGB per trace, 3 hint signals, 1 "present" flag per trace
BPM_BINS = np.arange(40.0, 181.0)                 # output grid, 1 BPM apart
NFFT = 2048                                       # spectral head: 0.88 BPM per FFT bin at 30 Hz
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vitals_net_weights.npz")
MIN_WINDOW_SEC = 6.0                              # shortest window the network was trained on


# --------------------------------------------------------------------------
# Input preparation (shared with training)
# --------------------------------------------------------------------------

def window_input(rgb):
    """
    rgb: list of N_TRACES arrays (L, 3), evenly sampled at FS, or None where a trace is missing.
    Returns the network input, (N_IN, L) float32.

    Every colour is turned into relative change (divide by its own mean) and
    band-passed to the heart-rate band, so the network sees what the hand-made
    methods see. All colour channels share ONE scale (the face's level), so the
    ratios between R, G and B, which carry the pulse, survive.
    """
    L = next(len(x) for x in rgb if x is not None)
    rel = np.zeros((N_TRACES, 3, L))
    present = np.zeros(N_TRACES)
    for i, x in enumerate(rgb):
        if x is None or (x.mean(axis=0) < 2.0).any():
            continue
        rel[i] = bandpass(x / x.mean(axis=0) - 1.0, HR_LOW, HR_HIGH).T
        present[i] = 1.0
    face = rgb[0] if present[0] else next((x for x, p in zip(rgb, present) if p and x is not None), None)
    scale = np.median(rel[0].std(axis=1)) if present[0] else np.median(rel[present > 0].std(axis=2))
    rel = np.clip(rel / (scale + 1e-9), -8.0, 8.0)
    hints = np.zeros((3, L))
    if face is not None:
        for k, fn in enumerate((pos_pulse, chrom_pulse, green_pulse)):
            h = bandpass(fn(face), HR_LOW, HR_HIGH)
            hints[k] = np.clip(h / (h.std() + 1e-9), -8.0, 8.0)
    flags = np.repeat(present[:, None], L, axis=1)
    return np.concatenate([rel.reshape(3 * N_TRACES, L), hints, flags]).astype(np.float32)


def prepare(times, patch_traces, background, fs=FS):
    """App data (frame times, {trace: (N,3)}, wall (N,3)) -> (network input, L) or None if no face colour."""
    t = np.asarray(times, dtype=float)
    rgb = []
    for name in TRACES:
        raw = background if name == "background" else patch_traces.get(name)
        x = None if raw is None else _fill_missing(t, np.asarray(raw, dtype=float))
        rgb.append(None if x is None else resample_uniform(t, x, fs)[1])
    if rgb[0] is None:
        return None
    return window_input(rgb)


# --------------------------------------------------------------------------
# Forward pass (numpy)
# --------------------------------------------------------------------------

def _conv(x, w, b, dilation=1):
    """Same-length 1-D convolution. x (C_in, L), w (C_out, C_in, K), b (C_out,)."""
    k = w.shape[2]
    pad = dilation * (k - 1) // 2
    xp = np.pad(x, ((0, 0), (pad, pad)))
    L = x.shape[1]
    out = b[:, None] + sum(w[:, :, j] @ xp[:, j * dilation:j * dilation + L] for j in range(k))
    return out


def bin_matrix(fs=FS):
    """(NFFT//2+1, len(BPM_BINS)) linear interpolation from FFT bins onto the BPM grid."""
    freqs_bpm = np.fft.rfftfreq(NFFT, d=1.0 / fs) * 60.0
    m = np.zeros((len(freqs_bpm), len(BPM_BINS)))
    for j, b in enumerate(BPM_BINS):
        i = np.searchsorted(freqs_bpm, b) - 1
        frac = (b - freqs_bpm[i]) / (freqs_bpm[i + 1] - freqs_bpm[i])
        m[i, j], m[i + 1, j] = 1.0 - frac, frac
    return m


_BINS = bin_matrix()


def spectrum_to_bins(feat):
    """(C, L) features -> (C, len(BPM_BINS)) log power on the BPM grid (Hann window, zero-padded FFT)."""
    L = feat.shape[1]
    power = np.abs(np.fft.rfft(feat * np.hanning(L), n=NFFT, axis=1)) ** 2
    binned = power @ _BINS
    return np.log1p(binned / (binned.mean(axis=1, keepdims=True) + 1e-9))


class PulseNet:
    """The trained network. Weights and layout come from vitals_net_weights.npz."""

    def __init__(self, path=WEIGHTS):
        d = np.load(path)
        self.w = {k: d[k].astype(np.float64) for k in d.files if k != "meta"}
        self.meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
        self.dilations = [int(x) for x in self.w["dilations"]]

    def forward(self, x):
        """x (N_IN, L) -> (waveform (L,), logits over BPM_BINS)."""
        w = self.w
        h = np.maximum(_conv(x, w["mix.w"], w["mix.b"]), 0.0)
        for i, d in enumerate(self.dilations):
            r = np.maximum(_conv(h, w[f"block{i}.w"], w[f"block{i}.b"], d), 0.0)
            h = h + _conv(r, w[f"block{i}.w2"], w[f"block{i}.b2"])
        wave = _conv(h, w["wave.w"], w["wave.b"])[0]
        s = spectrum_to_bins(h)
        s = np.maximum(_conv(s, w["spec.w1"], w["spec.b1"]), 0.0)            # mix channels, per bin
        logits = _conv(s, w["spec.w2"], w["spec.b2"])[0]                     # neighbouring bins
        return wave, logits


_net = None


def estimate(x):
    """
    Network input -> dict(bpm, confidence, posterior, pulse).
    confidence: probability that the heart rate is within 5 BPM of bpm (the network's own estimate).
    """
    global _net
    if _net is None:
        _net = PulseNet()
    wave, logits = _net.forward(x.astype(np.float64))
    p = np.exp(logits - logits.max())
    p /= p.sum()
    i = int(np.argmax(p))
    bpm = BPM_BINS[i]
    if 0 < i < len(p) - 1:                                        # parabolic interpolation between bins
        a, b, c = np.log(p[i - 1] + 1e-12), np.log(p[i] + 1e-12), np.log(p[i + 1] + 1e-12)
        denom = a - 2 * b + c
        if denom < 0:
            bpm += 0.5 * (a - c) / denom
    confidence = float(p[np.abs(BPM_BINS - bpm) <= 5.0].sum())
    return {"bpm": float(bpm), "confidence": confidence, "posterior": p, "pulse": wave}


def available():
    return os.path.exists(WEIGHTS)
