"""
signal_lite.py -- the three scipy.signal functions vitals_dsp uses, in numpy only.

Why: scipy is ~14 MB to download in the browser (Pyodide), about half of what
the web version loads, and vitals_dsp needs only butter, filtfilt and detrend.
vitals_dsp imports scipy when it is installed (desktop, tests) and falls back
to this file when it isn't (the browser). tests/test_signal_lite.py checks
that both give the same numbers.

Same defaults as scipy: butter(N, Wn, btype) -> (b, a) for a digital filter;
filtfilt with odd padding of 3 * max(len(a), len(b)) samples and lfilter_zi
initial conditions; detrend removes a least-squares line.
"""

from collections import OrderedDict

import numpy as np


# ---------------------------------------------------------------- butter

def _zpk_to_ba(z, p, k):
    b = np.real(k * np.poly(z)) if len(z) else np.array([np.real(k)])
    return b, np.real(np.poly(p))


def butter(N, Wn, btype="low"):
    """Digital Butterworth filter, Wn as a fraction of Nyquist (scipy.signal.butter, output='ba')."""
    btype = {"band": "bandpass", "high": "highpass", "low": "lowpass"}.get(btype, btype)
    Wn = np.atleast_1d(np.asarray(Wn, dtype=float))
    # Analogue prototype: poles evenly spaced on the left half of the unit circle.
    p = -np.exp(1j * np.pi * np.arange(-N + 1, N, 2) / (2 * N))
    z, k = np.array([], dtype=complex), 1.0
    fs = 2.0
    warped = 2 * fs * np.tan(np.pi * Wn / fs)            # pre-warp for the bilinear transform
    degree = len(p) - len(z)
    if btype == "lowpass":
        wo = warped[0]
        z, p, k = z * wo, p * wo, k * wo ** degree
    elif btype == "highpass":
        wo = warped[0]
        k = k * np.real(np.prod(-z) / np.prod(-p))
        z, p = np.append(wo / z, np.zeros(degree)), wo / p
    elif btype == "bandpass":
        bw, wo = warped[1] - warped[0], np.sqrt(warped[0] * warped[1])
        z_lp, p_lp = z * bw / 2, p * bw / 2
        z = np.concatenate([z_lp + np.sqrt(z_lp ** 2 - wo ** 2), z_lp - np.sqrt(z_lp ** 2 - wo ** 2),
                            np.zeros(degree)])
        p = np.concatenate([p_lp + np.sqrt(p_lp ** 2 - wo ** 2), p_lp - np.sqrt(p_lp ** 2 - wo ** 2)])
        k = k * bw ** degree
    else:
        raise ValueError(f"unsupported btype {btype!r}")
    # Bilinear transform to the digital domain.
    fs2 = 2 * fs
    degree = len(p) - len(z)
    k = k * np.real(np.prod(fs2 - z) / np.prod(fs2 - p))
    z = np.append((fs2 + z) / (fs2 - z), -np.ones(degree))
    p = (fs2 + p) / (fs2 - p)
    return _zpk_to_ba(z, p, k)


# ---------------------------------------------------------------- filtfilt

def _lfilter_zi(b, a):
    """Initial state for a step response that starts in steady state (scipy.signal.lfilter_zi)."""
    n = max(len(a), len(b))
    a = np.pad(a, (0, n - len(a)))
    b = np.pad(b, (0, n - len(b)))
    companion = np.zeros((n - 1, n - 1))
    companion[0] = -a[1:]
    companion[1:, :-1] = np.eye(n - 2)
    return np.linalg.solve(np.eye(n - 1) - companion.T, b[1:] - a[1:] * b[0])


def _lfilter(b, a, x, zi):
    """Direct form II transposed along axis 0. x: (n, m), zi: (order, m)."""
    order = len(zi)
    z = zi.copy()
    y = np.empty_like(x)
    for i in range(len(x)):
        xi = x[i]
        yi = b[0] * xi + z[0]
        for j in range(order - 1):
            z[j] = b[j + 1] * xi + z[j + 1] - a[j + 1] * yi
        z[order - 1] = b[order] * xi - a[order] * yi
        y[i] = yi
    return y


_OPERATORS = OrderedDict()   # (n, b, a) -> n x n matrix; a few window lengths recur constantly


def _filtfilt_matrix(b, a, n):
    """
    filtfilt is linear in its input, so for a given filter and length it is
    one n x n matrix. Build it once (filter the identity matrix: all columns
    at once) and every later call is a matrix product. A sample-by-sample
    loop in Python on each call would be far too slow in the browser.
    """
    key = (n, b.tobytes(), a.tobytes())
    m = _OPERATORS.get(key)
    if m is not None:
        _OPERATORS.move_to_end(key)
        return m
    pad = 3 * max(len(a), len(b))
    if n <= pad:
        raise ValueError(f"The length of the input vector x must be greater than padlen, which is {pad}.")
    x = np.eye(n)
    ext = np.concatenate([2 * x[0] - x[pad:0:-1], x, 2 * x[-1] - x[-2:-pad - 2:-1]])   # odd extension
    zi = _lfilter_zi(b, a)[:, None]
    y = _lfilter(b, a, ext, zi * ext[0])
    y = _lfilter(b, a, y[::-1], zi * y[-1])[::-1]
    m = y[pad:-pad]
    _OPERATORS[key] = m
    if len(_OPERATORS) > 16:
        _OPERATORS.popitem(last=False)
    return m


def filtfilt(b, a, x, axis=-1):
    """Zero-phase filtering, forwards then backwards (scipy.signal.filtfilt defaults)."""
    b = np.asarray(b, dtype=float) / a[0]
    a = np.asarray(a, dtype=float) / a[0]
    x = np.moveaxis(np.asarray(x, dtype=float), axis, 0)
    m = _filtfilt_matrix(b, a, x.shape[0])
    y = m @ x.reshape(x.shape[0], -1)
    return np.moveaxis(y.reshape(x.shape), 0, axis)


# ---------------------------------------------------------------- detrend

def detrend(data, axis=-1):
    """Remove the least-squares straight line (scipy.signal.detrend, type='linear')."""
    x = np.moveaxis(np.asarray(data, dtype=float), axis, 0)
    n = x.shape[0]
    t = np.column_stack([np.arange(n, dtype=float), np.ones(n)])
    flat = x.reshape(n, -1)
    coef, *_ = np.linalg.lstsq(t, flat, rcond=None)
    return np.moveaxis((flat - t @ coef).reshape(x.shape), 0, axis)
