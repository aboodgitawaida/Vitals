"""
perf.py -- small tools for keeping the live app fast.

StageTimer          measures how long each step of a frame takes, so the
                    on-screen readout can show where the time goes.
BackgroundAnalyzer  runs the heart-rate analysis in a separate process, so the
                    camera loop never stops to wait for it.

Why a separate PROCESS and not a thread: Python runs only one thread of Python
code at a time (the GIL), so a thread would still take turns with the camera
loop. A second process has its own interpreter and runs truly in parallel on
another CPU core. The cost is copying the inputs across (~200 KB every 0.5 s),
which is tiny.
"""

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

from vitals_dsp import analyze_heart_rate


class StageTimer:
    """Smoothed milliseconds per named stage (exponential moving average)."""

    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self.ms = {}

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - start) * 1000)

    def add(self, name, ms):
        old = self.ms.get(name)
        self.ms[name] = ms if old is None else self.smoothing * old + (1 - self.smoothing) * ms

    def get(self, name):
        return self.ms.get(name)


def _run(job):
    """Runs inside the worker process. Returns (job id, analysis, milliseconds)."""
    start = time.perf_counter()
    result = analyze_heart_rate(**job["kwargs"])
    return job["id"], result, (time.perf_counter() - start) * 1000


class BackgroundAnalyzer:
    """
    At most one analysis in flight. The app calls submit() when it wants a new
    estimate and poll() every frame; if the worker is still busy, submit()
    returns False and the app simply tries again next frame.

    Each job carries an id. invalidate() (called when the user resets, changes
    method, toggles light mode...) makes any result still in flight be thrown
    away, because it was computed from data the app no longer trusts.

    background=False runs everything inline (same results, no parallelism),
    for debugging and for comparing speed.
    """

    def __init__(self, background=True):
        self.background = background
        # "spawn" = start a fresh Python for the worker. It is the macOS default anyway;
        # setting it explicitly makes Linux (CI) behave the same way.
        self.pool = (ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
                     if background else None)
        self.future = None
        self.pending_meta = None
        self.inline_result = None
        self.next_id = 0
        self.valid_from = 0

    def busy(self):
        return self.future is not None and not self.future.done()

    def submit(self, meta, **kwargs):
        """meta: anything the app wants back with the result (e.g. the time of the snapshot)."""
        if self.future is not None or self.inline_result is not None:
            return False                     # previous job running, or finished but not collected yet
        job = {"id": self.next_id, "kwargs": kwargs}
        self.next_id += 1
        self.pending_meta = meta
        if self.background:
            self.future = self.pool.submit(_run, job)
        else:
            self.inline_result = _run(job)
        return True

    def poll(self):
        """Returns (meta, analysis, ms) once a job has finished, else None."""
        if self.inline_result is not None:
            done, self.inline_result = self.inline_result, None
        elif self.future is not None and self.future.done():
            done, self.future = self.future.result(), None
        else:
            return None
        job_id, result, ms = done
        if job_id < self.valid_from:
            return None                      # computed before a reset: discard
        return self.pending_meta, result, ms

    def invalidate(self):
        self.valid_from = self.next_id

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
