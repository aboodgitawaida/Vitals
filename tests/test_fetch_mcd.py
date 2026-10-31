"""fetch_mcd_rppg: subject sampling and conversion to the ground-truth layout evaluate_video reads (no network)."""

import numpy as np

from evaluate_video import read_ground_truth, reference_bpm
from fetch_mcd_rppg import FPS, choose, write_ground_truth


def _rows():
    rows = []
    for pid in range(1000, 1040):
        for step in ("before", "after"):
            for cam in ("FullHDwebcam", "USBVideo"):
                rows.append({"patient_id": str(pid), "step": step, "camera": cam})
    rows = [r for r in rows if not (r["patient_id"] == "1001" and r["step"] == "after")]   # incomplete pair
    return rows


def test_choose_is_reproducible_and_complete():
    a, b = choose(_rows(), 10, seed=0), choose(_rows(), 10, seed=0)
    assert a == b
    assert len(a) == 20
    assert all(r["camera"] == "FullHDwebcam" for r in a)
    pids = {r["patient_id"] for r in a}
    assert len(pids) == 10 and "1001" not in pids
    for p in pids:
        assert {r["step"] for r in a if r["patient_id"] == p} == {"before", "after"}
    assert choose(_rows(), 10, seed=1) != a


def test_ground_truth_round_trip(tmp_path):
    bpm = 84.0
    t = np.arange(int(40 * FPS)) / FPS
    ppg = 500 + 40 * np.sin(2 * np.pi * bpm / 60 * t)
    sync = "\n".join(f"{v:.0f} 0.001" for v in ppg)
    path = tmp_path / "ground_truth.txt"
    write_ground_truth(sync, 83.0, path)
    times, wave, hr = read_ground_truth(path)
    assert len(times) == len(wave) == len(hr) == len(t)
    assert abs(times[-1] - t[-1]) < 1e-3
    assert abs(reference_bpm(times, wave, 30.0) - bpm) < 2.0
