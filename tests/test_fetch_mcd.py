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


def _meta(fps, n, start="2023-11-30 15:43:29.600000"):
    from datetime import datetime, timedelta
    t0 = datetime.fromisoformat(start)
    return "\n" + "\n".join(f"{i + 1}  {t0 + timedelta(seconds=i / fps)}" for i in range(n)) + "\n"


def test_true_fps_from_frame_timestamps():
    from fetch_mcd_rppg import true_fps
    assert abs(true_fps(_meta(29.97, 900)) - 29.97) < 0.01
    assert abs(true_fps(_meta(24.0, 600)) - 24.0) < 0.01


def test_retime_fixes_a_wrong_header_rate(tmp_path, monkeypatch):
    """A 30 fps video whose header said 24: features and reference move to the true times together."""
    import build_mcd_dataset as b
    from fetch_mcd_rppg import read_fps
    n, bpm = 30 * 40, 72.0
    folder = tmp_path / "1234_before"
    folder.mkdir()
    t_true = np.arange(n) / 30.0
    np.savez_compressed(folder / "features.npz", t=np.arange(n) / 24.0, face=np.ones(n, bool),
                        moving=np.zeros(n, bool), background=np.zeros((n, 3)), pose=np.zeros((n, 5)))
    sync = "\n".join(f"{500 + 40 * np.sin(2 * np.pi * bpm / 60 * x):.1f} 0.001" for x in t_true)
    files = {"meta/x.txt": _meta(30.0, n + 20), "ppg_sync/x.txt": sync}
    monkeypatch.setattr(b, "fetch_text", files.__getitem__)
    row = {"meta": "meta/x.txt", "ppg_sync": "ppg_sync/x.txt", "pulse": "70"}
    name, status = b.retime((row, str(folder)))
    assert "true 30.00" in status, status
    assert abs(read_fps(str(folder)) - 30.0) < 0.05
    f = b.load_features(str(folder))
    assert abs(f["t"][-1] - t_true[-1]) < 0.05
    times, wave, _ = read_ground_truth(folder / "ground_truth.txt")
    assert abs(reference_bpm(times, wave, 30.0) - bpm) < 2.0
    assert b.retime((row, str(folder)))[1] == "done already"


def test_split_folders_skips_missing_and_faceless(tmp_path):
    import json
    from build_mcd_dataset import split_folders
    (tmp_path / "splits.json").write_text(json.dumps({"train": ["1", "2", "3"], "validation": []}))
    for name, face in (("1_before", 1.0), ("1_after", 0.9), ("2_before", 0.2), ("3_after", 0.6)):
        (tmp_path / name).mkdir()
        f = np.zeros(100, bool)
        f[:int(face * 100)] = True
        np.savez_compressed(tmp_path / name / "features.npz", face=f)
    got = [p.rsplit("/", 1)[1] for p in split_folders(str(tmp_path), "train")]
    assert got == ["1_before", "1_after", "3_after"]
