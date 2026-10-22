"""make_test_video -> evaluate_video end to end, with a stand-in for FaceMesh (MediaPipe not needed)."""

import types

import cv2
import numpy as np

from evaluate_video import (evaluate_subject, extract_features, find_subjects, metrics, read_ground_truth,
                            reference_bpm)
from face_regions import REGIONS
from make_test_video import make_video


class FakeMesh:
    def __init__(self):
        tpl = np.random.default_rng(0).uniform(0.4, 0.6, (468, 2))
        for idx, (cx, cy) in zip(REGIONS.values(), [(0.5, 0.3), (0.43, 0.55), (0.57, 0.55)]):
            for k, i in enumerate(idx):
                a = 2 * np.pi * k / len(idx)
                tpl[i] = (cx + 0.05 * np.cos(a), cy + 0.03 * np.sin(a))
        tpl[234], tpl[454], tpl[1] = (0.36, 0.5), (0.64, 0.5), (0.5, 0.5)
        tpl[33], tpl[263], tpl[13], tpl[14] = (0.44, 0.42), (0.56, 0.42), (0.5, 0.68), (0.5, 0.685)
        self.tpl = tpl

    def process(self, img):
        lm = [types.SimpleNamespace(x=x, y=y) for x, y in self.tpl]
        return types.SimpleNamespace(multi_face_landmarks=[types.SimpleNamespace(landmark=lm)])


def test_known_bpm_video_is_recovered(tmp_path):
    img = np.full((360, 640, 3), (175, 180, 185), np.uint8)
    cv2.ellipse(img, (320, 185), (109, 140), 0, 0, 360, (95, 125, 185), -1)
    out = make_video(img, str(tmp_path / "subj"), bpm=84, seconds=22, size=(640, 360),
                     box=(0.33, 0.12, 0.34, 0.78))
    (name, video, gt), = find_subjects([out])
    gt_t, gt_ppg, gt_hr = read_ground_truth(gt)
    assert abs(np.mean(gt_hr) - 84) < 3
    feats = extract_features(video, face_mesh=FakeMesh(), cache_dir=None)
    assert feats["face"].all() and len(feats["t"]) == 22 * 30
    res = evaluate_subject(feats, lambda te: reference_bpm(gt_t, gt_ppg, te))
    assert metrics([(e, r) for _, e, r in res["windows"]["full pipeline (auto)"]])["MAE"] < 2.0
    shown = [(e, r) for _, e, r in res["screen"] if e is not None]
    assert shown and metrics(shown)["MAE"] < 2.0
