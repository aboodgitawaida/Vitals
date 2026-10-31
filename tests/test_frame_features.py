"""Per-frame features: pose has 5 numbers, missing face resets motion tracking."""

import types

import numpy as np

from frame_features import FeatureExtractor, head_pose
from vitals_stream import POSE_FIELDS


def _face(shift=0.0, mouth=0.01):
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.35, 0.65, (468, 2))
    pts[234], pts[454], pts[1] = (0.35, 0.5), (0.65, 0.5), (0.5, 0.5)
    pts[33], pts[263] = (0.42, 0.42), (0.58, 0.42)
    pts[13], pts[14] = (0.5, 0.62), (0.5, 0.62 + mouth)
    pts[:, 0] += shift
    return types.SimpleNamespace(landmark=[types.SimpleNamespace(x=x, y=y) for x, y in pts])


def test_head_pose_fields():
    lm = np.array([[p.x * 640, p.y * 360] for p in _face(mouth=0.05).landmark])
    pose = head_pose(lm, 640, 360)
    assert len(pose) == len(POSE_FIELDS)
    assert abs(pose[4] - 0.05 * 360 / (0.3 * 640)) < 1e-6           # mouth gap / face width


def test_motion_detected_and_reset_on_face_loss():
    frame = np.full((360, 640, 3), 150, np.uint8)
    fx = FeatureExtractor()
    assert not fx.process(frame, _face()).fast_motion                 # first frame: nothing to compare
    assert fx.process(frame, _face(shift=0.05)).fast_motion          # jumped 5% of the frame
    assert not fx.process(frame, None).face
    assert fx.process(frame, _face(shift=0.2)).movement == 0.0       # after a gap: no false "motion"
