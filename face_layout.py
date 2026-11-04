"""
face_layout.py -- where on the face we measure and which pixels and motions
count, with no OpenCV import.

Kept free of cv2 so the browser version (web/, running this code in Pyodide)
uses exactly the same regions, thresholds and lighting coach as the desktop
app. face_regions.py and frame_features.py re-export everything here, so
existing imports keep working.
"""

import numpy as np

# Landmark indices outlining each region (convex hull, so order doesn't matter).
REGIONS = {
    "forehead": [10, 338, 297, 332, 334, 296, 336, 9, 107, 66, 105, 103, 67, 109],
    "left cheek": [50, 101, 118, 117, 116, 123, 187, 205, 36],
    "right cheek": [280, 330, 347, 346, 345, 352, 411, 425, 266],
}
# The whole face, for the heart-rate signal: the face outline minus eyes with brows, and lips.
# Measured on real recordings (MCD-rPPG), one big region gives a much cleaner pulse than the
# 16 small patches: on a small face each patch averages few pixels, and its edges move with
# every pixel of landmark jitter. The patches are still measured, for the on-screen SNR map.
FACE_TRACE = "face"
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
             377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
FACE_EXCLUDE = [
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,    # left eye
     70, 63, 105, 66, 107, 55, 65, 52, 53, 46],                                      # + brow
    [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,  # right eye
     300, 293, 334, 296, 336, 285, 295, 282, 283, 276],                               # + brow
    [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],  # lips
]
MIN_FACE_PIXELS = 200     # fewer usable face pixels -> no face colour this frame

# Each region is split into a grid of patches: (columns, rows).
GRID = {"forehead": (4, 2), "left cheek": (2, 2), "right cheek": (2, 2)}

PATCHES = [f"{region.replace(' ', '_')}_p{i}"
           for region, (cols, rows) in GRID.items() for i in range(cols * rows)]
TRACES = PATCHES + [FACE_TRACE]      # every colour trace kept per frame

MIN_PATCH_PIXELS = 25     # fewer usable pixels -> patch counts as missing this frame
DARK_FRACTION = 0.6       # darker than 60% of the region's median brightness -> dropped (hair, brows, shadow)
CLIP_LEVEL = 250          # any channel this bright = saturated sensor (glare) -> dropped

MOTION_LIMIT = 0.025      # nose movement per frame (fraction of face width) that counts as fast motion
SMALL_MOTION = 0.008      # smaller movement: no pause, but auto mode then prefers POS
SMOOTHING = 0.5           # landmark smoothing 0..1 (higher = steadier patches, more lag)

# More than one person in view (frame_features.FaceLock): measure one, follow them.
MAX_FACES = 3              # faces tracked per frame
FACE_LOCK_JUMP = 0.75      # the measured face moving further than this (face widths) in one frame is someone else
FACE_LOCK_SEC = 3.0        # measured person gone this long -> pick again (= live_session.FACE_GAP_SEC)


def patch_region(patch_name):
    """'left_cheek_p3' -> 'left cheek'"""
    return patch_name.rsplit("_p", 1)[0].replace("_", " ")


SHADOW_RATIO = 2.5         # brighter cheek / darker cheek above this: "one side of face in shadow"


def assess_lighting(stats, fps):
    """
    Lighting coach: turn measurements into ONE concrete instruction.
    Returns (level, message), level in {"good", "fair", "none"}.

    Why these checks:
    * Too dark: the camera raises its gain, and sensor noise swamps the pulse.
    * Too much glare: saturated pixels can't change, so they carry no pulse.
    * Uneven: one side in deep shadow -> those patches are mostly noise. Only a deep shadow:
      on validation (40 people) the network was as accurate with one cheek 1.5-2x brighter
      than the other (92% of 15 s windows within 5 BPM) as in even light (95%), so a warning
      at 1.5x mostly told people to fix nothing.
    * Low fps: many webcams slow down in low light to expose longer.
    """
    lum = stats["luminance"]
    if not lum:
        return "none", "Light: no usable skin pixels"
    brightness = float(np.mean(list(lum.values())))
    if brightness < 70:
        return "none", f"Light: too dark ({brightness:.0f}/255) - add front light or press L"
    if fps and fps < 20:
        return "fair", f"Light: camera slowed to {fps:.0f} fps - add light"
    if stats["clipped_fraction"] > 0.05:
        return "fair", "Light: glare on skin - move light or tilt away"
    if "left cheek" in lum and "right cheek" in lum:
        ratio = max(lum["left cheek"], lum["right cheek"]) / max(min(lum["left cheek"], lum["right cheek"]), 1)
        if ratio > SHADOW_RATIO:
            return "fair", "Light: one side of face in shadow - face the light"
    if brightness < 100:
        return "fair", f"Light: dim ({brightness:.0f}/255) - more light helps"
    return "good", f"Light: good ({brightness:.0f}/255)"


# Framing coach. The network was trained on people sitting upright, facing the camera, with
# the whole face in view (MCD-rPPG: face width 11-20% of the frame, head roll under 0.13 rad
# for 99% of people). Outside that, say how to sit instead of only "hold still".
FRAME_OUTSIDE_MAX = 0.10  # share of the face outline outside the picture
FACE_SIZE_MIN = 0.09      # face width / frame width (smaller: too far for enough skin pixels)
ROLL_MAX = 0.30           # head tilt (radians, ~17 degrees)


def assess_framing(framing):
    """
    framing: {"outside", "size", "roll"} from frame_features.face_framing (or features.js).
    Returns a message saying how to sit, or None when the framing is fine.
    """
    if not framing:
        return None
    if framing["outside"] > FRAME_OUTSIDE_MAX:
        return "Face partly out of view - raise the screen or sit back so your whole face is in the picture"
    if framing["size"] < FACE_SIZE_MIN:
        return "Face too small - move closer to the camera"
    if abs(framing["roll"]) > ROLL_MAX:
        return "Head tilted - face the camera straight on"
    return None


def snr_level(snr):
    if snr is None:
        return "none"
    return "good" if snr >= 3 else "fair" if snr >= 0 else "none"


def layout_config():
    """The settings the browser's colour measurement needs (web/features.js), as plain data."""
    keys = ("REGIONS", "GRID", "PATCHES", "MIN_PATCH_PIXELS", "DARK_FRACTION", "CLIP_LEVEL",
            "MOTION_LIMIT", "SMALL_MOTION", "SMOOTHING", "FACE_TRACE", "FACE_OVAL", "FACE_EXCLUDE",
            "MIN_FACE_PIXELS", "MAX_FACES", "FACE_LOCK_JUMP", "FACE_LOCK_SEC")
    return {k: globals()[k] for k in keys}


if __name__ == "__main__":
    # The web page reads these from web/face_layout.json, so it can start measuring before
    # Python has loaded. After changing a setting above, regenerate it:
    #   python face_layout.py > web/face_layout.json
    # (tests/test_web_features.py fails if the two differ.)
    import json
    print(json.dumps(layout_config(), indent=1))
