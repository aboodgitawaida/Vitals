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
# Each region is split into a grid of patches: (columns, rows).
GRID = {"forehead": (4, 2), "left cheek": (2, 2), "right cheek": (2, 2)}

PATCHES = [f"{region.replace(' ', '_')}_p{i}"
           for region, (cols, rows) in GRID.items() for i in range(cols * rows)]

MIN_PATCH_PIXELS = 25     # fewer usable pixels -> patch counts as missing this frame
DARK_FRACTION = 0.6       # darker than 60% of the region's median brightness -> dropped (hair, brows, shadow)
CLIP_LEVEL = 250          # any channel this bright = saturated sensor (glare) -> dropped

MOTION_LIMIT = 0.025      # nose movement per frame (fraction of face width) that counts as fast motion
SMALL_MOTION = 0.008      # smaller movement: no pause, but auto mode then prefers POS
SMOOTHING = 0.5           # landmark smoothing 0..1 (higher = steadier patches, more lag)


def patch_region(patch_name):
    """'left_cheek_p3' -> 'left cheek'"""
    return patch_name.rsplit("_p", 1)[0].replace("_", " ")


def assess_lighting(stats, fps):
    """
    Lighting coach: turn measurements into ONE concrete instruction.
    Returns (level, message), level in {"good", "fair", "none"}.

    Why these checks:
    * Too dark: the camera raises its gain, and sensor noise swamps the pulse.
    * Too much glare: saturated pixels can't change, so they carry no pulse.
    * Uneven: one side in shadow -> those patches are mostly noise.
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
        if ratio > 1.5:
            return "fair", "Light: one side of face in shadow - face the light"
    if brightness < 100:
        return "fair", f"Light: dim ({brightness:.0f}/255) - more light helps"
    return "good", f"Light: good ({brightness:.0f}/255)"


def snr_level(snr):
    if snr is None:
        return "none"
    return "good" if snr >= 3 else "fair" if snr >= 0 else "none"
