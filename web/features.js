// features.js -- browser port of face_regions.extract_patch_colors / face_color / background_color
// and frame_features.FeatureExtractor: everything measured from ONE video frame.
//
// It needs the camera's pixels, so it runs in the page, not in Python. The regions,
// grid and thresholds are not copied here: they come from face_layout.py at startup
// (see worker.js), so the browser and desktop versions measure the same pixels.
// tests/test_web_features.py checks this file against the Python code.
//
// Frames are RGBA ({data, width, height}, like ImageData); landmarks are [[x, y], ...]
// in pixels. Colours are returned as [r, g, b] like the Python code.

function convexHull(points) {
  // Andrew's monotone chain on integer points.
  const pts = points.map((p) => [p[0], p[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length < 3) return pts;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

function fillConvex(mask, w, h, hull, value = 1) {
  // Scanline fill of a convex polygon, boundary included (like cv2.fillConvexPoly).
  if (hull.length === 0) return;
  let y0 = Infinity, y1 = -Infinity;
  for (const p of hull) { y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
  for (let y = Math.max(y0, 0); y <= Math.min(y1, h - 1); y++) {
    let xl = Infinity, xr = -Infinity;
    for (let i = 0; i < hull.length; i++) {
      const a = hull[i], b = hull[(i + 1) % hull.length];
      if ((a[1] <= y && y <= b[1]) || (b[1] <= y && y <= a[1])) {
        if (a[1] === b[1]) {
          xl = Math.min(xl, a[0], b[0]); xr = Math.max(xr, a[0], b[0]);
        } else {
          const x = a[0] + ((y - a[1]) * (b[0] - a[0])) / (b[1] - a[1]);
          xl = Math.min(xl, x); xr = Math.max(xr, x);
        }
      }
    }
    if (xl > xr) continue;
    const from = Math.max(Math.round(xl), 0), to = Math.min(Math.round(xr), w - 1);
    for (let x = from; x <= to; x++) mask[y * w + x] = value;
  }
}

function regionCrop(lm, indices, W, H) {
  // Bounding box of the region plus its mask inside that box (face_regions.region_crop).
  const pts = indices.map((i) => [Math.trunc(lm[i][0]), Math.trunc(lm[i][1])]);
  let x = Infinity, y = Infinity, xm = -Infinity, ym = -Infinity;
  for (const [px, py] of pts) { x = Math.min(x, px); y = Math.min(y, py); xm = Math.max(xm, px); ym = Math.max(ym, py); }
  let w = xm - x + 1, h = ym - y + 1;
  x = Math.max(x, 0); y = Math.max(y, 0);
  w = Math.max(Math.min(w, W - x), 1); h = Math.max(Math.min(h, H - y), 1);
  const mask = new Uint8Array(w * h);
  fillConvex(mask, w, h, convexHull(pts.map(([px, py]) => [px - x, py - y])));
  return { x, y, w, h, mask };
}

function median(values) {
  const v = Float32Array.from(values).sort();
  const n = v.length, m = n >> 1;
  return n % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

export function extractPatchColors(frame, lm, cfg) {
  // Mean RGB of the usable skin in every patch (face_regions.extract_patch_colors).
  const { data, width: W, height: H } = frame;
  const colors = {}, patches = {}, regions = {};
  const luminance = {};
  let total = 0, clippedTotal = 0;
  for (const [name, indices] of Object.entries(cfg.REGIONS)) {
    const { x, y, w, h, mask } = regionCrop(lm, indices, W, H);
    // The crop may be cut off at the frame's right/bottom edge (as numpy slicing does).
    const cw = Math.min(w, W - x), ch = Math.min(h, H - y);
    // skin_filter: drop pixels darker than DARK_FRACTION of the median, and glare.
    const idx = [], lum = [];
    for (let r = 0; r < ch; r++) {
      for (let c = 0; c < cw; c++) {
        if (!mask[r * w + c]) continue;
        const o = ((y + r) * W + (x + c)) * 4;
        idx.push(r * w + c);
        lum.push(0.114 * data[o + 2] + 0.587 * data[o + 1] + 0.299 * data[o]);
      }
    }
    for (let k = 0; k < mask.length; k++) total += mask[k];
    const kept = new Uint8Array(w * h);
    if (idx.length) {
      const limit = Math.fround(cfg.DARK_FRACTION * median(lum));
      for (let k = 0; k < idx.length; k++) {
        const r = Math.floor(idx[k] / w), c = idx[k] % w;
        const o = ((y + r) * W + (x + c)) * 4;
        const clipped = Math.max(data[o], data[o + 1], data[o + 2]) >= cfg.CLIP_LEVEL;
        if (clipped) clippedTotal++;
        else if (Math.fround(lum[k]) > limit) kept[idx[k]] = 1;
      }
    }
    regions[name] = { x, y, w, h, mask, kept };

    // Mean colour of kept pixels in a rectangle of the crop.
    const meanIn = (c0, r0, c1, r1) => {
      let n = 0, sr = 0, sg = 0, sb = 0;
      for (let r = r0; r < Math.min(r1, ch); r++) {
        for (let c = c0; c < Math.min(c1, cw); c++) {
          if (!kept[r * w + c]) continue;
          const o = ((y + r) * W + (x + c)) * 4;
          sr += data[o]; sg += data[o + 1]; sb += data[o + 2]; n++;
        }
      }
      return n ? { n, rgb: [sr / n, sg / n, sb / n] } : { n: 0, rgb: null };
    };
    const whole = meanIn(0, 0, w, h);
    if (whole.n) {
      const [r, g, b] = whole.rgb;
      luminance[name] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
    const [cols, rows] = cfg.GRID[name];
    const key = name.replace(/ /g, "_");
    for (let i = 0; i < cols * rows; i++) {
      const cx = i % cols, cy = Math.floor(i / cols);
      const x0 = Math.floor((cx * w) / cols), x1 = Math.floor(((cx + 1) * w) / cols);
      const y0 = Math.floor((cy * h) / rows), y1 = Math.floor(((cy + 1) * h) / rows);
      const p = `${key}_p${i}`;
      patches[p] = [x + x0, y + y0, x + x1, y + y1];
      const m = meanIn(x0, y0, x1, y1);
      colors[p] = m.n >= cfg.MIN_PATCH_PIXELS ? m.rgb : null;
    }
  }
  const stats = { luminance, clipped_fraction: clippedTotal / Math.max(total, 1) };
  return { colors, patches, regions, stats };
}

export function faceColor(frame, lm, cfg) {
  // Mean RGB of the whole face minus eyes, brows and lips (face_regions.face_color).
  const { data, width: W, height: H } = frame;
  const { x, y, w, h, mask } = regionCrop(lm, cfg.FACE_OVAL, W, H);
  for (const indices of cfg.FACE_EXCLUDE) {
    const pts = indices.map((i) => [Math.trunc(lm[i][0]) - x, Math.trunc(lm[i][1]) - y]);
    fillConvex(mask, w, h, convexHull(pts), 0);
  }
  const cw = Math.min(w, W - x), ch = Math.min(h, H - y);
  let n = 0, sr = 0, sg = 0, sb = 0;
  for (let r = 0; r < ch; r++) {
    for (let c = 0; c < cw; c++) {
      if (!mask[r * w + c]) continue;
      const o = ((y + r) * W + (x + c)) * 4;
      if (Math.max(data[o], data[o + 1], data[o + 2]) >= cfg.CLIP_LEVEL) continue;
      sr += data[o]; sg += data[o + 1]; sb += data[o + 2]; n++;
    }
  }
  return n >= cfg.MIN_FACE_PIXELS ? [sr / n, sg / n, sb / n] : null;
}

export function backgroundColor(frame, lm, cfg) {
  // Mean RGB of the wall in the top corners, skipping boxes that overlap the head.
  const { data, width: w, height: h } = frame;
  let fx0 = Infinity, fy0 = Infinity, fx1 = -Infinity, fy1 = -Infinity;
  for (const [px, py] of lm) { fx0 = Math.min(fx0, px); fy0 = Math.min(fy0, py); fx1 = Math.max(fx1, px); fy1 = Math.max(fy1, py); }
  const pad = 0.35 * (fx1 - fx0);
  const head = [fx0 - pad, fy0 - 2 * pad, fx1 + pad, fy1 + pad];
  const bw = Math.trunc(0.18 * w), bh = Math.trunc(0.22 * h);
  const candidates = [[10, 40, 10 + bw, 40 + bh], [w - 10 - bw, 40, w - 10, 40 + bh]];
  const boxes = candidates.filter((b) => b[2] < head[0] || b[0] > head[2] || b[3] < head[1] || b[1] > head[3]);
  if (!boxes.length) return { rgb: null, boxes };
  let n = 0, sr = 0, sg = 0, sb = 0;
  for (const [x0, y0, x1, y1] of boxes) {
    for (let y = Math.max(y0, 0); y < Math.min(y1, h); y++) {
      for (let x = Math.max(x0, 0); x < Math.min(x1, w); x++) {
        const o = (y * w + x) * 4;
        if (Math.max(data[o], data[o + 1], data[o + 2]) >= cfg.CLIP_LEVEL) continue;
        sr += data[o]; sg += data[o + 1]; sb += data[o + 2]; n++;
      }
    }
  }
  return { rgb: n < 100 ? null : [sr / n, sg / n, sb / n], boxes };
}

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export function headPose(lm, w, h) {
  // nose x, nose y, face width, roll, mouth opening (frame_features.head_pose).
  const roll = Math.atan2(lm[263][1] - lm[33][1], lm[263][0] - lm[33][0]);
  const faceW = dist(lm[234], lm[454]) + 1e-9;
  return [lm[1][0] / w, lm[1][1] / h, faceW / w, roll, dist(lm[13], lm[14]) / faceW];
}

export class FeatureExtractor {
  // Keeps the little bit of state needed between frames (frame_features.FeatureExtractor).
  constructor(cfg) { this.cfg = cfg; this.reset(); }

  reset() { this.smooth = null; this.prevNose = null; }

  process(frame, lm) {
    if (!lm) { this.prevNose = null; return { face: false }; }
    const cfg = this.cfg;
    const faceWidth = dist(lm[234], lm[454]);
    const movement = this.prevNose ? dist(lm[1], this.prevNose) / faceWidth : 0;
    this.prevNose = [lm[1][0], lm[1][1]];
    const fast = movement > cfg.MOTION_LIMIT;
    if (!this.smooth || fast) this.smooth = lm.map((p) => [p[0], p[1]]);
    else this.smooth = this.smooth.map((p, i) => [
      cfg.SMOOTHING * p[0] + (1 - cfg.SMOOTHING) * lm[i][0],
      cfg.SMOOTHING * p[1] + (1 - cfg.SMOOTHING) * lm[i][1]]);
    const patchInfo = extractPatchColors(frame, this.smooth, cfg);
    patchInfo.colors[cfg.FACE_TRACE] = faceColor(frame, this.smooth, cfg);
    const bg = backgroundColor(frame, this.smooth, cfg);
    return {
      face: true, colors: patchInfo.colors, bg_rgb: bg.rgb,
      pose: headPose(lm, frame.width, frame.height),
      movement, fast_motion: fast, small_motion: movement > cfg.SMALL_MOTION,
      overlay: { patches: patchInfo.patches, regions: patchInfo.regions, bg_boxes: bg.boxes },
      stats: patchInfo.stats,
    };
  }
}
