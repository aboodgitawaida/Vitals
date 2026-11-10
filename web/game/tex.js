// tex.js -- every surface in the game is painted here, procedurally, at load time.
// No downloaded assets: wallpaper, linoleum, boards, ceiling tiles, metal, cloth,
// skin, signs and the flashlight's cookie are all canvases built from noise.

import * as THREE from "three";

// ------------------------------------------------------------------ noise

const hash = (x, y, s) => {
  let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(s, 1442695041)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
};

// Value noise that wraps every `px` lattice cells in x and `py` in y (0 = no wrap),
// so a texture tiles without a seam.
function vnoise(x, y, px, py, s) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const wx = (a) => (px ? ((a % px) + px) % px : a);
  const wy = (a) => (py ? ((a % py) + py) % py : a);
  const x0 = wx(xi), x1 = wx(xi + 1), y0 = wy(yi), y1 = wy(yi + 1);
  const a = hash(x0, y0, s), b = hash(x1, y0, s), c = hash(x0, y1, s), d = hash(x1, y1, s);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

// Fractal noise over u, v in [0, 1): `f` cells across, tiling in x (and y if tileY).
export function fbm(u, v, f, oct = 5, s = 0, tileY = true) {
  let t = 0, amp = 0.5, n = 0, fx = f;
  for (let i = 0; i < oct; i++) {
    t += amp * vnoise(u * fx, v * fx, fx, tileY ? fx : 0, s + i * 31);
    n += amp; amp *= 0.5; fx *= 2;
  }
  return t / n;
}

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const mix = (a, b, t) => a + (b - a) * t;
const mix3 = (a, b, t) => [mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t)];
const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];

export const rand = (() => { let s = 1337; return () => ((s = Math.imul(s ^ (s >>> 15), 2246822507) + 0x9e3779b9 | 0) >>> 0) / 4294967296; })();

// ------------------------------------------------------------------ painters

const tick = () => new Promise((r) => setTimeout(r, 0));

// Paint w x h pixels with painter(u, v) -> [r, g, b, height, roughness] (all 0..1).
// Returns colour, bump and roughness canvases.
async function paint(w, h, painter, after) {
  const mk = () => { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; };
  const col = mk(), bump = mk(), rough = mk();
  const cc = col.getContext("2d"), bc = bump.getContext("2d"), rc = rough.getContext("2d");
  const ci = cc.createImageData(w, h), bi = bc.createImageData(w, h), ri = rc.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    const v = y / h;
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 4;
      const p = painter(x / w, v);
      ci.data[o] = clamp(p[0]) * 255; ci.data[o + 1] = clamp(p[1]) * 255; ci.data[o + 2] = clamp(p[2]) * 255; ci.data[o + 3] = 255;
      const hb = clamp(p[3]) * 255; bi.data[o] = bi.data[o + 1] = bi.data[o + 2] = hb; bi.data[o + 3] = 255;
      const rb = clamp(p[4]) * 255; ri.data[o] = ri.data[o + 1] = ri.data[o + 2] = rb; ri.data[o + 3] = 255;
    }
    if ((y & 63) === 63) await tick();     // keep the loading screen alive
  }
  cc.putImageData(ci, 0, 0); bc.putImageData(bi, 0, 0); rc.putImageData(ri, 0, 0);
  if (after) after(cc, w, h, bc);
  return { col, bump, rough };
}

function tex(canvas, srgb, aniso) {
  const t = new THREE.CanvasTexture(canvas);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = aniso;
  return t;
}

async function material(w, h, painter, opts = {}, after) {
  const { col, bump, rough } = await paint(w, h, painter, after);
  const a = opts.aniso ?? 4;
  return new THREE.MeshStandardMaterial({
    map: tex(col, true, a), bumpMap: tex(bump, false, a), bumpScale: opts.bump ?? 2,
    roughnessMap: tex(rough, false, a), roughness: 1, metalness: opts.metal ?? 0,
  });
}

// Old wallpaper above a painted wainscot, water damage from above, mould from below.
function wallPainter(seed) {
  const paper = [[0.50, 0.49, 0.41], [0.44, 0.47, 0.42], [0.52, 0.46, 0.40]][seed % 3];
  return (u, v) => {
    const yF = 3 - v * 3;                                   // metres above the floor
    const dirt = fbm(u, v, 4, 5, seed);
    const fine = fbm(u, v, 32, 3, seed + 5);
    let c, h = 0.5, r = 0.85;
    if (yF < 0.14) {                                        // skirting board
      c = mul([0.17, 0.12, 0.09], 0.8 + fine * 0.4); h = 0.75; r = 0.6;
    } else if (yF < 1.0) {                                  // wainscot panel
      c = mul([0.27, 0.30, 0.27], 0.85 + fine * 0.25);
      const gu = Math.min(Math.abs(u - 0.04), Math.abs(u - 0.5), Math.abs(u - 0.96));
      const gy = Math.min(Math.abs(yF - 0.2), Math.abs(yF - 0.92));
      const groove = Math.min(gu * 2, gy / 1.5);
      if (groove < 0.012) { h = 0.25; c = mul(c, 0.6); }
      r = 0.55 + fine * 0.2;
    } else if (yF < 1.07) {                                 // chair rail
      c = mul([0.22, 0.16, 0.11], 0.8 + fine * 0.3); h = 0.85; r = 0.5;
    } else {                                                // wallpaper
      const stripe = 0.5 + 0.5 * Math.sin(u * Math.PI * 2 * 9);
      const d = Math.abs(((u * 6) % 1) - 0.5) + Math.abs(((yF * 2.2) % 1) - 0.5);
      const motif = smooth(0.34, 0.28, d) - smooth(0.24, 0.2, d) * 0.6;
      c = mul(paper, 0.9 + stripe * 0.06 + motif * 0.08 + (fine - 0.5) * 0.12);
      const peel = smooth(0.66, 0.68, fbm(u, v, 5, 5, seed + 9));
      if (peel > 0) {
        c = mix3(c, mul([0.62, 0.60, 0.55], 0.85 + fine * 0.3), peel);
        h = 0.5 - peel * 0.25;
      }
      const edge = smooth(0.64, 0.66, fbm(u, v, 5, 5, seed + 9)) - peel;
      h += edge * 0.4;
    }
    // Water running down from the ceiling: stretched noise, brown, with a darker tide line.
    const streak = fbm(u, v * 0.15, 24, 4, seed + 2, false);
    const wet = smooth(0.52, 0.72, fbm(u, v, 3, 4, seed + 3)) * Math.pow(1 - v, 0.6) * smooth(0.45, 0.65, streak);
    c = mix3(c, mul([0.30, 0.22, 0.12], 0.9), wet * 0.65);
    const tide = smooth(0.54, 0.56, fbm(u, v, 3, 4, seed + 3)) - smooth(0.57, 0.6, fbm(u, v, 3, 4, seed + 3));
    c = mul(c, 1 - tide * 0.35 * (1 - v));
    // Mould and grime near the floor.
    const mould = smooth(0.6, 0.72, fbm(u, v, 10, 4, seed + 4)) * smooth(0.55, 1.0, v);
    c = mix3(c, [0.05, 0.07, 0.04], mould * 0.8);
    c = mul(c, 0.55 + dirt * 0.55 - smooth(0.85, 1, v) * 0.2);
    r = clamp(r - wet * 0.25);
    return [c[0], c[1], c[2], h + (fine - 0.5) * 0.15, r];
  };
}

// Checkered linoleum, 4 x 4 tiles per 2 m cell, heavily used.
function linoPainter(u, v) {
  const i = Math.floor(u * 4), j = Math.floor(v * 4);
  const fu = (u * 4) % 1, fv = (v * 4) % 1;
  const edge = Math.min(fu, 1 - fu, fv, 1 - fv);
  const fine = fbm(u, v, 40, 3, 11);
  const tileVar = hash(i, j, 7);
  let c = (i + j) % 2 ? [0.58, 0.56, 0.48] : [0.17, 0.20, 0.18];
  c = mul(c, 0.85 + tileVar * 0.2 + (fine - 0.5) * 0.15);
  const grime = fbm(u, v, 3, 5, 12);
  const wear = smooth(0.35, 0.75, grime);
  c = mix3(c, [0.16, 0.14, 0.11], wear * 0.6);
  const scuff = smooth(0.62, 0.7, fbm(u * 1, v * 0.1, 30, 3, 13, false));
  c = mul(c, 1 - scuff * 0.25);
  let h = 0.6, r = 0.55 + grime * 0.3;
  if (edge < 0.02) { c = mul(c, 0.35); h = 0.2; r = 0.9; }
  const chip = smooth(0.78, 0.8, fbm(u, v, 16, 3, 14));
  if (chip > 0) { c = mix3(c, [0.12, 0.10, 0.08], chip); h = 0.25; }
  const wet = smooth(0.62, 0.7, fbm(u, v, 2, 4, 15));
  c = mul(c, 1 - wet * 0.3);
  r = clamp(mix(r, 0.12, wet));
  return [c[0], c[1], c[2], h + (fine - 0.5) * 0.1, r];
}

// Worn wooden floorboards.
function boardPainter(u, v) {
  const rows = 10, j = Math.floor(v * rows), fv = (v * rows) % 1;
  const off = hash(j, 3, 21);
  const len = 0.5;
  const k = Math.floor((u + off) / len), fu = ((u + off) / len) % 1;
  const pv = hash(j, k, 22);
  const grain = fbm(u + off, v * 0.08 + j * 0.37, 3, 5, 23, false);
  const rings = 0.5 + 0.5 * Math.sin((grain * 18 + pv * 6) * Math.PI);
  let c = mix3([0.20, 0.13, 0.08], [0.33, 0.22, 0.13], rings * 0.6 + pv * 0.4);
  const dust = fbm(u, v, 3, 5, 24);
  c = mix3(c, [0.36, 0.33, 0.28], smooth(0.5, 0.8, dust) * 0.45);
  let h = 0.6 + rings * 0.08, r = 0.6 + dust * 0.3;
  if (fv < 0.035 || fv > 0.965 || fu < 0.008 || fu > 0.992) { c = mul(c, 0.25); h = 0.15; r = 0.95; }
  const scratch = smooth(0.7, 0.75, fbm(u * 0.08, v, 60, 2, 25, false));
  c = mul(c, 1 + scratch * 0.25);
  return [c[0], c[1], c[2], h, r];
}

// Drop-ceiling tiles with water stains.
function ceilPainter(u, v) {
  const fu = (u * 2) % 1, fv = (v * 2) % 1;
  const edge = Math.min(fu, 1 - fu, fv, 1 - fv);
  const speck = fbm(u, v, 64, 2, 31);
  let c = mul([0.55, 0.54, 0.50], 0.8 + speck * 0.35);
  const s = fbm(u, v, 2, 5, 32);
  const stain = smooth(0.55, 0.62, s);
  c = mix3(c, [0.42, 0.33, 0.20], stain * 0.6);
  const ring = smooth(0.6, 0.615, s) - smooth(0.625, 0.64, s);
  c = mul(c, 1 - ring * 0.5);
  let h = 0.5 + (speck - 0.5) * 0.4, r = 0.95;
  if (edge < 0.012) { c = [0.16, 0.16, 0.15]; h = 0.8; r = 0.5; }
  c = mul(c, 0.6 + fbm(u, v, 4, 4, 33) * 0.4);
  return [c[0], c[1], c[2], h, r];
}

// A painted wooden door with recessed panels, paint chipped through to the wood.
function doorPainter(u, v) {
  const fine = fbm(u, v, 24, 3, 41);
  let c = mul([0.24, 0.27, 0.24], 0.85 + fine * 0.25);
  let h = 0.7;
  const inPanel = (x0, x1, y0, y1) => u > x0 && u < x1 && v > y0 && v < y1;
  for (const [x0, x1, y0, y1] of [[0.14, 0.86, 0.08, 0.42], [0.14, 0.86, 0.52, 0.92]]) {
    if (inPanel(x0, x1, y0, y1)) {
      const e = Math.min(u - x0, x1 - u, (v - y0) * 0.5, (y1 - v) * 0.5);
      h = e < 0.03 ? 0.4 + e * 6 : 0.55;
    }
  }
  const chip = smooth(0.66, 0.68, fbm(u, v, 8, 5, 42));
  c = mix3(c, mul([0.30, 0.19, 0.11], 0.8 + fine * 0.4), chip);
  const grime = fbm(u, v, 3, 4, 43);
  c = mul(c, 0.6 + grime * 0.5);
  // Hand marks around the handle.
  const hand = Math.exp(-((u - 0.85) ** 2 * 60 + (v - 0.52) ** 2 * 30));
  c = mul(c, 1 - hand * 0.45);
  return [c[0], c[1], c[2], h - chip * 0.1, 0.6 + grime * 0.3 - hand * 0.2];
}

function metalPainter(u, v) {
  const fine = fbm(u, v, 30, 3, 51);
  let c = mul([0.30, 0.34, 0.31], 0.85 + fine * 0.2);
  const rust = smooth(0.6, 0.75, fbm(u, v * 0.4, 6, 5, 52, false)) * (0.4 + v * 0.6);
  c = mix3(c, [0.30, 0.15, 0.07], rust);
  const scratch = smooth(0.74, 0.78, fbm(u, v * 0.05, 50, 2, 53, false));
  c = mix3(c, [0.55, 0.55, 0.52], scratch * 0.5);
  const vent = (v > 0.08 && v < 0.2 && ((u * 30) % 1) < 0.5 && u > 0.2 && u < 0.8) ? 1 : 0;
  if (vent) c = mul(c, 0.15);
  return [c[0], c[1], c[2], vent ? 0.1 : 0.6 + rust * 0.2, 0.45 + rust * 0.4];
}

function clothPainter(u, v) {
  const weave = (Math.sin(u * 600) * Math.sin(v * 600)) * 0.5 + 0.5;
  let c = mul([0.52, 0.50, 0.44], 0.85 + weave * 0.1);
  const stain = smooth(0.55, 0.7, fbm(u, v, 3, 5, 61));
  c = mix3(c, [0.38, 0.30, 0.18], stain * 0.7);
  c = mul(c, 0.7 + fbm(u, v, 6, 4, 62) * 0.4);
  return [c[0], c[1], c[2], 0.5 + weave * 0.2, 0.95];
}

function woodPainter(u, v) {
  const g = fbm(u * 0.1, v, 30, 4, 71, false);
  const rings = 0.5 + 0.5 * Math.sin(g * 30);
  let c = mix3([0.12, 0.07, 0.04], [0.24, 0.15, 0.09], rings);
  c = mul(c, 0.7 + fbm(u, v, 4, 4, 72) * 0.5);
  return [c[0], c[1], c[2], 0.5 + rings * 0.1, 0.45];
}

function skinPainter(u, v) {
  const m = fbm(u, v, 8, 5, 81);
  const vein = smooth(0.48, 0.5, fbm(u, v, 5, 5, 82)) - smooth(0.5, 0.52, fbm(u, v, 5, 5, 82));
  let c = mix3([0.20, 0.18, 0.17], [0.38, 0.34, 0.31], m);
  c = mix3(c, [0.22, 0.12, 0.14], smooth(0.6, 0.8, fbm(u, v, 3, 4, 83)) * 0.5);
  c = mix3(c, [0.18, 0.12, 0.14], vein * 0.6);
  return [c[0], c[1], c[2], 0.5 + m * 0.3 - vein * 0.2, 0.6 + m * 0.2];
}

// ------------------------------------------------------------------ flat canvases

function canvas(w, h, draw) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  return c;
}

export function textTexture(text, opts = {}) {
  const w = opts.w ?? 512, h = opts.h ?? 128;
  const c = canvas(w, h, (g) => {
    g.clearRect(0, 0, w, h);
    if (opts.bg) { g.fillStyle = opts.bg; g.fillRect(0, 0, w, h); }
    g.font = opts.font ?? `600 ${h * 0.55}px "Cormorant Garamond", Georgia, serif`;
    g.textAlign = "center"; g.textBaseline = "middle";
    if (opts.smear) {
      // Painted with a finger: several offset passes, then drips.
      for (let i = 0; i < 6; i++) {
        g.globalAlpha = 0.18;
        g.fillStyle = opts.color ?? "#5a1810";
        g.fillText(text, w / 2 + (rand() - 0.5) * 6, h / 2 + (rand() - 0.5) * 6);
      }
      g.globalAlpha = 0.6;
      for (let i = 0; i < 26; i++) {
        const x = w * 0.08 + rand() * w * 0.84, y = h * 0.55, len = rand() * h * 0.45;
        g.fillRect(x, y, 2 + rand() * 2, len);
      }
      g.globalAlpha = 1;
    } else {
      g.fillStyle = opts.color ?? "#d8d0bc";
      g.fillText(text, w / 2, h / 2);
    }
  });
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export function paperTexture(lines) {
  const c = canvas(256, 340, (g, w, h) => {
    g.fillStyle = "#cfc6b0"; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 2000; i++) { g.fillStyle = `rgba(80,60,30,${rand() * 0.06})`; g.fillRect(rand() * w, rand() * h, 3, 3); }
    g.fillStyle = "rgba(40,30,20,0.75)";
    for (let i = 0; i < lines; i++) g.fillRect(24, 40 + i * 18, 150 + rand() * 60, 3);
  });
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

// The flashlight's projected pattern: hot centre, a ring, a soft outer spill, lens smudges.
export function cookieTexture() {
  const c = canvas(256, 256, (g, w, h) => {
    const grd = g.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
    grd.addColorStop(0, "rgba(255,255,255,1)");
    grd.addColorStop(0.18, "rgba(255,250,240,0.95)");
    grd.addColorStop(0.32, "rgba(200,195,185,0.55)");
    grd.addColorStop(0.42, "rgba(255,250,235,0.75)");
    grd.addColorStop(0.5, "rgba(120,115,110,0.35)");
    grd.addColorStop(0.85, "rgba(40,38,36,0.12)");
    grd.addColorStop(1, "rgba(0,0,0,0)");
    g.fillStyle = grd; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 40; i++) {
      g.fillStyle = `rgba(0,0,0,${rand() * 0.08})`;
      g.beginPath(); g.arc(w / 2 + (rand() - 0.5) * w * 0.6, h / 2 + (rand() - 0.5) * h * 0.6, 4 + rand() * 18, 0, 7); g.fill();
    }
  });
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

// A foggy window: pale light behind dirty glass and a cross frame.
export function windowTexture() {
  const c = canvas(256, 320, (g, w, h) => {
    const grd = g.createLinearGradient(0, 0, 0, h);
    grd.addColorStop(0, "#b8c0bf"); grd.addColorStop(1, "#7f8786");
    g.fillStyle = grd; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 3000; i++) { g.fillStyle = `rgba(60,55,45,${rand() * 0.1})`; g.fillRect(rand() * w, rand() * h, 2, 2 + rand() * 8); }
    g.fillStyle = "#1b1a17";
    g.fillRect(0, 0, w, 14); g.fillRect(0, h - 14, w, 14); g.fillRect(0, 0, 14, h); g.fillRect(w - 14, 0, 14, h);
    g.fillRect(w / 2 - 6, 0, 12, h); g.fillRect(0, h / 2 - 6, w, 12);
  });
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

// ------------------------------------------------------------------ everything

export async function makeMaterials(onProgress) {
  const steps = [
    ["wall0", () => material(512, 768, wallPainter(0), { bump: 3 })],
    ["wall1", () => material(512, 768, wallPainter(1), { bump: 3 })],
    ["wall2", () => material(512, 768, wallPainter(2), { bump: 3 })],
    ["lino", () => material(512, 512, linoPainter, { bump: 2, aniso: 8 })],
    ["boards", () => material(512, 512, boardPainter, { bump: 2, aniso: 8 })],
    ["ceil", () => material(256, 256, ceilPainter, { bump: 2 })],
    ["door", () => material(256, 512, doorPainter, { bump: 3 })],
    ["metal", () => material(256, 256, metalPainter, { bump: 2, metal: 0.4 })],
    ["cloth", () => material(256, 256, clothPainter, { bump: 1 })],
    ["wood", () => material(256, 256, woodPainter, { bump: 1 })],
    ["skin", () => material(256, 256, skinPainter, { bump: 3 })],
  ];
  const M = {};
  for (let i = 0; i < steps.length; i++) {
    onProgress?.(i / steps.length, steps[i][0]);
    M[steps[i][0]] = await steps[i][1]();
  }
  M.dark = new THREE.MeshStandardMaterial({ color: 0x0b0a09, roughness: 0.7 });
  M.brass = new THREE.MeshStandardMaterial({ color: 0x9c7a3c, roughness: 0.35, metalness: 1 });
  M.chrome = new THREE.MeshStandardMaterial({ color: 0x8a8a86, roughness: 0.3, metalness: 1 });
  M.plastic = new THREE.MeshStandardMaterial({ color: 0x23211e, roughness: 0.5 });
  M.ceramic = new THREE.MeshStandardMaterial({ color: 0xb9b5a8, roughness: 0.25 });
  M.paper = new THREE.MeshStandardMaterial({ map: paperTexture(10), roughness: 0.9 });
  onProgress?.(1, "done");
  return M;
}
