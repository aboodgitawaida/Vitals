// app.js -- the browser version of vitals_app.py.
//
// Page (this file):  camera -> mirror -> MediaPipe FaceLandmarker -> skin-patch colours
//                    (features.js) -> dashboard
// Worker (worker.js): live_session.py in Pyodide: buffer, analysis, tracker, status text
//
// Frames are batched while the worker is busy (one analysis at a time, like the
// desktop app's background process), so the camera view never waits for Python.

import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";
import { FeatureExtractor } from "./features.js";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const VIEW_WIDTH = 960;          // the on-screen copy (measurements use the full camera frame)
const GAP_RESET_SEC = 1.0;       // no camera frames for this long (tab hidden): start over
const SLOW_MESH_MS = 50;         // face tracking slower than this on "GPU": try CPU instead
const HR_LOW_BPM = 42, HR_HIGH_BPM = 240;
const LEVEL_TEXT = { good: "GOOD SIGNAL", fair: "FAIR SIGNAL", none: "NO READING" };
const TINT = { good: [110, 220, 120], fair: [250, 195, 60], none: [235, 110, 110] };
const REJECTED = [170, 70, 200];  // hair, brows, shadow, glare: not used

const $ = (id) => document.getElementById(id);
const state = {
  worker: null, landmarker: null, extractor: null, cfg: null, stream: null, video: null,
  grab: null, grabCtx: null, view: $("view"), viewCtx: null,
  pending: [], busy: false, info: null, running: false,
  t0: 0, lastFrame: null, fps: 0, method: "auto", background: true, light: false, showMap: true,
  showPerf: false, perf: {}, delegate: null, switching: false, meshFrames: 0,
  camFps: 0, lastPresented: null,
};

// ---------------------------------------------------------------- startup

function startWorker() {
  return new Promise((resolve, reject) => {
    const w = new Worker("worker.js", { type: "module" });
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "status") setLoad(m.text);
      else if (m.type === "ready") resolve({ worker: w, cfg: m.cfg });
      else if (m.type === "info") onInfo(m);
      else if (m.type === "error") { console.error(m.text); if (!state.cfg) reject(new Error(m.text)); else state.busy = false; }
    };
    w.onerror = (e) => reject(new Error(e.message || "worker failed"));
    w.postMessage({ type: "init" });
  });
}

let fileset = null;
async function createLandmarker(delegate) {
  fileset ??= await FilesetResolver.forVisionTasks(WASM_URL);
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO", numFaces: 1,
    minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
}

async function startLandmarker() {
  try {
    return { landmarker: await createLandmarker("GPU"), delegate: "GPU" };
  } catch (err) {
    console.warn("GPU face tracking unavailable, using CPU", err);
    return { landmarker: await createLandmarker("CPU"), delegate: "CPU" };
  }
}

function maybeSwitchToCpu() {
  // Without a real GPU (software WebGL), "GPU" face tracking can be 10x slower than CPU.
  if (state.delegate !== "GPU" || state.switching || state.meshFrames < 30 || state.perf.mesh < SLOW_MESH_MS) return;
  state.switching = true;
  console.warn(`face tracking takes ${state.perf.mesh.toFixed(0)} ms on GPU: switching to CPU`);
  createLandmarker("CPU").then((lm) => {
    if (!state.running) { lm.close(); return; }
    const old = state.landmarker;
    Object.assign(state, { landmarker: lm, delegate: "CPU", switching: false });
    delete state.perf.mesh;
    old.close();
  }).catch((err) => console.warn("CPU face tracking failed, staying on GPU", err));
}

async function startCamera() {
  // HD on purpose: each patch colour is an average, and more pixels = less noise.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
  });
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;
  await video.play();
  return { stream, video };
}

function setLoad(text) { $("load-status").textContent = text; }

// Face tracking and Python start downloading as soon as the page opens, while the
// visitor reads the intro; the camera is only switched on when they click Start.
let loading = null;
function preload() {
  if (!loading) {
    loading = { lm: startLandmarker(), py: startWorker() };
    Promise.all([loading.lm, loading.py]).then(
      () => { if (!state.running && !state.starting) setLoad("Ready."); },
      () => {});                                     // failures are reported when Start is clicked
  }
  return loading;
}

function discardPreload() {
  // Something failed to load: release what did load, so the next Start loads afresh.
  const old = loading;
  loading = null;
  old?.lm.then((r) => r.landmarker.close(), () => {});
  old?.py.then((r) => r.worker.terminate(), () => {});
}

async function start() {
  const btn = $("start");
  btn.disabled = true;
  state.starting = true;
  if (!navigator.mediaDevices?.getUserMedia) {
    setLoad("This browser cannot use the camera here (the page needs HTTPS).");
    btn.disabled = false;
    state.starting = false;
    return;
  }
  const { lm: lmLoad, py: pyLoad } = preload();
  const cam = await startCamera().then((value) => ({ value }), (reason) => ({ reason }));
  if (cam.reason) {
    // Keep the preloaded face tracking and Python: only the camera needs another try.
    const err = cam.reason;
    console.error(err);
    const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
    setLoad(denied ? "Camera permission was denied. Allow camera access for this site and try again."
                   : `Could not start the camera: ${err?.message || err}`);
    btn.disabled = false;
    state.starting = false;
    return;
  }
  Object.assign(state, cam.value);
  setLoad("Camera on. Finishing loading…");
  const [lm, py] = await Promise.allSettled([lmLoad, pyLoad]);
  state.starting = false;
  if (lm.status === "rejected" || py.status === "rejected") {
    const err = (lm.status === "rejected" ? lm : py).reason;
    console.error(err);
    discardPreload();
    stop();
    setLoad(`Could not start: ${err?.message || err}`);
    return;
  }
  Object.assign(state, lm.value, py.value);
  state.extractor = new FeatureExtractor(state.cfg);
  const w = state.video.videoWidth, h = state.video.videoHeight;
  state.grab = document.createElement("canvas");
  state.grab.width = w; state.grab.height = h;
  state.grabCtx = state.grab.getContext("2d", { willReadFrequently: true });
  state.view.width = Math.min(VIEW_WIDTH, w);
  state.view.height = Math.round((state.view.width * h) / w);
  state.viewCtx = state.view.getContext("2d");
  $("intro").hidden = true;
  $("app").hidden = false;
  console.log(`camera: ${w}x${h}`);
  state.running = true;
  state.t0 = performance.now();
  scheduleFrame();
}

function stop() {
  const wasRunning = state.running;
  state.running = false;
  state.stream?.getTracks().forEach((t) => t.stop());
  if (wasRunning) {                     // these came from the preload: load fresh ones for next time
    loading = null;
    state.worker?.terminate();
    state.landmarker?.close();
  }
  Object.assign(state, {
    stream: null, worker: null, landmarker: null, cfg: null, pending: [], busy: false, info: null,
    lastFrame: null, lastPresented: null, fps: 0, camFps: 0, perf: {}, meshFrames: 0, switching: false,
  });
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  setLight(false, false);
  $("app").hidden = true;
  $("intro").hidden = false;
  $("start").disabled = false;
  setLoad("");
  if (wasRunning) preload();             // so starting again is quick
}

// ---------------------------------------------------------------- per frame

function scheduleFrame() {
  if (!state.running) return;
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) state.video.requestVideoFrameCallback(onFrame);
  else requestAnimationFrame(onFrame);
}

function onFrame(_ts, video) {
  if (!state.running) return;
  const now = (performance.now() - state.t0) / 1000;
  // The camera's own frame rate (frames it delivered, processed or not): this is what drops in
  // low light. Without requestVideoFrameCallback, fall back to the processing rate.
  const presented = video?.presentedFrames;
  if (state.lastFrame !== null) {
    const dt = now - state.lastFrame;
    if (dt > GAP_RESET_SEC) {                      // tab was hidden: old samples no longer line up
      control("reset");
      state.extractor.reset();
      state.pending = [];
    } else {
      state.fps = 0.9 * state.fps + 0.1 / Math.max(dt, 1e-3);
      if (presented !== undefined && state.lastPresented !== null) {
        state.camFps = 0.9 * state.camFps + 0.1 * (presented - state.lastPresented) / Math.max(dt, 1e-3);
      }
    }
  }
  state.lastFrame = now;
  state.lastPresented = presented ?? null;

  // Mirror view, full resolution: FaceLandmarker and the colour measurement both use it.
  const { grab, grabCtx: g } = state;
  const W = grab.width, H = grab.height;
  let t = performance.now();
  const lap = (stage) => { const n = performance.now(); timeStage(stage, n - t); t = n; };
  g.setTransform(-1, 0, 0, 1, W, 0);
  g.drawImage(state.video, 0, 0, W, H);
  g.setTransform(1, 0, 0, 1, 0, 0);
  lap("grab");

  const result = state.landmarker.detectForVideo(grab, performance.now());
  const face = result.faceLandmarks?.[0];
  const lm = face ? face.map((p) => [Math.fround(p.x * W), Math.fround(p.y * H)]) : null;
  lap("mesh");
  state.meshFrames++;
  const frame = lm ? g.getImageData(0, 0, W, H) : { width: W, height: H };
  lap("read");
  const ff = state.extractor.process(frame, lm);
  lap("patches");

  state.pending.push({
    t: now, face: ff.face, colors: ff.colors, bg: ff.bg_rgb, pose: ff.pose,
    fast: ff.fast_motion, small: ff.small_motion, stats: ff.stats,
    fps: state.lastPresented === null ? state.fps : state.camFps,
  });
  flush();
  drawView(ff);
  lap("draw");
  renderPerf();
  maybeSwitchToCpu();
  scheduleFrame();
}

function timeStage(stage, ms) {
  const p = state.perf;
  p[stage] = p[stage] === undefined ? ms : 0.9 * p[stage] + 0.1 * ms;
}

function renderPerf() {
  // Where each frame's time goes (the desktop app's P readout). At 30 fps the budget is 33 ms.
  const el = $("perf");
  if (!state.showPerf) { el.textContent = ""; return; }
  const g = (k) => (state.perf[k] === undefined ? "-" : state.perf[k].toFixed(0));
  const work = ["grab", "mesh", "read", "patches", "draw"].reduce((a, k) => a + (state.perf[k] || 0), 0);
  el.textContent = `${Math.round(state.fps)} fps (camera ${Math.round(state.camFps)}) | ${state.delegate} | work ${work.toFixed(0)} ms/frame (camera ${g("grab")}, ` +
    `mesh ${g("mesh")}, pixels ${g("read")}, patches ${g("patches")}, draw ${g("draw")}) | analysis ${g("analysis")} ms`;
}

function flush() {
  if (state.busy || !state.pending.length || !state.worker) return;
  state.busy = true;
  state.worker.postMessage({ type: "frames", frames: state.pending });
  state.pending = [];
}

function onInfo(msg) {
  state.busy = false;
  state.info = msg.info;
  if (msg.ran) timeStage("analysis", msg.ms);
  renderDashboard(msg.info);
  flush();
}

function control(action, value) {
  state.worker?.postMessage({ type: "control", action, value });
}

// ---------------------------------------------------------------- drawing

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function snrLevel(snr) {
  if (snr === null || snr === undefined) return "none";
  return snr >= 3 ? "good" : snr >= 0 ? "fair" : "none";
}

function drawView(ff) {
  const ctx = state.viewCtx, v = state.view;
  const k = v.width / state.grab.width;
  ctx.drawImage(state.grab, 0, 0, v.width, v.height);
  if (!state.showMap || !ff.face) return;
  // Patch SNR map: skin pixels tinted by how clean each patch's pulse is; rejected pixels purple.
  const patchSnr = state.info?.patch_snr || {};
  for (const [name, r] of Object.entries(ff.overlay.regions)) {
    const [cols, rows] = state.cfg.GRID[name];
    const key = name.replace(/ /g, "_");
    const img = new ImageData(r.w, r.h);
    const tints = [];
    for (let i = 0; i < cols * rows; i++) tints.push(TINT[snrLevel(patchSnr[`${key}_p${i}`])]);
    for (let y = 0; y < r.h; y++) {
      const cy = Math.min(Math.floor((y * rows) / r.h), rows - 1);
      for (let x = 0; x < r.w; x++) {
        const j = y * r.w + x;
        if (!r.mask[j]) continue;
        const c = r.kept[j] ? tints[cy * cols + Math.min(Math.floor((x * cols) / r.w), cols - 1)] : REJECTED;
        img.data.set([c[0], c[1], c[2], 90], j * 4);
      }
    }
    const tmp = new OffscreenCanvas(r.w, r.h);
    tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.drawImage(tmp, r.x * k, r.y * k, r.w * k, r.h * k);
  }
  ctx.strokeStyle = "rgba(230,230,230,0.9)";
  ctx.fillStyle = "rgba(230,230,230,0.9)";
  ctx.font = "12px system-ui, sans-serif";
  ctx.lineWidth = 1;
  for (const [x0, y0, x1, y1] of ff.overlay.bg_boxes) {
    ctx.strokeRect(x0 * k + 0.5, y0 * k + 0.5, (x1 - x0) * k, (y1 - y0) * k);
    ctx.fillText("bg ref", x0 * k + 5, y0 * k + 15);
  }
}

function renderDashboard(info) {
  $("bpm").textContent = info.bpm === null ? "--" : Math.round(info.bpm);
  document.querySelector(".hr").dataset.level = info.level;
  $("level-text").textContent = info.state === "holding" ? "HOLDING (weak signal)" : LEVEL_TEXT[info.level];
  const snr = info.snr_db === null ? "SNR --" : `SNR ${info.snr_db >= 0 ? "+" : ""}${info.snr_db.toFixed(1)} dB`;
  const used = (info.method_used || (info.method === "auto" ? "pos" : info.method)).toUpperCase();
  let method = info.method === "auto" ? `AUTO>${used}` : info.method.toUpperCase();
  if (info.agreement !== null) method += ` · ${info.agreement}/3 agree`;
  $("details").textContent = `${snr} · ${method} · ${Math.round(state.fps)} fps`;
  $("lighting").textContent = info.lighting[1];
  $("lighting").dataset.level = info.lighting[0];
  $("status").textContent = info.status;
  $("features").textContent = `bg ${info.use_background ? "ON" : "OFF"} · motion ` +
    (info.motion_patches === null ? "-" : `${info.motion_patches}/${state.cfg.PATCHES.length}`);
  renderRegions(info.regions);
  drawSpectrum(info);
  drawLine($("pulse"), info.pulse);
  drawTrend(info.trend);
}

function renderRegions(regions) {
  const el = $("regions");
  el.replaceChildren(...Object.entries(regions).map(([name, [snr, share]]) => {
    const row = document.createElement("div");
    row.className = "region";
    const label = document.createElement("span");
    label.textContent = name;
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("span");
    const num = document.createElement("span");
    num.className = "num";
    if (snr === null) {
      num.textContent = "no skin";
      num.style.color = cssVar("--none");
    } else {
      fill.style.width = `${Math.max(0, Math.min(1, (snr + 6) / 15)) * 100}%`;
      fill.style.background = cssVar(`--${snrLevel(snr)}`);
      num.textContent = `${snr >= 0 ? "+" : ""}${snr.toFixed(1)}  ${Math.round(share * 100)}%`;
    }
    bar.append(fill);
    row.append(label, bar, num);
    return row;
  }));
}

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px system-ui, sans-serif";
  ctx.lineJoin = "round";
  return { ctx, w, h };
}

function placeholder(ctx, w, h, text) {
  ctx.fillStyle = cssVar("--muted");
  ctx.fillText(text, 10, h / 2);
}

function drawSpectrum(info) {
  // Power at each frequency in the heart-rate band. The tallest bump is the heart rate.
  const { ctx, w, h } = setupCanvas($("spectrum"));
  const top = 4, bottom = h - 16, x = (bpm) => ((bpm - HR_LOW_BPM) / (HR_HIGH_BPM - HR_LOW_BPM)) * w;
  ctx.strokeStyle = cssVar("--grid");
  ctx.fillStyle = cssVar("--muted");
  for (const bpm of [60, 90, 120, 150, 180, 210]) {
    ctx.beginPath(); ctx.moveTo(x(bpm), top); ctx.lineTo(x(bpm), bottom); ctx.stroke();
    ctx.fillText(bpm, x(bpm) - 8, h - 3);
  }
  if (!info.spectrum) return placeholder(ctx, w, bottom, "waiting for data…");
  const [bpms, power] = info.spectrum;
  let max = 1e-12;
  bpms.forEach((b, i) => { if (b >= HR_LOW_BPM && b <= HR_HIGH_BPM) max = Math.max(max, power[i]); });
  ctx.beginPath();
  let first = true;
  bpms.forEach((b, i) => {
    if (b < HR_LOW_BPM || b > HR_HIGH_BPM) return;
    const px = x(b), py = bottom - (power[i] / max) * (bottom - top);
    if (first) { ctx.moveTo(px, py); first = false; } else ctx.lineTo(px, py);
  });
  ctx.strokeStyle = cssVar("--wave");
  ctx.lineWidth = 2;
  ctx.stroke();
  if (info.peak_bpm !== null) {
    const color = cssVar(`--${info.level}`);
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x(info.peak_bpm), top); ctx.lineTo(x(info.peak_bpm), bottom); ctx.stroke();
    ctx.fillText(`peak ${Math.round(info.peak_bpm)}`, Math.min(x(info.peak_bpm) + 5, w - 60), top + 10);
  }
}

function drawLine(canvas, values) {
  const { ctx, w, h } = setupCanvas(canvas);
  if (!values || values.length < 2) return placeholder(ctx, w, h, "waiting for data…");
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const sd = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length) + 1e-9;
  ctx.beginPath();
  values.forEach((v, i) => {
    const z = Math.max(-3, Math.min(3, (v - mean) / sd));
    const px = (i / (values.length - 1)) * w, py = h / 2 - z * (h / 6.5);
    if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
  });
  ctx.strokeStyle = cssVar("--wave");
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawTrend(trend) {
  // Confident readings from the last 60 s; gaps where there was no reading.
  const { ctx, w, h } = setupCanvas($("trend"));
  const valid = trend.filter(([, b]) => b !== null).map(([, b]) => b);
  if (!valid.length) return placeholder(ctx, w, h, "no readings yet");
  const now = trend[trend.length - 1][0];
  const lo = Math.min(...valid) - 8, hi = Math.max(...valid) + 8;
  const left = 30, top = 4, bottom = h - 4;
  ctx.strokeStyle = cssVar("--grid");
  ctx.fillStyle = cssVar("--muted");
  ctx.lineWidth = 1;
  for (const f of [0, 0.5, 1]) {
    const y = bottom - f * (bottom - top);
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(w, y); ctx.stroke();
    ctx.fillText(Math.round(lo + f * (hi - lo)), 2, y + 4);
  }
  ctx.strokeStyle = cssVar("--good");
  ctx.lineWidth = 2;
  ctx.beginPath();
  let pen = false;
  for (const [t, b] of trend) {
    if (b === null) { pen = false; continue; }
    const px = left + (1 - (now - t) / 60) * (w - left), py = bottom - ((b - lo) / (hi - lo)) * (bottom - top);
    if (pen) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    pen = true;
  }
  ctx.stroke();
}

// ---------------------------------------------------------------- controls

function setMethod(method) {
  state.method = method;
  document.querySelectorAll("[data-method]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.method === method)));
  control("method", method);
}

function setBackground(on) {
  state.background = on;
  const b = $("bg-toggle");
  b.setAttribute("aria-pressed", String(on));
  b.textContent = `Background: ${on ? "on" : "off"}`;
  control("background", on);
}

function setLight(on, fullscreen = true) {
  // White full screen: the laptop screen becomes a soft front light.
  state.light = on;
  document.documentElement.dataset.theme = on ? "light" : "dark";
  $("light-toggle").setAttribute("aria-pressed", String(on));
  if (fullscreen) {
    if (on && !document.fullscreenElement) document.documentElement.requestFullscreen?.().catch(() => {});
    if (!on && document.fullscreenElement) document.exitFullscreen().catch(() => {});
  }
  // The camera re-adjusts exposure after a big lighting change; start fresh.
  control("reset");
  state.extractor?.reset();
  if (state.info) renderDashboard(state.info);
}

function setMap(on) {
  state.showMap = on;
  $("map-toggle").setAttribute("aria-pressed", String(on));
}

$("start").addEventListener("click", start);
$("stop").addEventListener("click", stop);
document.querySelectorAll("[data-method]").forEach((b) => b.addEventListener("click", () => setMethod(b.dataset.method)));
$("bg-toggle").addEventListener("click", () => setBackground(!state.background));
$("light-toggle").addEventListener("click", () => setLight(!state.light));
$("map-toggle").addEventListener("click", () => setMap(!state.showMap));
document.addEventListener("fullscreenchange", () => { if (!document.fullscreenElement && state.light) setLight(false, false); });
document.addEventListener("keydown", (e) => {
  if (!state.running || e.metaKey || e.ctrlKey || e.altKey) return;
  const methods = ["auto", "pos", "green"];
  if (e.key === "m") setMethod(methods[(methods.indexOf(state.method) + 1) % methods.length]);
  else if (e.key === "b") setBackground(!state.background);
  else if (e.key === "l") setLight(!state.light);
  else if (e.key === "r") setMap(!state.showMap);
  else if (e.key === "p") state.showPerf = !state.showPerf;
});

preload();
