// sensors.js -- what the game knows about your body.
//
//   camera  -> the Vitals pipeline (MediaPipe face mesh -> skin colour -> live_session.py in
//              Pyodide) for heart rate, plus MediaPipe blendshapes for the fast signals:
//              eyes widening, brows up, mouth opening, a flinch, looking away.
//   mic     -> loudness: gasps, screams, talking. The Tenant can hear it.
//   neither -> a simulated heart that reacts to what happens in the game, so the game
//              is playable anywhere. The HUD says SIMULATED when it is.
//
// Heart rate is slow (the first number takes ~7 s, then it follows a 15 s window), so it
// drives pacing; the face and the microphone give the instant reactions.

import { rand } from "./tex.js";

const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const GAP_RESET_SEC = 1.0;

export class Sensors {
  constructor() {
    this.camera = false; this.mic = false;
    this.bpm = null; this.confidence = 0; this.settling = true; this.status = "";
    this.face = { present: false, eyeWide: 0, brow: 0, jaw: 0, blink: 0, size: 0, lookAway: 0 };
    this.micDb = -90; this.micFloor = -60; this.loud = 0;
    this.startles = [];                  // timestamps (s) of sudden reactions
    this.t = 0;
    // The simulated body: arousal from scares and the monster, exertion from running.
    this.sim = { arousal: 0.05, exertion: 0, bpm: 68 + rand() * 8, rest: 66 + rand() * 8, calm: 0, next: 0, profile: {} };
  }

  get simulated() { return !this.camera; }

  // ---------------------------------------------------------- camera (Vitals)

  async startCamera(onStatus) {
    onStatus?.("Starting the camera…");
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false, video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
    });
    const video = document.createElement("video");
    video.muted = true; video.playsInline = true; video.srcObject = stream; await video.play();
    this.stream = stream; this.video = video;
    onStatus?.("Loading face tracking…");
    const [{ FaceLandmarker, FilesetResolver }, { FaceLock, FeatureExtractor }, cfg] = await Promise.all([
      import(`${MP}/vision_bundle.mjs`),
      import("../features.js"),
      fetch("../face_layout.json").then((r) => { if (!r.ok) throw new Error("could not load face_layout.json"); return r.json(); }),
    ]);
    const fileset = await FilesetResolver.forVisionTasks(`${MP}/wasm`);
    const make = (delegate) => FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate }, runningMode: "VIDEO", numFaces: 1,
      outputFaceBlendshapes: true, minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    try { this.landmarker = await make("GPU"); } catch { this.landmarker = await make("CPU"); }
    this.extractor = new FeatureExtractor(cfg); this.lock = new FaceLock(cfg);
    const w = video.videoWidth, h = video.videoHeight;
    this.grab = document.createElement("canvas"); this.grab.width = w; this.grab.height = h;
    this.g = this.grab.getContext("2d", { willReadFrequently: true });
    this.pending = []; this.busy = false; this.t0 = performance.now(); this.lastFrame = null; this.fps = 0;
    this.camera = true;
    onStatus?.("Loading the heart-rate analysis (Python)…");
    this.worker = await new Promise((resolve, reject) => {
      const wk = new Worker(new URL("../worker.js", import.meta.url), { type: "module" });
      let ready = false;
      wk.onmessage = (e) => {
        const m = e.data;
        if (m.type === "status") onStatus?.(m.text);
        else if (m.type === "ready") { ready = true; resolve(wk); }
        else if (m.type === "info") this.onInfo(m.info);
        else if (m.type === "error") { if (!ready) reject(new Error(m.text)); else this.busy = false; }
      };
      wk.onerror = (e) => reject(new Error(e.message || "worker failed"));
      wk.postMessage({ type: "init" });
      this.frameLoop();                  // measuring starts while Python loads
    });
    this.flush();
    setTimeout(() => this.lockExposure(), 2000);
    onStatus?.("Camera ready.");
  }

  async lockExposure() {
    // Auto exposure changes skin colour far more than a heartbeat does: freeze it once settled.
    const track = this.stream?.getVideoTracks()[0];
    if (!track?.getCapabilities) return;
    const caps = track.getCapabilities(), now = track.getSettings(), adv = {};
    if (caps.exposureMode?.includes("manual") && now.exposureTime) Object.assign(adv, { exposureMode: "manual", exposureTime: now.exposureTime });
    if (caps.whiteBalanceMode?.includes("manual") && now.colorTemperature) Object.assign(adv, { whiteBalanceMode: "manual", colorTemperature: now.colorTemperature });
    if (Object.keys(adv).length) try { await track.applyConstraints({ advanced: [adv] }); } catch {}
  }

  frameLoop() {
    const next = () => {
      if (!this.camera) return;
      if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) this.video.requestVideoFrameCallback((_, m) => this.onFrame(m));
      else requestAnimationFrame(() => this.onFrame());
    };
    this.nextFrame = next;
    next();
  }

  onFrame(meta) {
    try { this.processFrame(meta); } catch (err) { console.warn(err); }
    this.nextFrame();
  }

  processFrame(meta) {
    const stamp = meta?.captureTime ?? meta?.presentationTime ?? performance.now();
    const now = (stamp - this.t0) / 1000;
    if (this.lastFrame !== null && now <= this.lastFrame) return;
    if (this.lastFrame !== null) {
      const dt = now - this.lastFrame;
      if (dt > GAP_RESET_SEC) { this.control("reset"); this.extractor.reset(); this.pending = []; }
      else this.fps = 0.9 * this.fps + 0.1 / Math.max(dt, 1e-3);
    }
    this.lastFrame = now;
    const { grab, g } = this, W = grab.width, H = grab.height;
    g.setTransform(-1, 0, 0, 1, W, 0); g.drawImage(this.video, 0, 0, W, H); g.setTransform(1, 0, 0, 1, 0, 0);
    const result = this.landmarker.detectForVideo(grab, performance.now());
    const faces = (result.faceLandmarks || []).map((f) => f.map((p) => [Math.fround(p.x * W), Math.fround(p.y * H)]));
    const pick = this.lock.pick(faces, now, W, H);
    if (this.lock.switched) { this.control("reset"); this.extractor.reset(); this.pending = []; }
    const lm = pick === null ? null : faces[pick];
    const frame = lm ? g.getImageData(0, 0, W, H) : { width: W, height: H };
    const ff = this.extractor.process(frame, lm, faces.filter((_, i) => i !== pick));
    this.pending.push({ t: now, face: ff.face, colors: ff.colors, bg: ff.bg_rgb, pose: ff.pose, fast: ff.fast_motion, small: ff.small_motion, stats: ff.stats, fps: this.fps });
    this.flush();
    this.readFace(lm, pick === null ? null : result.faceBlendshapes?.[pick], W);
  }

  flush() {
    if (!this.worker || this.busy || !this.pending.length) return;
    this.busy = true;
    this.worker.postMessage({ type: "frames", frames: this.pending });
    this.pending = [];
  }

  control(action, value) { this.worker?.postMessage({ type: "control", action, value }); }

  onInfo(info) {
    this.busy = false;
    this.bpm = info.bpm ?? null; this.confidence = info.confidence ?? 0; this.settling = !!info.settling; this.status = info.status ?? "";
    this.flush();
  }

  readFace(lm, shapes, W) {
    const f = this.face, wasPresent = f.present;
    f.present = !!lm;
    if (!lm) { f.lookAway += 1 / 30; return; }
    f.lookAway = 0;
    const s = {};
    for (const c of shapes?.categories ?? []) s[c.categoryName] = c.score;
    const eyeWide = ((s.eyeWideLeft ?? 0) + (s.eyeWideRight ?? 0)) / 2;
    const brow = s.browInnerUp ?? 0, jaw = s.jawOpen ?? 0, blink = ((s.eyeBlinkLeft ?? 0) + (s.eyeBlinkRight ?? 0)) / 2;
    const size = Math.hypot(lm[454][0] - lm[234][0], lm[454][1] - lm[234][1]) / W;     // face width: leaning in or back
    // A startle: the face changes a lot in a fraction of a second.
    const react = eyeWide + brow + jaw * 0.7;
    this.reactAvg ??= react; this.sizeAvg ??= size;
    const jump = react - this.reactAvg, flinch = this.sizeAvg ? (this.sizeAvg - size) / this.sizeAvg : 0;
    if (wasPresent && (jump > 0.35 || flinch > 0.12)) this.startle("face");
    this.reactAvg += (react - this.reactAvg) * 0.03; this.sizeAvg += (size - this.sizeAvg) * 0.02;
    Object.assign(f, { eyeWide, brow, jaw, blink, size });
  }

  // ---------------------------------------------------------- microphone

  async startMic() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false } });
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = ctx.createMediaStreamSource(stream), an = ctx.createAnalyser(); an.fftSize = 1024;
    src.connect(an);
    this.micStream = stream; this.micCtx = ctx; this.analyser = an; this.micBuf = new Float32Array(an.fftSize);
    this.mic = true;
  }

  readMic() {
    if (!this.mic) return;
    this.analyser.getFloatTimeDomainData(this.micBuf);
    let e = 0; for (const x of this.micBuf) e += x * x;
    const db = 10 * Math.log10(e / this.micBuf.length + 1e-12);
    this.micDb = db;
    // The room's own noise floor follows slowly; a sound well above it is "loud".
    this.micFloor += (Math.min(db, this.micFloor + 3) - this.micFloor) * 0.01;
    const over = db - this.micFloor;
    const prev = this.loud;
    this.loud = Math.max(0, Math.min(1, (over - 8) / 30));
    if (this.loud > 0.45 && prev < 0.2) this.startle("voice");
  }

  // ---------------------------------------------------------- the simulated body

  // What a scare does to the simulated player depends on a hidden profile, different every
  // run, so the director has something real to learn.
  simKick(type, amount) {
    const p = this.sim.profile;
    p[type] ??= 0.25 + rand() * 0.9;
    this.sim.arousal = Math.min(1.2, this.sim.arousal + amount * p[type]);
    if (rand() < amount * p[type] * 1.4) this.startle("sim");
  }

  // ---------------------------------------------------------- per frame

  startle(kind) {
    const last = this.startles[this.startles.length - 1];
    if (last !== undefined && this.t - last < 1.2) return;
    this.startles.push(this.t);
    if (this.startles.length > 50) this.startles.shift();
    this.onStartle?.(kind);
  }

  update(dt, ctx) {
    this.t += dt;
    this.readMic();
    if (this.camera) return;
    const s = this.sim;
    s.exertion += ((ctx.sprinting ? 1 : ctx.moving ? 0.25 : 0) - s.exertion) * dt * (ctx.sprinting ? 0.25 : 0.12);
    // The monster near you, a chase, darkness: a steady pull upward. Breathing slowly pulls down.
    const pull = ctx.threat * 0.5 + (ctx.chase ? 0.6 : 0) + (ctx.dark ? 0.04 : 0);
    s.calm = ctx.breathing ? Math.min(1, s.calm + dt * 0.5) : Math.max(0, s.calm - dt);
    const decay = 0.045 + s.calm * 0.12 + (ctx.hidden && ctx.breathing ? 0.08 : 0);
    s.arousal += (pull - s.arousal) * dt * (pull > s.arousal ? 0.35 : decay);
    const target = s.rest + s.arousal * 52 + s.exertion * 30 + Math.sin(this.t * 0.31) * 1.5;
    s.bpm += (target - s.bpm) * dt / 3.5;
    // Report like the camera does: about once a second, a little noisy.
    s.next -= dt;
    if (s.next <= 0) { s.next = 1; this.bpm = s.bpm + (rand() - 0.5) * 2; this.confidence = 0.9; this.settling = false; }
  }
}
