// main.js -- STILL: a first-person horror game that listens to your heart.
//
// Wires the pieces together: renderer and post (post.js), the building (level.js), the
// Tenant (monster.js), sound (audio.js), your body (sensors.js) and the Director that
// watches it (director.js). Then the player, the story, the menus.

import * as THREE from "three";
import { makeMaterials, cookieTexture, rand } from "./tex.js";
import { Level, cellOf } from "./level.js";
import { Monster } from "./monster.js";
import { GameAudio } from "./audio.js";
import { Sensors } from "./sensors.js";
import { Director, SCARES } from "./director.js";
import { Post } from "./post.js";

const $ = (id) => document.getElementById(id);
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const EYE = 1.62, CROUCH_EYE = 1.08, RADIUS = 0.28;

// ------------------------------------------------------------------ settings

const DEFAULTS = { volume: 0.9, sensitivity: 1, brightness: 1, showHeart: true, heartAudio: true, safety: true, safetyBpm: 150, quality: "high" };
function loadSettings() {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem("still.settings") || "{}") }; } catch { return { ...DEFAULTS }; }
}
function saveSettings(s) { try { localStorage.setItem("still.settings", JSON.stringify(s)); } catch {} }

// ------------------------------------------------------------------ the story

const NOTES = {
  mgmt: {
    title: "NOTICE TO ALL TENANTS",
    body: "Quiet hours are now in effect at all hours.\n\nDo not run in the halls. Do not raise your voice. If you hear someone walking behind you, keep walking. Do not turn around to check.\n\nIf you must hide, hide well, and be calm. It does not see well.\n\nIt does not need to.\n\n— The Management",
  },
  diary: {
    title: "A page torn from a notebook",
    body: "Day 9. I can hear my heart at night now. Not feel it. Hear it, like it's in the room with me.\n\nLast night something stopped outside my door and listened. I held my breath until I saw stars. It left when my heart slowed down.\n\nI think that's how it hunts. I think it's the only way it hunts.\n\nIf Eli comes looking for me: don't panic. Please don't panic.\n\n— M.",
  },
  log: {
    title: "Manager's log",
    body: "Took the key to 309 back off Ms. Arden's door. She hasn't answered in a week. It's on my desk. I'm not going in there.\n\nSomeone keeps sliding heart monitor printouts under my door. The rhythm is mine. The times are all from when I was asleep.\n\nI've started sleeping in the storage room, 304. It's quiet there.",
  },
};

const INTRO = [
  [0, "Halvorsen Apartments. Third floor."],
  [3.6, "Mira's last voicemail came from Room 309."],
  [7.4, "“It's so quiet here, Eli. Quiet enough to hear your own heart.”"],
  [11.6, "“And it can hear it too.”"],
];

const TAPE = [
  [0, "Eli. If you're listening to this, you came. I knew you would."],
  [4.5, "Don't run. Running is loud. Everything is loud here."],
  [8.5, "It can hear your heart. I don't know how. It just can."],
  [12.5, "When it's close, hide. Breathe slow. Four in. Hold. Six out."],
  [17.0, "The stairwell door opens now. It opens for whoever hears this."],
  [21.5, "I'm sorry. I'm so sorry. It's standing right behind m—"],
];

// ------------------------------------------------------------------ the player

class Player {
  constructor(game) {
    this.g = game;
    this.pos = new THREE.Vector3(5, 0, 5); this.yaw = Math.PI; this.pitch = 0;
    this.vel = new THREE.Vector3(); this.crouch = false; this.crouchT = 0; this.stamina = 1;
    this.hidden = null; this.flashlight = true; this.bob = 0; this.stepAcc = 0; this.speed = 0;
    this.sprinting = false; this.breathing = false;
  }
  forward() { return new THREE.Vector3(-Math.sin(this.yaw), 0, -Math.cos(this.yaw)); }
  look() { return new THREE.Vector3(-Math.sin(this.yaw) * Math.cos(this.pitch), Math.sin(this.pitch), -Math.cos(this.yaw) * Math.cos(this.pitch)); }
  eye() { return this.g.camera.position.clone(); }
  // In the field of view and not behind a wall.
  canSee(p) {
    const e = this.eye(), d = p.clone().sub(e), len = d.length();
    if (len < 0.01) return true;
    if (d.normalize().dot(this.look()) < Math.cos(THREE.MathUtils.degToRad(this.g.camera.fov * 0.62))) return false;
    return this.g.level.lineOfSight(e, p);
  }
}

// ------------------------------------------------------------------ HUD

class Hud {
  constructor(game) {
    this.g = game; this.ecg = $("ecg").getContext("2d"); this.trace = new Float32Array(220); this.beatAt = -1; this.x = 0;
    this.subs = []; this.toastT = 0;
  }
  objective(text) { const el = $("objective"); el.textContent = text; el.classList.remove("show"); void el.offsetWidth; el.classList.add("show"); }
  prompt(text) { $("prompt").textContent = text || ""; $("prompt").classList.toggle("on", !!text); }
  hint(text, sec = 4) { $("hint").textContent = text; $("hint").classList.add("on"); clearTimeout(this.hintTimer); this.hintTimer = setTimeout(() => $("hint").classList.remove("on"), sec * 1000); }
  subtitles(lines, onDone) {
    this.subs.forEach(clearTimeout); this.subs = [];
    const el = $("subtitle");
    lines.forEach(([t, text], i) => this.subs.push(setTimeout(() => { el.textContent = text; el.classList.add("on"); if (i === lines.length - 1) this.subs.push(setTimeout(() => { el.classList.remove("on"); onDone?.(); }, 4200)); }, t * 1000)));
  }
  flash(a, color = "#fff") { const f = $("flash"); f.style.background = color; f.style.transition = "none"; f.style.opacity = a; requestAnimationFrame(() => { f.style.transition = "opacity 0.6s"; f.style.opacity = 0; }); }
  glitch(sec) { $("glitch").classList.add("on"); this.g.staticT = sec; setTimeout(() => $("glitch").classList.remove("on"), sec * 1000); }
  beat() { this.beatAt = 0; }
  draw(dt, bpm, simulated) {
    $("vitals").hidden = !this.g.settings.showHeart;
    $("bpm").textContent = bpm === null ? "--" : Math.round(bpm);
    $("src").textContent = simulated ? "SIMULATED" : "CAMERA";
    const g = this.ecg, W = 220, H = 48;
    const adv = Math.max(1, Math.round(dt * 90));
    for (let k = 0; k < adv; k++) {
      let y = 0;
      if (this.beatAt >= 0) {
        const p = this.beatAt; this.beatAt++;
        y = p < 2 ? -0.15 : p < 4 ? 1 : p < 6 ? -0.45 : p < 12 ? 0.12 * Math.sin((p - 6) / 6 * Math.PI) : 0;
        if (p > 14) this.beatAt = -1;
      }
      this.trace[this.x] = y + (rand() - 0.5) * 0.03; this.x = (this.x + 1) % W;
    }
    g.clearRect(0, 0, W, H);
    g.strokeStyle = this.g.monster.state === "chase" ? "#e2584a" : "#9fd8b0"; g.lineWidth = 1.5; g.beginPath();
    for (let i = 0; i < W; i++) {
      const idx = (this.x + i) % W, y = H * 0.62 - this.trace[idx] * H * 0.5;
      if (i === 0) g.moveTo(i, y); else g.lineTo(i, y);
    }
    g.stroke();
    const grd = g.createLinearGradient(0, 0, W, 0); grd.addColorStop(0, "rgba(6,8,7,1)"); grd.addColorStop(0.25, "rgba(6,8,7,0)");
    g.fillStyle = grd; g.fillRect(0, 0, W, H);
  }
}

// ------------------------------------------------------------------ the game

class Game {
  constructor() {
    this.settings = loadSettings();
    this.state = "loading"; this.T = 0; this.keys = {}; this.flags = {};
    this.staticT = 0; this.radioSurge = 0; this.pulse = 0; this.shake = 0; this.red = 0; this.black = 0;
    this.blackoutUntil = 0; this.scripted = null; this.checkpoint = null; this.reading = null;
    this.setupRenderer();
  }

  setupRenderer() {
    const r = this.renderer = new THREE.WebGLRenderer({ canvas: $("game"), antialias: false, powerPreference: "high-performance" });
    r.setPixelRatio(Math.min(devicePixelRatio, this.settings.quality === "high" ? 1.5 : 1));
    r.setSize(innerWidth, innerHeight);
    r.shadowMap.enabled = true; r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.toneMapping = THREE.ACESFilmicToneMapping; r.toneMappingExposure = 1.2;
    r.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x070807);
    this.scene.fog = new THREE.FogExp2(0x0a0c0b, 0.085);
    this.camera = new THREE.PerspectiveCamera(72, innerWidth / innerHeight, 0.05, 90);
    this.scene.add(this.camera);
    this.scene.add(new THREE.HemisphereLight(0x8a9490, 0x1a1612, 0.12));
    // The flashlight: a spot with a projected lens pattern, and a faint bounce near you.
    const f = this.flash = new THREE.SpotLight(0xfff0d8, 70, 32, 0.46, 0.5, 2);
    f.map = cookieTexture();
    f.castShadow = true; f.shadow.mapSize.set(1024, 1024); f.shadow.camera.near = 0.15; f.shadow.camera.far = 32;
    f.shadow.bias = -0.0004; f.shadow.normalBias = 0.03;
    this.scene.add(f, f.target);
    this.bounce = new THREE.PointLight(0xffe8d0, 0.5, 4, 2); this.scene.add(this.bounce);
    this.flashDir = new THREE.Vector3(0, 0, -1);
    this.post = new Post(r, this.scene, this.camera);
    addEventListener("resize", () => {
      r.setSize(innerWidth, innerHeight); this.camera.aspect = innerWidth / innerHeight; this.camera.updateProjectionMatrix(); this.post.resize();
    });
  }

  async load() {
    // Fonts first, so signs and plaques are painted in them.
    try { await document.fonts.load('600 40px "Cormorant Garamond"'); await document.fonts.load('500 40px "IBM Plex Mono"'); } catch {}
    this.M = await makeMaterials((p, name) => {
      $("load-bar").style.width = `${Math.round(p * 85)}%`;
      $("load-text").textContent = `Painting the building… ${name}`;
    });
    $("load-text").textContent = "Building the third floor…";
    await new Promise((r) => setTimeout(r, 0));
    this.level = new Level(this.scene, this.M);
    this.audio = new GameAudio();
    this.monster = new Monster(this.scene, this.M, this.level, this.audio);
    this.sensors = new Sensors();
    this.player = new Player(this);
    this.hud = new Hud(this);
    this.director = new Director(this);
    this.sensors.onStartle = () => this.director.onStartle();
    this.setupDust();
    this.setupInteractions();
    this.bindInput();
    this.bindMenus();
    // Compile shaders now rather than on the first frame of play.
    this.camera.position.set(8, EYE, 12); this.camera.lookAt(30, EYE, 12);
    this.renderer.compile(this.scene, this.camera);
    $("load-bar").style.width = "100%";
    $("loading").hidden = true;
    this.show("title");
    this.state = "title";
    this.last = performance.now();
    requestAnimationFrame((t) => this.frame(t));
  }

  // Dust in the flashlight beam: only visible inside the cone.
  setupDust() {
    const n = 500, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { pos[i * 3] = (rand() - 0.5) * 10; pos[i * 3 + 1] = rand() * 3; pos[i * 3 + 2] = (rand() - 0.5) * 10; }
    const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this.dustU = { uLight: { value: new THREE.Vector3() }, uDir: { value: new THREE.Vector3() }, uOn: { value: 1 }, uTime: { value: 0 }, uCenter: { value: new THREE.Vector3() } };
    const mat = new THREE.ShaderMaterial({
      uniforms: this.dustU, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      vertexShader: `uniform vec3 uCenter; uniform float uTime; varying vec3 vW;
        void main(){ vec3 p = position; p.x += sin(uTime*0.3 + position.y*3.0)*0.3; p.y = mod(p.y + uTime*0.03, 3.0);
          p.xz = mod(p.xz - uCenter.xz + 5.0, 10.0) - 5.0 + uCenter.xz;
          vW = p; vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_PointSize = 2.2 * (3.0 / -mv.z); gl_Position = projectionMatrix * mv; }`,
      fragmentShader: `uniform vec3 uLight, uDir; uniform float uOn; varying vec3 vW;
        void main(){ vec3 d = vW - uLight; float l = length(d); float c = dot(d / l, uDir);
          float a = smoothstep(0.9, 0.97, c) * smoothstep(9.0, 1.0, l) * smoothstep(0.2, 0.8, l) * uOn;
          vec2 q = gl_PointCoord - 0.5; a *= smoothstep(0.5, 0.0, length(q));
          gl_FragColor = vec4(vec3(1.0, 0.95, 0.85) * a * 0.55, a); }`,
    });
    this.dust = new THREE.Points(geo, mat); this.dust.frustumCulled = false; this.scene.add(this.dust);
  }

  // ------------------------------------------------------------ interactions

  setupInteractions() {
    const L = this.level, I = this.inter = [];
    for (const d of L.doors) {
      I.push({
        pos: d.center, r: 2.1,
        label: () => {
          if (d.kind === "Q" && d.locked) return this.flags.key ? "Unlock Room 309" : "Locked";
          if (d.kind === "E" && d.locked) return "Locked";
          if (d.kind === "E") return "Leave";
          return d.open ? "Close door" : "Open door";
        },
        use: () => this.useDoor(d),
      });
    }
    for (const lk of L.lockers) I.push({ pos: lk.pos.clone().addScaledVector(lk.face, 0.45).setY(1.1), r: 1.6, label: () => "Hide", use: () => this.hide(lk) });
    for (const [k, mesh] of Object.entries(L.notes)) I.push({ pos: mesh.position, r: 1.8, label: () => "Read", use: () => this.read(k) });
    I.push({ pos: L.keyMesh.position, r: 1.8, label: () => "Take key", use: () => this.takeKey(), enabled: () => !this.flags.key });
    I.push({ pos: L.tapeMesh.position, r: 1.9, label: () => "Play tape", use: () => this.playTape(), enabled: () => !this.flags.tape && !this.scripted });
  }

  bestInteractable() {
    if (this.player.hidden) return { label: () => "Leave hiding place", use: () => this.unhide() };
    const e = this.camera.position, look = this.player.look();
    let best = null, bestScore = Infinity;
    for (const it of this.inter) {
      if (it.enabled && !it.enabled()) continue;
      const d = it.pos.clone().sub(e), len = d.length();
      if (len > it.r) continue;
      const ang = Math.acos(clamp(d.normalize().dot(look), -1, 1));
      if (ang > 0.55) continue;
      const score = ang + len * 0.15;
      if (score < bestScore) { bestScore = score; best = it; }
    }
    return best;
  }

  useDoor(d) {
    if (d.kind === "Q" && d.locked) {
      if (!this.flags.key) {
        this.audio.thump(this.audio.now, 160, 0.3, this.audio.dry);
        if (!this.flags.triedDoor) { this.flags.triedDoor = true; this.hud.objective("Find the key to 309"); this.hud.hint("Locked. The manager's office might have a key."); }
        return;
      }
      d.locked = false; this.audio.click(); this.hud.objective("Find out what happened to Mira");
    }
    if (d.kind === "E" && d.locked) { this.audio.thump(this.audio.now, 160, 0.3, this.audio.dry); this.hud.hint("It won't open. Not yet."); return; }
    this.level.setDoor(d, !d.open);
    this.audio.creak(d.center, 0.5);
    this.monster.hear(d.center, 9, "door");
  }

  read(k) {
    this.reading = k;
    $("note-title").textContent = NOTES[k].title;
    $("note-body").textContent = NOTES[k].body;
    $("note").hidden = false;
    this.audio.click();
    if (k === "log" && !this.flags.key) this.hud.objective("Take the key to 309");
  }
  closeNote() { this.reading = null; $("note").hidden = true; }

  takeKey() {
    this.flags.key = true; this.level.keyMesh.visible = false; this.audio.pickup();
    this.hud.objective("Open Room 309");
    this.hud.hint("Key to 309 taken.");
    this.save("key");
  }

  hide(lk) {
    const P = this.player;
    P.hidden = lk; P.wasFlash = P.flashlight; P.flashlight = false;
    lk.doors.forEach((d) => (d.visible = false));
    P.yaw = Math.atan2(-lk.face.x, -lk.face.z); P.pitch = 0; P.lockerYaw = P.yaw;
    $("locker").classList.add("on");
    this.audio.creak(lk.pos.clone().setY(1), 0.4);
    if (!this.flags.hidOnce) { this.flags.hidOnce = true; this.hud.hint("Breathe slowly. It can hear your heart.", 6); }
  }

  unhide() {
    const P = this.player, lk = P.hidden;
    P.hidden = null; P.flashlight = P.wasFlash ?? true;
    lk.doors.forEach((d) => (d.visible = true));
    P.pos.copy(lk.pos).addScaledVector(lk.face, 0.75); P.pos.y = 0;
    $("locker").classList.remove("on");
    this.audio.creak(lk.pos.clone().setY(1), 0.5);
    this.monster.hear(lk.pos, 6, "locker");
  }

  // ------------------------------------------------------------ story beats

  playTape() {
    this.flags.tapeStarted = true;
    this.scripted = "tape";
    const led = this.level.tapeMesh.userData.led; led.material.emissiveIntensity = 3;
    this.audio.click();
    this.monster.despawn();
    this.director.setPhase("relief");
    this.audio.setAmbient(0.35);
    const hiss = this.audio.voice("tv"); this.audio.place(hiss.pan, this.level.tapeMesh.position); hiss.gain.gain.value = 0.05;
    for (const [t] of TAPE) setTimeout(() => this.audio.whisper(this.level.tapeMesh.position, 2.6, 0.35), t * 1000 + 200);
    this.hud.subtitles(TAPE.map(([t, s]) => [t, `MIRA: ${s}`]));
    setTimeout(() => {
      hiss.gain.gain.value = 0; hiss.stop(); led.material.emissiveIntensity = 0;
      this.flags.tape = true;
      const exit = this.level.doors.find((d) => d.kind === "E"); exit.locked = false;
      this.audio.sting(1.1); this.hud.flash(0.5); this.blackout(3.2, true);
      this.hud.objective("Get to the stairwell");
      // It is in the doorway. The only way out is past it, so: hide.
      const door = this.level.doorAt.get("14,7");
      setTimeout(() => {
        this.monster.spawn(new THREE.Vector3(door.x, 0, door.z + 0.6), "scripted");
        this.monster.face(this.player.pos);
        this.hud.hint("Hide. Now.", 4);
        setTimeout(() => {
          this.monster.setState("search"); this.monster.searchUntil = 22; this.scripted = null;
          this.director.setPhase("peak"); this.director.phaseLen = 30;
          this.save("tape");
        }, 4200);
      }, 2400);
    }, 24500);
  }

  end() {
    this.state = "end"; this.flags.ended = true;
    document.exitPointerLock?.();
    const r = this.director.report();
    const fmt = (x) => (x === null || x === undefined ? "--" : Math.round(x));
    const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
    $("end-source").textContent = this.sensors.simulated ? "Simulated heart (no camera). Play with a camera for your real profile." : "Measured by your camera, with Vitals.";
    $("end-stats").innerHTML = `
      <div><b>${fmt(r.baseline)}</b><span>resting BPM</span></div>
      <div><b>${fmt(r.avgBpm)}</b><span>average BPM</span></div>
      <div><b>${fmt(r.peakBpm)}</b><span>peak BPM at ${mmss(r.peakAt)}</span></div>
      <div><b>${r.startles}</b><span>times you flinched</span></div>`;
    $("end-peak").textContent = r.peakCause ? `Your heart peaked during: ${r.peakCause}.` : "";
    $("end-top").innerHTML = r.top.length ? r.top.map((x, i) => `<li><span>${i + 1}</span>${x.label}<em>${Math.round(clamp(x.score / 0.8) * 100)}%</em></li>`).join("") : "<li>Not enough scares to tell.</li>";
    $("end-least").textContent = r.least ? `Didn't work on you: ${r.least.label.toLowerCase()}.` : "";
    this.audio.setChase(0); this.audio.setAmbient(0.2); this.audio.setHeart(70, 0);
    this.hideAll(); this.show("end");
  }

  // ------------------------------------------------------------ scripted scares (director calls these)

  flickerLights(sec) { for (const L of this.level.lights) if (L.light.position.distanceTo(this.player.pos) < 16) L.flickerUntil = this.T + sec; this.audio.thump(this.audio.now, 220, 0.12, this.audio.dry); }

  blackout(sec, keepFlash = false) {
    this.blackoutUntil = this.T + sec;
    if (!keepFlash) this.flashDeadUntil = this.T + Math.min(1.6, sec * 0.4);
    this.audio.thump(this.audio.now, 40, 0.7, this.audio.dry);
  }

  borrowMonster() {
    const m = this.monster;
    this.borrowed = m.active ? { pos: m.pos.clone(), state: m.state } : null;
  }
  returnMonster() {
    const b = this.borrowed, m = this.monster; this.borrowed = null; m.mouthOpen = null;
    if (b) { m.pos.copy(b.pos); m.setState(b.state === "scripted" ? "stalk" : b.state); } else m.despawn();
  }

  glimpse(q) {
    if (this.monster.active && this.player.canSee(this.monster.headPos)) return;
    this.borrowMonster();
    this.monster.spawn(q, "scripted"); this.monster.face(this.player.pos);
    this.scripted = { kind: "glimpse", t: 0, until: 1.4 + rand() * 1.2, seen: 0 };
    this.audio.setRadio(0.8); this.radioSurge = 1.2;
  }

  behind(q) {
    this.borrowMonster();
    this.monster.spawn(q, "scripted"); this.monster.face(this.player.pos);
    this.scripted = { kind: "behind", t: 0, until: 6, seen: 0 };
    this.audio.whisper(q.clone().setY(2), 1.2, 0.5);
  }

  lunge() {
    this.borrowMonster();
    const f = this.player.look(), e = this.camera.position;
    const q = e.clone().addScaledVector(f, 0.85); q.y = 0;
    this.monster.spawn(q, "scripted"); this.monster.face(e); this.monster.mouthOpen = 1;
    this.monster.root.position.copy(q); this.monster.root.position.y = e.y - 2.25;
    this.scripted = { kind: "lunge", t: 0, until: 0.45 };
    this.audio.sting(1.3); this.hud.flash(0.35, "#300"); this.shake = 0.6; this.red = 0.4;
  }

  updateScripted(dt) {
    const s = this.scripted;
    if (!s || typeof s === "string") return;
    s.t += dt;
    const m = this.monster, sees = this.player.canSee(m.headPos.clone().setY(1.6));
    if (s.kind === "glimpse") {
      // It is there until you look straight at it with the light, or it decides to go.
      const center = this.player.look().dot(m.headPos.clone().setY(1.6).sub(this.camera.position).normalize());
      if (sees && center > 0.985 && this.player.flashlight) s.seen += dt;
      if (s.seen > 0.25 || s.t > s.until) { this.flickerLights(0.6); this.scripted = null; this.returnMonster(); }
    } else if (s.kind === "behind") {
      if (sees) s.seen += dt;
      if (s.seen > 0.12 && !s.stung) { s.stung = true; s.until = s.t + 0.35; this.audio.sting(0.8); this.hud.flash(0.25); this.director.onStartle?.call(this.director); }
      if (s.t > s.until) { if (!s.stung) this.audio.whisper(m.headPos, 1.4, 0.6); this.scripted = null; this.returnMonster(); this.blackout(0.5); }
    } else if (s.kind === "lunge") {
      const e = this.camera.position, f = this.player.look();
      m.root.position.copy(e).addScaledVector(f, 0.8 - s.t * 0.4); m.root.position.y = e.y - 2.25 + Math.sin(s.t * 40) * 0.02;
      m.face(e); m.root.position.y = e.y - 2.25;
      if (s.t > s.until) { this.scripted = null; this.returnMonster(); this.blackout(1.4); }
    }
  }

  // ------------------------------------------------------------ death and checkpoints

  save(name) { this.checkpoint = { name, pos: this.player.pos.clone(), yaw: this.player.yaw, flags: { ...this.flags } }; }

  die(how) {
    if (this.state !== "play") return;
    this.state = "dying"; this.dyingT = 0; this.deathHow = how;
    const P = this.player;
    if (P.hidden) { const lk = P.hidden; this.monster.pos.copy(lk.pos).addScaledVector(lk.face, 0.75); }
    this.monster.setState("scripted"); this.monster.mouthOpen = 1;
    this.audio.sting(1.4); this.audio.setChase(0); this.shake = 0.8; this.red = 0.6;
    this.dyingBpm = this.sensors.bpm;
  }

  updateDying(dt) {
    this.dyingT += dt;
    const head = this.monster.headPos, e = this.camera.position;
    const want = head.clone().sub(e); const yaw = Math.atan2(-want.x, -want.z), pitch = Math.atan2(want.y, Math.hypot(want.x, want.z));
    this.player.yaw += (yaw - this.player.yaw) * Math.min(1, dt * 8); this.player.pitch += (pitch - this.player.pitch) * Math.min(1, dt * 8);
    this.monster.face(e);
    this.black = clamp((this.dyingT - 0.9) / 0.8);
    this.applyCamera(dt);
    if (this.dyingT > 1.9) {
      this.state = "dead"; document.exitPointerLock?.();
      $("dead-title").textContent = this.deathHow === "found" ? "IT HEARD YOUR HEART" : "IT CAUGHT YOU";
      $("dead-stats").textContent = this.dyingBpm ? `Your heart was at ${Math.round(this.dyingBpm)} BPM.` : "";
      this.show("dead");
    }
  }

  respawn() {
    const cp = this.checkpoint, P = this.player;
    if (P.hidden) this.unhide();
    this.flags = { ...cp.flags };
    P.pos.copy(cp.pos); P.yaw = cp.yaw; P.pitch = 0; P.vel.set(0, 0, 0);
    this.monster.despawn(); this.monster.mouthOpen = null; this.scripted = null;
    this.level.keyMesh.visible = !this.flags.key;
    this.director.setPhase("relief"); this.director.phaseLen = 30;
    this.black = 0; this.red = 0;
    this.hideAll(); this.show("hud", true); this.state = "play"; this.lockPointer();
  }

  safetyPause() {
    if (this.state !== "play") return;
    this.state = "break"; document.exitPointerLock?.();
    this.audio.setChase(0); this.audio.setAmbient(0.15);
    $("break-bpm").textContent = Math.round(this.sensors.bpm);
    this.show("break");
  }

  // ------------------------------------------------------------ flow

  show(id, keep) { if (!keep) for (const el of document.querySelectorAll(".screen")) el.hidden = true; $(id).hidden = false; }
  hideAll() { for (const el of document.querySelectorAll(".screen")) el.hidden = true; $("hud").hidden = true; }

  async begin(mode) {
    this.audio.init(); this.audio.setVolume(this.settings.volume);
    this.monster.voice ??= this.audio.voice("monster");
    this.audio.onBeat = () => { this.hud.beat(); this.pulse = 1; };
    this.show("calibrate");
    $("cal-status").textContent = "";
    if (mode !== "sim") {
      try {
        if (mode === "cam+mic") this.sensors.startMic().catch((e) => console.warn("no microphone", e));
        await this.sensors.startCamera((t) => ($("cal-status").textContent = t));
      } catch (err) {
        console.warn(err);
        $("cal-status").textContent = `Camera unavailable (${err.message || err}). Using a simulated heart.`;
        this.sensors.camera = false;
        await new Promise((r) => setTimeout(r, 2200));
      }
    }
    await this.calibrate();
    this.startPlay();
  }

  // Sit still: your resting heart rate is the baseline everything else is measured from.
  async calibrate() {
    const S = this.sensors, need = S.simulated ? 4 : 14, t0 = performance.now(), readings = [];
    $("cal-title").textContent = "Sit still. Breathe normally.";
    $("cal-sub").textContent = S.simulated ? "Simulating a resting heart…" : "Look at the screen. Measuring your resting heart rate.";
    let good = 0;
    while (true) {
      await new Promise((r) => setTimeout(r, 250));
      S.update(0.25, { threat: 0 });
      const el = (performance.now() - t0) / 1000;
      if (S.bpm !== null && (S.simulated || (!S.settling && S.confidence > 0.5))) { readings.push(S.bpm); good += 0.25; }
      $("cal-bpm").textContent = S.bpm === null ? "--" : Math.round(S.bpm);
      $("cal-conf").textContent = S.simulated ? "simulated" : S.bpm === null ? (S.status || "finding your pulse…") : `${Math.round(S.confidence * 100)}% sure`;
      $("cal-ring").style.setProperty("--p", clamp(good / need));
      if (good >= need) break;
      if (el > 75) { $("cal-status").textContent = "Couldn't get a steady reading. Check the lighting, or carry on with a simulated heart."; break; }
    }
    if (readings.length) { readings.sort((a, b) => a - b); this.director.baseline = readings[(readings.length / 2) | 0]; }
    else { this.sensors.camera = false; this.director.baseline = this.sensors.sim.rest; }
  }

  startPlay() {
    this.hideAll(); this.show("hud", true);
    this.state = "play";
    this.player.pos.set(5, 0, 5); this.player.yaw = Math.PI; this.player.pitch = -0.05;
    this.save("start");
    this.black = 1;
    this.hud.subtitles(INTRO, () => this.hud.objective("Find Room 309"));
    setTimeout(() => this.hud.hint("WASD move · Shift run · Ctrl crouch · F flashlight · E interact · hold Space to breathe slowly", 9), 15000);
    this.lockPointer();
  }

  lockPointer() { try { const p = $("game").requestPointerLock?.(); p?.catch?.(() => {}); } catch {} }

  pause() {
    if (this.state !== "play") return;
    this.state = "paused"; this.show("pause"); this.audio.ctx?.suspend();
  }
  resume() {
    this.hideAll(); this.show("hud", true); this.state = "play"; this.audio.ctx?.resume(); this.lockPointer();
  }

  // ------------------------------------------------------------ input

  bindInput() {
    addEventListener("keydown", (e) => {
      this.keys[e.code] = true;
      if (e.code === "Tab") e.preventDefault();
      if (this.state === "play") {
        if (this.reading && (e.code === "KeyE" || e.code === "Escape")) { this.closeNote(); return; }
        if (e.code === "KeyE") this.bestInteractable()?.use();
        if (e.code === "KeyF" && !this.player.hidden) { this.player.flashlight = !this.player.flashlight; this.audio.click(); }
        if (e.code === "Backquote" || e.code === "F3") { e.preventDefault(); $("debug").hidden = !$("debug").hidden; }
      } else if (this.state === "paused" && e.code === "Escape") this.resume();
    });
    addEventListener("keyup", (e) => { this.keys[e.code] = false; });
    addEventListener("mousemove", (e) => {
      if (document.pointerLockElement !== $("game") || this.state !== "play") return;
      const s = 0.0022 * this.settings.sensitivity, P = this.player;
      P.yaw -= e.movementX * s; P.pitch = clamp(P.pitch - e.movementY * s, -1.45, 1.45);
      if (P.hidden) { let d = P.yaw - P.lockerYaw; d = clamp(d, -0.55, 0.55); P.yaw = P.lockerYaw + d; P.pitch = clamp(P.pitch, -0.35, 0.3); }
    });
    $("game").addEventListener("click", () => { if (this.state === "play") { if (this.reading) this.closeNote(); else this.lockPointer(); } });
    $("note").addEventListener("click", () => this.closeNote());
    document.addEventListener("pointerlockchange", () => {
      if (document.pointerLockElement === $("game")) this.hadLock = true;
      else if (this.hadLock && this.state === "play") this.pause();
    });
    addEventListener("blur", () => { this.keys = {}; });
  }

  bindMenus() {
    const on = (id, fn) => $(id).addEventListener("click", fn);
    on("btn-new", () => this.show("consent"));
    on("btn-settings", () => { this.settingsBack = "title"; this.show("settings"); });
    on("btn-controls", () => { this.controlsBack = "title"; this.show("controls"); });
    on("btn-about", () => this.show("about"));
    for (const b of document.querySelectorAll("[data-back]")) b.addEventListener("click", () => this.show(b.dataset.back === "auto" ? (this.state === "paused" ? "pause" : "title") : b.dataset.back));
    const agree = $("agree");
    const sync = () => { for (const b of document.querySelectorAll(".mode")) b.disabled = !agree.checked; };
    agree.addEventListener("change", sync); sync();
    for (const b of document.querySelectorAll(".mode")) b.addEventListener("click", () => this.begin(b.dataset.mode));
    on("btn-resume", () => this.resume());
    on("btn-pause-settings", () => this.show("settings"));
    on("btn-checkpoint", () => { this.audio.ctx?.resume(); this.respawn(); });
    on("btn-quit", () => location.reload());
    on("btn-retry", () => this.respawn());
    on("btn-dead-quit", () => location.reload());
    on("btn-end-again", () => location.reload());
    on("btn-break-resume", () => { this.director.safetyFor = 0; this.director.setPhase("relief"); this.resume(); });
    on("btn-break-quit", () => location.reload());
    // Settings.
    const S = this.settings;
    const bind = (id, key, conv = Number) => {
      const el = $(id);
      if (el.type === "checkbox") el.checked = S[key]; else el.value = S[key];
      el.addEventListener("input", () => {
        S[key] = el.type === "checkbox" ? el.checked : conv(el.value); saveSettings(S);
        if (key === "volume") this.audio?.setVolume(S.volume);
        $(`${id}-v`) && ($(`${id}-v`).textContent = el.value);
      });
      $(`${id}-v`) && ($(`${id}-v`).textContent = el.value);
    };
    bind("set-volume", "volume"); bind("set-sens", "sensitivity"); bind("set-bright", "brightness");
    bind("set-heart", "showHeart"); bind("set-heartaudio", "heartAudio"); bind("set-safety", "safety"); bind("set-safetybpm", "safetyBpm");
  }

  // ------------------------------------------------------------ per frame

  frame(now) {
    requestAnimationFrame((t) => this.frame(t));
    const dt = Math.min(0.05, (now - this.last) / 1000); this.last = now;
    if (this.state === "paused" || this.state === "break") { this.post.render(this.T); return; }
    this.T += dt;
    if (this.state === "play") this.updatePlay(dt);
    else if (this.state === "dying") { this.updateDying(dt); this.monster.animate(dt, this.T, 0.5); }
    else if (this.state === "title" || this.state === "loading") this.updateTitle(dt);
    this.level.update(dt, this.T);
    this.updateLights(dt);
    this.updateTV();
    this.updateGrade(dt);
    this.post.render(this.T);
  }

  updateTitle(dt) {
    // A slow drift down the corridor, the flashlight on, towards the exit sign.
    const t = this.T * 0.05, x = 6 + ((t * 10) % 34);
    this.camera.position.set(x, EYE + Math.sin(this.T * 0.7) * 0.02, 12.1 + Math.sin(this.T * 0.23) * 0.3);
    this.player.yaw = -Math.PI / 2 + Math.sin(this.T * 0.17) * 0.25; this.player.pitch = -0.08 + Math.sin(this.T * 0.11) * 0.05;
    this.camera.rotation.set(0, 0, 0); this.camera.rotation.order = "YXZ";
    this.camera.rotation.y = this.player.yaw; this.camera.rotation.x = this.player.pitch;
    this.updateFlashlight(dt, 1);
  }

  updatePlay(dt) {
    const P = this.player, S = this.sensors, D = this.director, m = this.monster, A = this.audio, k = this.keys;
    // Movement.
    P.breathing = !!k.Space;
    if (!P.hidden && !this.reading && this.scripted !== "tape-locked") {
      P.crouch = !!(k.ControlLeft || k.ControlRight || k.KeyC);
      const want = new THREE.Vector3(), f = P.forward(), r = new THREE.Vector3(-f.z, 0, f.x);
      if (k.KeyW || k.ArrowUp) want.add(f); if (k.KeyS || k.ArrowDown) want.sub(f);
      if (k.KeyD || k.ArrowRight) want.add(r); if (k.KeyA || k.ArrowLeft) want.sub(r);
      const moving = want.lengthSq() > 0;
      P.sprinting = moving && !!(k.ShiftLeft || k.ShiftRight) && !P.crouch && P.stamina > 0.05;
      P.stamina = clamp(P.stamina + (P.sprinting ? -dt / 6 : dt / 9));
      const speed = P.crouch ? 1.05 : P.sprinting ? 3.9 : 2.0;
      if (moving) want.normalize().multiplyScalar(speed);
      P.vel.lerp(want, Math.min(1, dt * (moving ? 9 : 12)));
      P.pos.addScaledVector(P.vel, dt);
      this.level.collide(P.pos, RADIUS);
      P.speed = Math.hypot(P.vel.x, P.vel.z);
      // Footsteps, and how far they carry.
      P.stepAcc += P.speed * dt;
      const stride = P.sprinting ? 1.05 : P.crouch ? 0.55 : 0.75;
      if (P.stepAcc > stride && P.speed > 0.3) {
        P.stepAcc = 0;
        A.footstep(P.sprinting ? 0.5 : P.crouch ? 0.1 : 0.25);
        m.hear(P.pos, P.sprinting ? 13 : P.crouch ? 1.5 : 4.5, "steps");
      }
      if (!this.flags.leftStart && this.level.room(P.pos.x, P.pos.z) !== "301") this.flags.leftStart = true;
    } else { P.speed = 0; P.sprinting = false; }
    P.crouchT += ((P.crouch ? 1 : 0) - P.crouchT) * Math.min(1, dt * 8);

    // Your body.
    const dist = m.active ? m.pos.distanceTo(P.pos) : 99;
    S.update(dt, { sprinting: P.sprinting, moving: P.speed > 0.3, threat: m.active ? clamp(1 - dist / 12) : 0, chase: m.state === "chase", dark: this.T < this.blackoutUntil, breathing: P.breathing, hidden: !!P.hidden });
    D.update(dt);
    // It hears your voice, and your heart.
    this.hearT = (this.hearT ?? 0) - dt;
    if (this.hearT <= 0 && m.active) {
      this.hearT = 1;
      if (S.loud > 0.15) m.hear(P.pos, 4 + S.loud * 22, "voice");
      if (!P.hidden && D.heartRadius() > 0) m.hear(P.pos, D.heartRadius(), "heart");
    }
    const res = m.update(dt, this.T, P, D);
    if (res === "caught" || res === "found") { this.die(res); return; }
    this.updateScripted(dt);

    // Exit.
    if (this.flags.tape && P.pos.x > 54.6) { this.end(); return; }

    // Sound and HUD.
    A.listener(this.camera.position, P.look());
    const radio = clamp(m.active && m.state !== "scripted" ? 1 - dist / 16 : 0) + (this.radioSurge > 0 ? 0.6 : 0);
    this.radioSurge = Math.max(0, this.radioSurge - dt);
    A.setRadio(clamp(radio));
    const bpm = S.bpm ?? D.baseline ?? 70;
    A.setHeart(bpm, this.settings.heartAudio ? (P.hidden ? 0.75 : 0.12 + D.fear * 0.55) : 0);
    if (m.state !== "chase") A.setChase(0);
    this.hud.draw(dt, S.bpm, S.simulated);
    const it = this.reading ? null : this.bestInteractable();
    this.hud.prompt(it ? `[E] ${it.label()}` : "");
    $("crosshair").classList.toggle("on", !!it);
    $("stamina").style.opacity = P.stamina < 0.98 ? 1 : 0; $("stamina-fill").style.width = `${P.stamina * 100}%`;
    $("hearing").classList.toggle("on", !!P.hidden && m.suspicion > 0.05);
    $("hearing").style.setProperty("--s", m.suspicion);
    if (!$("debug").hidden) this.drawDebug();
    this.black = Math.max(0, this.black - dt * 0.5);
    this.applyCamera(dt);
  }

  applyCamera(dt) {
    const P = this.player, c = this.camera;
    if (P.hidden) c.position.copy(P.hidden.eye);
    else {
      P.bob += P.speed * dt * (P.sprinting ? 2.4 : 2.9);
      const amp = Math.min(1, P.speed / 2) * (P.sprinting ? 0.055 : 0.035);
      c.position.set(P.pos.x + Math.cos(P.bob * 0.5) * amp * 0.5, THREE.MathUtils.lerp(EYE, CROUCH_EYE, P.crouchT) + Math.abs(Math.sin(P.bob)) * amp, P.pos.z);
    }
    this.shake = Math.max(0, this.shake - dt * 1.5);
    const sh = this.shake * this.shake * 0.08;
    c.rotation.order = "YXZ";
    c.rotation.y = P.yaw + (rand() - 0.5) * sh; c.rotation.x = P.pitch + (rand() - 0.5) * sh;
    c.rotation.z = Math.sin(P.bob * 0.5) * 0.004 * Math.min(1, P.speed);
    const fovT = 72 + (this.monster.state === "chase" ? 6 : 0) - this.director.fear * 3;
    c.fov += (fovT - c.fov) * Math.min(1, dt * 2); c.updateProjectionMatrix();
    this.updateFlashlight(dt, this.player.flashlight ? 1 : 0);
  }

  updateFlashlight(dt, on) {
    const c = this.camera, f = this.flash;
    const look = new THREE.Vector3(0, 0, -1).applyQuaternion(c.quaternion);
    this.flashDir.lerp(look, Math.min(1, dt * 14)).normalize();
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(c.quaternion);
    f.position.copy(c.position).addScaledVector(right, 0.18).add(new THREE.Vector3(0, -0.12, 0));
    f.target.position.copy(f.position).addScaledVector(this.flashDir, 5);
    let k = on;
    if (this.flashDeadUntil && this.T < this.flashDeadUntil) k = rand() < 0.05 ? 0.3 : 0;
    // It flickers when the Tenant is near.
    const m = this.monster;
    if (m.active && m.state !== "scripted") { const d = m.pos.distanceTo(this.player.pos); if (d < 9 && rand() < (1 - d / 9) * 0.15) k *= rand() * 0.6; }
    f.intensity = 70 * k;
    this.bounce.position.copy(c.position).addScaledVector(this.flashDir, 0.6);
    this.bounce.intensity = 0.45 * k;
    this.dustU.uLight.value.copy(f.position); this.dustU.uDir.value.copy(this.flashDir); this.dustU.uOn.value = k;
    this.dustU.uTime.value = this.T; this.dustU.uCenter.value.copy(c.position);
  }

  updateLights() {
    const T = this.T, blackout = T < this.blackoutUntil, calm = this.director?.lightsCalm;
    for (const L of this.level.lights) {
      let k = 1;
      const n = Math.sin(T * 13.1 + L.phase) * Math.sin(T * 7.3 + L.phase * 2);
      if (L.mode === "buzz") k = calm ? 0.95 : (n > 0.93 ? 0.15 : 0.85 + n * 0.1);
      else if (L.mode === "broken") k = (Math.sin(T * 0.6 + L.phase) > 0.75 && n > 0.2) ? 0.9 : 0.0;
      else if (L.mode === "tv") k = 0.55 + rand() * 0.5;
      if (T < L.flickerUntil) k *= rand() < 0.5 ? 0 : 0.6 + rand() * 0.4;
      if (blackout && L.mode !== "steady-exit" && L.light.color.g < 0.9) k = 0;
      if (this.flags.tape && L.light.color.r > 0.9 && L.light.color.g < 0.5) k *= 1.6;      // 309's red lamp after the tape
      L.light.intensity = L.base * k;
      if (L.bulb) L.bulb.material.color.copy(L.bulb.userData.color).multiplyScalar(4 * k);
    }
  }

  updateTV() {
    this.tvT = (this.tvT ?? 0) + 1;
    if (this.tvT % 3) return;
    for (const tv of this.level.tvs) {
      const g = tv.ctx, img = g.createImageData(96, 72), d = img.data;
      for (let i = 0; i < d.length; i += 4) { const v = rand() * 200; d[i] = v * 0.8; d[i + 1] = v * 0.9; d[i + 2] = v; d[i + 3] = 255; }
      g.putImageData(img, 0, 0);
      tv.tex.needsUpdate = true;
    }
  }

  updateGrade(dt) {
    const u = this.post.u, D = this.director;
    this.pulse = Math.max(0, this.pulse - dt * 4);
    this.staticT = Math.max(0, this.staticT - dt);
    this.red = Math.max(0, this.red - dt * 0.6);
    const m = this.monster, near = m && m.active && m.state !== "scripted" ? clamp(1 - m.pos.distanceTo(this.player.pos) / 6) : 0;
    u.uFear.value = this.state === "title" ? 0.15 : D.fear;
    u.uPulse.value = this.pulse * (this.state === "play" ? 1 : 0);
    u.uStatic.value = clamp((this.staticT > 0 ? 0.6 : 0) + near * near * 0.25 + (this.scripted?.kind === "lunge" ? 0.4 : 0));
    u.uRed.value = this.red;
    u.uBlack.value = this.black;
    u.uBright.value = this.settings.brightness;
  }

  drawDebug() {
    const D = this.director, S = this.sensors, m = this.monster;
    const rows = Object.entries(D.stats).map(([k, s]) => `${k.padEnd(12)} n=${s.n} score=${s.mean.toFixed(2)}`).join("\n");
    $("debug").textContent =
      `fear ${D.fear.toFixed(2)}  heart ${D.hr.toFixed(2)}  startle ${D.spike.toFixed(2)}  slope ${D.fearSlope.toFixed(3)}\n` +
      `bpm ${S.bpm === null ? "--" : S.bpm.toFixed(1)} (${S.simulated ? "sim" : "camera " + Math.round(S.confidence * 100) + "%"})  baseline ${D.baseline?.toFixed(1)}  heard at ${D.heartRadius().toFixed(1)} m\n` +
      `mic ${S.mic ? S.micDb.toFixed(0) + " dB, loud " + S.loud.toFixed(2) : "off"}  face ${S.camera ? (S.face.present ? `eyes ${S.face.eyeWide.toFixed(2)} brow ${S.face.brow.toFixed(2)}` : "not seen") : "off"}\n` +
      `phase ${D.phase} ${D.phaseT.toFixed(0)}/${D.phaseLen.toFixed(0)} s  next scare in ${(D.nextScare - D.t).toFixed(0)} s\n` +
      `tenant ${m.state}  suspicion ${m.suspicion.toFixed(2)}  ${m.active ? m.pos.distanceTo(this.player.pos).toFixed(1) + " m" : ""}\n\n${rows}`;
  }
}

const game = new Game();
window.__still = game;
game.load().catch((err) => { console.error(err); $("load-text").textContent = `Could not start: ${err.message || err}`; });
