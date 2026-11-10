// director.js -- the part of the game that is watching you.
//
// 1. Fear model: one number, 0..1, from heart rate above your own baseline (slow), sudden
//    reactions -- face, voice, flinch (fast) -- and how wide your eyes are.
// 2. Pacing: build -> peak -> relief, over and over, aimed at YOUR sweet spot: recreational
//    fear follows an inverted U (Clasen, Scrivner et al.), so pinning someone at maximum is
//    not the goal; being just below where they would quit is.
// 3. Learning: every scare is a trial. The response to it (how much fear rose in the next
//    12 s, and whether you startled) updates that scare type's score, and repetition wears
//    a scare out (startle habituates). The next scare is picked from what has worked on you
//    and has not been used lately: a bandit with a habituation penalty.
// 4. Timing: unpredictability is the strongest lever there is (stress tracks uncertainty
//    more than the threat itself; de Berker et al. 2016), so intervals are random, and a
//    scare that is due waits for the moment you start to relax.

import * as THREE from "three";
import { rand } from "./tex.js";

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));

export const SCARES = {
  flicker: { label: "Lights failing", kick: 0.22 },
  blackout: { label: "Total darkness", kick: 0.35 },
  whisper: { label: "A whisper behind you", kick: 0.35 },
  slam: { label: "A door slamming", kick: 0.4 },
  knock: { label: "Knocking from inside the walls", kick: 0.3 },
  footsteps: { label: "Footsteps following you", kick: 0.35 },
  phone: { label: "A phone ringing in an empty room", kick: 0.2 },
  glimpse: { label: "A figure at the end of the hall", kick: 0.5 },
  behind: { label: "Something standing behind you", kick: 0.6 },
  doubleheart: { label: "A second heartbeat", kick: 0.3 },
  glitch: { label: "Your own heart monitor failing", kick: 0.25 },
  radio: { label: "The radio, with nothing there", kick: 0.25 },
  lunge: { label: "It, in your face", kick: 0.9 },
};

export class Director {
  constructor(game) {
    this.g = game;
    this.fear = 0; this.hr = 0; this.spike = 0; this.prevFear = 0; this.fearSlope = 0;
    this.baseline = null;
    this.phase = "intro"; this.phaseT = 0; this.phaseLen = 30; this.t = 0;
    this.nextScare = 18 + rand() * 10; this.dueSince = null;
    this.stats = {}; for (const k in SCARES) this.stats[k] = { n: 0, mean: 0.2, uses: [] };
    this.trials = []; this.history = []; this.lunges = 0; this.lastLunge = -999;
    this.highFor = 0; this.safetyFor = 0; this.peakBpm = 0; this.peakAt = 0; this.peakCause = null;
    this.bpmSum = 0; this.bpmN = 0; this.startleCount = 0; this.lastScare = null; this.events = [];
    this.lightsCalm = false;
  }

  onStartle() { this.spike = Math.min(1, this.spike + 0.45); this.startleCount++; for (const tr of this.trials) tr.startled = true; }

  // ---------------------------------------------------------- the fear model

  heartRadius() {
    // How far your heartbeat carries, in metres. A calm heart is silent; +30 BPM is ~9 m.
    const s = this.g.sensors;
    if (s.bpm === null || this.baseline === null) return 0;
    return clamp((s.bpm - this.baseline - 4) * 0.33, 0, 9);
  }

  updateFear(dt) {
    const s = this.g.sensors;
    if (s.bpm !== null && this.baseline !== null) {
      const w = s.camera ? clamp(s.confidence * 1.2) : 1;
      this.hr += (clamp((s.bpm - this.baseline) / 32) - this.hr) * Math.min(1, dt * 0.8 * w);
      this.bpmSum += s.bpm * dt; this.bpmN += dt;
      if (s.bpm > this.peakBpm) { this.peakBpm = s.bpm; this.peakAt = this.t; this.peakCause = this.lastScare && this.t - this.lastScare.t < 20 ? this.lastScare.type : (this.g.monster.state === "chase" ? "chase" : null); }
    }
    this.spike *= Math.exp(-dt / 3);
    const face = s.camera && s.face.present ? clamp((s.face.eyeWide + s.face.brow) * 0.6) : 0;
    const target = clamp(0.62 * this.hr + 0.35 * this.spike + 0.12 * face + (this.g.monster.state === "chase" ? 0.15 : 0));
    this.fear += (target - this.fear) * Math.min(1, dt / 1.2);
    this.fearSlope = this.fearSlope * 0.95 + ((this.fear - this.prevFear) / Math.max(dt, 1e-3)) * 0.05;
    this.prevFear = this.fear;
  }

  // ---------------------------------------------------------- per frame

  update(dt) {
    this.t += dt; this.phaseT += dt;
    this.updateFear(dt);
    this.updateTrials();
    this.safety(dt);
    const g = this.g, m = g.monster;
    if (g.scripted) return;

    // Sweet spot: too high for too long and it backs off; too low and it leans in.
    this.highFor = this.fear > 0.85 && m.state !== "chase" ? this.highFor + dt : 0;
    if (this.highFor > 12 && this.phase !== "relief") this.setPhase("relief");

    switch (this.phase) {
      case "intro":
        if (g.flags.leftStart && this.phaseT > 20) this.setPhase("build");
        break;
      case "build":
        if (!m.active && this.phaseT > 8) this.bringIn();
        if (this.phaseT > this.phaseLen || this.fear > 0.72) this.setPhase("peak");
        break;
      case "peak":
        if (this.phaseT > this.phaseLen && m.state !== "chase") this.setPhase("relief");
        break;
      case "relief":
        if (this.falseRelief && this.phaseT > this.falseRelief) { this.falseRelief = null; this.fire(this.pick(["whisper", "knock", "doubleheart", "flicker"])); }
        if (this.phaseT > this.phaseLen) this.setPhase("build");
        break;
    }
    this.scheduler();
  }

  setPhase(p) {
    this.phase = p; this.phaseT = 0;
    const g = this.g, m = g.monster;
    this.events.push({ t: this.t, phase: p });
    if (p === "build") { this.phaseLen = 40 + rand() * 40; this.lightsCalm = false; g.audio.setAmbient(1); }
    if (p === "peak") {
      this.phaseLen = 20 + rand() * 15;
      const big = this.pick(["glimpse", "behind", "blackout", "lunge", "slam"]);
      if (big) this.fire(big);
      if (m.active && m.state !== "chase") m.hear(g.player.pos, 99, "director");   // it is told roughly where you are
      else if (!m.active) this.bringIn(true);
    }
    if (p === "relief") {
      this.phaseLen = 25 + rand() * 20; this.lightsCalm = true;
      this.falseRelief = rand() < 0.35 ? this.phaseLen * (0.55 + rand() * 0.3) : null;
      m.leave(); g.audio.setAmbient(0.6);
    }
  }

  // The Tenant arrives somewhere you cannot see, far enough away to be fair.
  bringIn(hunting = false) {
    const g = this.g, p = g.player.pos;
    const pts = g.level.patrolPoints().filter((q) => q.distanceTo(p) > 14 && !g.player.canSee(q.clone().setY(1.5)));
    const at = pts[(rand() * pts.length) | 0] ?? g.monster.lair;
    g.monster.spawn(at, "stalk");
    if (hunting) g.monster.hear(p, 99, "director");
  }

  // Where it wanders next. The director knows where you are and leans it your way.
  stalkTarget(monster) {
    const g = this.g;
    if (this.phase === "relief") return monster.lair;
    if (rand() < (this.phase === "peak" ? 0.8 : 0.5)) {
      const q = g.level.randomFloorNear(g.player.pos, this.phase === "peak" ? 3 : 7, this.phase === "peak" ? 9 : 15);
      if (q) return q;
    }
    const pts = g.level.patrolPoints();
    return pts[(rand() * pts.length) | 0];
  }

  onChase() { this.events.push({ t: this.t, chase: true }); this.g.audio.setChase(1); this.g.hud.flash(0.25); }
  onLost() { this.g.audio.setChase(0); }

  // ---------------------------------------------------------- choosing and firing scares

  scheduler() {
    if (this.t < this.nextScare) return;
    if (this.phase === "intro" && !this.g.flags.leftStart) return;
    if (this.g.monster.state === "chase") return;
    // Due. Wait for the moment you relax (fear falling), but not forever.
    this.dueSince ??= this.t;
    const relaxing = this.fearSlope < -0.004, waited = this.t - this.dueSince;
    if (!relaxing && waited < 7) return;
    const pool = this.phase === "relief" ? [] : ["flicker", "whisper", "slam", "knock", "footsteps", "phone", "glimpse", "behind", "doubleheart", "glitch", "radio", "blackout"];
    const type = this.pick(pool);
    if (type) this.fire(type);
    const low = this.fear < 0.25 && this.phaseT > 30;
    this.nextScare = this.t + (low ? 8 : 12) + rand() * (low ? 12 : 22);
    this.dueSince = null;
  }

  // The scare with the best score: what has worked on you, plus curiosity about what has
  // not been tried, minus how often it was used lately, plus noise.
  pick(pool) {
    let best = null, bestScore = -Infinity;
    for (const type of pool) {
      if (!this.available(type)) continue;
      const s = this.stats[type];
      const recent = s.uses.filter((u) => this.t - u < 150).length;
      const score = s.mean + 0.28 / Math.sqrt(s.n + 1) - 0.22 * recent + rand() * 0.18;
      if (score > bestScore) { bestScore = score; best = type; }
    }
    return best;
  }

  available(type) {
    const g = this.g, m = g.monster;
    switch (type) {
      case "lunge": return this.lunges < 2 && this.t - this.lastLunge > 150 && this.t > 120 && !g.player.hidden;
      case "glimpse": return !g.player.hidden && m.state !== "chase" && !!this.glimpseSpot();
      case "behind": return !g.player.hidden && m.state !== "chase" && !!this.behindSpot();
      case "slam": return !!this.slamDoor();
      case "blackout": return !this.lightsCalm;
      default: return true;
    }
  }

  fire(type) {
    const g = this.g, p = g.player.pos, ok = this.run(type);
    if (ok === false) return;
    const s = this.stats[type]; s.uses.push(this.t);
    this.lastScare = { type, t: this.t };
    this.trials.push({ type, t0: this.t, fear0: this.fear, peak: this.fear, startled: false });
    this.history.push({ t: this.t, type, phase: this.phase });
    if (g.sensors.simulated) g.sensors.simKick(type, SCARES[type].kick);
    g.debugLog?.(`scare: ${type}`);
  }

  updateTrials() {
    for (const tr of this.trials) tr.peak = Math.max(tr.peak, this.fear);
    const done = this.trials.filter((tr) => this.t - tr.t0 > 12);
    this.trials = this.trials.filter((tr) => this.t - tr.t0 <= 12);
    for (const tr of done) {
      const r = clamp(tr.peak - tr.fear0 + (tr.startled ? 0.3 : 0), 0, 1.3);
      const s = this.stats[tr.type];
      s.n++; s.mean += (r - s.mean) / Math.min(s.n + 1, 6);    // recent trials count more: you habituate
      tr.response = r;
    }
  }

  // ---------------------------------------------------------- the scares themselves

  run(type) {
    const g = this.g, A = g.audio, P = g.player, fwd = P.forward();
    const behind = (d, side = 0) => P.pos.clone().addScaledVector(fwd, -d).add(new THREE.Vector3(-fwd.z * side, 1.6, fwd.x * side));
    switch (type) {
      case "flicker": g.flickerLights(2.5 + rand() * 2); return;
      case "blackout": g.blackout(5 + rand() * 4); return;
      case "whisper": A.whisper(behind(0.7, rand() < 0.5 ? 0.4 : -0.4), 2.2 + rand()); return;
      case "knock": { const q = g.level.randomFloorNear(P.pos, 2, 5); if (!q) return false; A.knock(q.setY(1.3), 3 + ((rand() * 3) | 0)); return; }
      case "footsteps": {
        let i = 0; const steps = 4 + ((rand() * 3) | 0);
        const iv = setInterval(() => { A.footstep(0.7, behind(2.2 - i * 0.15)); if (++i >= steps) clearInterval(iv); }, 520);
        return;
      }
      case "phone": { const q = g.level.randomFloorNear(P.pos, 5, 12); if (!q) return false; A.phone(q.setY(1), 2); return; }
      case "slam": { const d = this.slamDoor(); d.open = false; d.slamTimer = 0.6; A.slam(d.center); return; }
      case "doubleheart": A.doubleHeart(9, (g.sensors.bpm ?? 75) * 1.13 + 6); return;
      case "glitch": g.hud.glitch(1.4); A.setRadio(1); setTimeout(() => A.setRadio(0), 900); return;
      case "radio": g.radioSurge = 3; return;
      case "glimpse": { const q = this.glimpseSpot(); g.glimpse(q); return; }
      case "behind": { const q = this.behindSpot(); g.behind(q); return; }
      case "lunge": this.lunges++; this.lastLunge = this.t; g.lunge(); return;
    }
  }

  // A point straight ahead, 9-18 m away, in plain sight.
  glimpseSpot() {
    const g = this.g, P = g.player, f = P.forward(), eye = P.eye();
    for (let d = 18; d >= 9; d -= 1.5) {
      const q = P.pos.clone().addScaledVector(f, d);
      const [c, r] = [Math.floor(q.x / 2), Math.floor(q.z / 2)];
      if (g.level.ch(c, r) !== ".") continue;
      if (g.level.lineOfSight(eye, q.clone().setY(1.5))) return q.setY(0);
    }
    return null;
  }

  behindSpot() {
    const g = this.g, P = g.player, f = P.forward();
    const q = P.pos.clone().addScaledVector(f, -2.6);
    const [c, r] = [Math.floor(q.x / 2), Math.floor(q.z / 2)];
    if (g.level.ch(c, r) !== "." || !g.level.lineOfSight(P.eye(), q.clone().setY(1.5))) return null;
    return q.setY(0);
  }

  slamDoor() {
    const g = this.g;
    let best = null;
    for (const d of g.level.doors) {
      if (!d.open || d.kind !== "D") continue;
      const dist = d.center.distanceTo(g.player.pos);
      // Not one you are standing in.
      if (dist > 3 && dist < 14 && (!best || dist < best.center.distanceTo(g.player.pos))) best = d;
    }
    return best;
  }

  // ---------------------------------------------------------- safety

  safety(dt) {
    const s = this.g.sensors, st = this.g.settings;
    if (!st.safety || s.simulated || s.bpm === null) { this.safetyFor = 0; return; }
    this.safetyFor = s.bpm > st.safetyBpm ? this.safetyFor + dt : 0;
    if (this.safetyFor > 8) { this.safetyFor = 0; this.g.safetyPause(); }
  }

  // ---------------------------------------------------------- the end-of-game report

  report() {
    const tried = Object.entries(this.stats).filter(([, s]) => s.n > 0).sort((a, b) => b[1].mean - a[1].mean);
    return {
      top: tried.slice(0, 3).map(([k, s]) => ({ type: k, label: SCARES[k].label, score: s.mean, n: s.n })),
      least: tried.length > 1 ? { type: tried[tried.length - 1][0], label: SCARES[tried[tried.length - 1][0]].label } : null,
      peakBpm: this.peakBpm, peakAt: this.peakAt, peakCause: this.peakCause ? (SCARES[this.peakCause]?.label ?? "Being chased") : null,
      avgBpm: this.bpmN ? this.bpmSum / this.bpmN : null, baseline: this.baseline, startles: this.startleCount,
      scares: this.history.length, duration: this.t,
    };
  }
}
