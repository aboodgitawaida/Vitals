// monster.js -- "the Tenant": a figure that is almost a person. Too tall, no face,
// arms too long, and it moves wrong: smooth, then a snap of the head, then smooth again.
// (Perceptual mismatch -- a human body with machine-like motion -- is the best supported
// route to the uncanny; Kätsyri et al. 2015.)
//
// Behaviour has two layers, like Alien: Isolation. The Director (director.js) knows where
// you are and decides when the Tenant is on the floor and roughly where it wanders. The
// Tenant itself only knows what it hears and sees: footsteps, your voice through the
// microphone, doors -- and your heartbeat.

import * as THREE from "three";
import { rand } from "./tex.js";

const SPEED = { stalk: 1.1, investigate: 1.7, search: 1.3, chase: 3.7, leave: 1.6 };

export class Monster {
  constructor(scene, M, level, audio) {
    this.level = level; this.audio = audio;
    this.root = new THREE.Group(); scene.add(this.root);
    this.build(M);
    this.pos = new THREE.Vector3(49, 0, 5); this.yaw = 0;
    this.state = "off"; this.path = null; this.repath = 0; this.target = null;
    this.walk = 0; this.twitch = 0; this.headTarget = new THREE.Euler(); this.stateTime = 0;
    this.lastSeen = null; this.suspicion = 0; this.heard = null; this.noticed = 0;
    this.lair = new THREE.Vector3(49, 0, 5);
    this.setVisible(false);
  }

  build(M) {
    const skin = M.skin.clone(); skin.roughness = 0.75; skin.color.setRGB(0.42, 0.39, 0.38);
    const rag = new THREE.MeshStandardMaterial({ color: 0x15130f, roughness: 0.95, side: THREE.DoubleSide });
    const limb = (r, len) => { const g = new THREE.CapsuleGeometry(r, len, 4, 10); g.translate(0, -len / 2 - r, 0); return new THREE.Mesh(g, skin); };
    const j = (parent, x, y, z) => { const g = new THREE.Group(); g.position.set(x, y, z); parent.add(g); return g; };
    const body = this.body = j(this.root, 0, 0, 0);
    this.hips = j(body, 0, 1.18, 0);
    const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.16, 0.62, 4, 12), skin);
    torso.scale.set(0.82, 1, 0.58); torso.position.y = 0.45; this.hips.add(torso);
    this.chest = j(this.hips, 0, 0.78, 0);
    const ribs = new THREE.Mesh(new THREE.CapsuleGeometry(0.19, 0.18, 4, 12), skin); ribs.scale.set(1.15, 1, 0.6); this.chest.add(ribs);
    this.neck = j(this.chest, 0, 0.22, 0.02);
    const neckM = limb(0.045, 0.22); neckM.rotation.x = Math.PI; neckM.position.y = 0; this.neck.add(neckM);
    this.head = j(this.neck, 0, 0.3, 0.02);
    const skull = new THREE.Mesh(new THREE.SphereGeometry(0.15, 24, 16), skin); skull.scale.set(0.85, 1.3, 0.95); this.head.add(skull);
    // No eyes. Only a mouth, closed until it is very close.
    this.mouth = new THREE.Mesh(new THREE.SphereGeometry(0.05, 16, 10), new THREE.MeshBasicMaterial({ color: 0x050202 }));
    this.mouth.scale.set(0.9, 0.08, 0.3); this.mouth.position.set(0, -0.06, 0.13); this.head.add(this.mouth);
    this.arms = []; this.legs = [];
    for (const s of [-1, 1]) {
      const sh = j(this.chest, s * 0.25, 0.06, 0);
      const up = limb(0.04, 0.5); sh.add(up);
      const el = j(sh, 0, -0.6, 0); const fo = limb(0.035, 0.56); el.add(fo);
      const hand = j(el, 0, -0.66, 0);
      for (let f = 0; f < 4; f++) { const fi = limb(0.012, 0.2); fi.position.x = (f - 1.5) * 0.018; fi.rotation.z = (f - 1.5) * 0.05; hand.add(fi); }
      this.arms.push({ sh, el, s });
      const hip = j(this.hips, s * 0.11, 0, 0);
      const th = limb(0.06, 0.48); hip.add(th);
      const kn = j(hip, 0, -0.58, 0); const sn = limb(0.045, 0.5); kn.add(sn);
      this.legs.push({ hip, kn, s });
    }
    // Ribs under the skin.
    for (let i = 0; i < 5; i++) {
      const rib = new THREE.Mesh(new THREE.TorusGeometry(0.17 - i * 0.008, 0.012, 6, 18, Math.PI * 0.9), skin);
      rib.rotation.set(Math.PI / 2 + 0.25, 0, Math.PI * 0.05); rib.scale.set(1.1, 0.62, 1);
      rib.position.set(0, 0.1 - i * 0.075, 0.015); this.chest.add(rib);
    }
    // Long, wet black hair, hanging over where the face should be.
    const hairMat = new THREE.MeshStandardMaterial({ color: 0x050404, roughness: 0.35, metalness: 0.1 });
    const scalp = new THREE.Mesh(new THREE.SphereGeometry(0.155, 20, 12, 0, Math.PI * 2, 0, Math.PI * 0.6), hairMat);
    scalp.scale.set(0.9, 1.34, 1.0); this.head.add(scalp);
    // A curtain of hair from the crown to the chest, all the way round: strands painted into an
    // alpha mask, ragged at the ends, hanging where the face should be.
    const c = document.createElement("canvas"); c.width = 512; c.height = 512;
    const g = c.getContext("2d");
    g.fillStyle = "#000"; g.fillRect(0, 0, 512, 512);
    for (let k = 0; k < 900; k++) {
      const x = rand() * 512, len = 512 * (0.45 + rand() * 0.55), w = 1 + rand() * 2.5;
      g.strokeStyle = `rgb(${200 + rand() * 55},${200 + rand() * 55},${200 + rand() * 55})`; g.lineWidth = w;
      g.beginPath(); g.moveTo(x, 0);
      g.bezierCurveTo(x + (rand() - 0.5) * 20, len * 0.4, x + (rand() - 0.5) * 30, len * 0.7, x + (rand() - 0.5) * 40, len); g.stroke();
    }
    g.fillStyle = "#fff"; g.fillRect(0, 0, 512, 60);
    const alpha = new THREE.CanvasTexture(c);
    const hairCurtain = new THREE.MeshStandardMaterial({ color: 0x060505, roughness: 0.35, metalness: 0.1, alphaMap: alpha, alphaTest: 0.45, side: THREE.DoubleSide });
    for (const [rt, rb, h, y] of [[0.135, 0.2, 0.95, -0.3], [0.125, 0.17, 0.8, -0.24]]) {
      const geo = new THREE.CylinderGeometry(rt, rb, h, 48, 8, true);
      const pos = geo.attributes.position;
      for (let v = 0; v < pos.count; v++) {           // a little sway and clumping
        const yy = pos.getY(v), t = (h / 2 - yy) / h;
        pos.setX(v, pos.getX(v) * (1 + (rand() - 0.5) * 0.25 * t));
        pos.setZ(v, pos.getZ(v) * (1 + (rand() - 0.5) * 0.25 * t) + t * 0.04);
      }
      geo.computeVertexNormals();
      const m = new THREE.Mesh(geo, hairCurtain); m.position.y = y; m.scale.set(1, 1, 1.1); m.castShadow = true;
      this.head.add(m);
    }
    const cloth = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.32, 0.55, 14, 1, true), rag);
    cloth.position.y = 0.0; this.hips.add(cloth);
    this.root.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    this.voice = this.audio.ctx ? this.audio.voice("monster") : null;
  }

  setVisible(v) { this.root.visible = v; }
  get active() { return this.state !== "off"; }
  get headPos() { return this.pos.clone().setY(2.2); }

  // ---------------------------------------------------------- director controls

  spawn(at, state = "stalk") {
    this.pos.copy(at); this.state = state; this.stateTime = 0; this.path = null; this.suspicion = 0;
    this.setVisible(true);
  }

  despawn() { this.state = "off"; this.setVisible(false); this.path = null; }

  leave() { if (this.active && this.state !== "chase") this.setState("leave"); }

  setState(s) { if (this.state !== s) { this.state = s; this.stateTime = 0; this.path = null; } }

  goTo(p) { this.target = p.clone(); this.path = null; }

  // A noise at `p` the Tenant could hear if it is within `radius`.
  hear(p, radius, kind) {
    if (!this.active || this.state === "scripted") return false;
    const d = this.pos.distanceTo(p);
    if (d > radius) return false;
    this.heard = { p: p.clone(), kind, t: performance.now() };
    if (this.state !== "chase") {
      // It does not hear exactly where: the further, the vaguer.
      const err = d * 0.18;
      this.goTo(new THREE.Vector3(p.x + (rand() - 0.5) * err, 0, p.z + (rand() - 0.5) * err));
      this.setState("investigate");
      this.target = this.heard.p.clone();
    }
    return true;
  }

  // ---------------------------------------------------------- per frame

  // Returns "caught" when it reaches you.
  update(dt, t, player, director) {
    this.stateTime += dt;
    if (!this.voice && this.audio.ctx) this.voice = this.audio.voice("monster");
    if (!this.active) { this.voiceLevel(0); return null; }
    const eye = this.headPos, me = player.pos.clone().setY(1.5);
    const dist = Math.hypot(player.pos.x - this.pos.x, player.pos.z - this.pos.z);

    // Sight: it sees you in a cone in front of it, and it sees your flashlight from further.
    let sees = false;
    if (!player.hidden && this.state !== "scripted") {
      const toP = new THREE.Vector3(player.pos.x - this.pos.x, 0, player.pos.z - this.pos.z).normalize();
      const fwd = new THREE.Vector3(Math.sin(this.yaw), 0, Math.cos(this.yaw));
      const inCone = toP.dot(fwd) > 0.35 || dist < 2.2;
      const range = player.flashlight ? 22 : (player.crouch ? 6 : 10);
      if (inCone && dist < range && this.level.lineOfSight(eye, me)) sees = true;
    }
    this.noticed = sees ? this.noticed + dt : Math.max(0, this.noticed - dt * 0.5);
    if (sees && this.noticed > (this.state === "chase" ? 0 : 0.35)) {
      this.lastSeen = player.pos.clone();
      if (this.state !== "chase") { this.setState("chase"); director.onChase(); }
    }

    switch (this.state) {
      case "stalk":
        if (!this.target || this.arrived()) this.target = director.stalkTarget(this);
        break;
      case "investigate":
        if (this.arrived()) { this.setState("search"); this.searchUntil = 4 + rand() * 4; }
        break;
      case "search":
        if (this.stateTime > this.searchUntil) this.setState("stalk");
        else if (!this.target || this.arrived()) this.target = this.level.randomFloorNear(this.pos, 1, 5) ?? this.pos.clone();
        break;
      case "chase":
        if (sees) { this.target = player.pos.clone(); }
        else if (this.lastSeen) {
          this.target = this.lastSeen.clone();
          if (this.arrived() || this.stateTime > 30) { this.setState("search"); this.searchUntil = 8; director.onLost(); }
        }
        if (sees) this.stateTime = 0;
        if (!sees && this.stateTime > 7) { this.setState("search"); this.searchUntil = 8; director.onLost(); }
        break;
      case "leave":
        this.target = this.lair;
        if (this.pos.distanceTo(this.lair) < 1.2 || (this.stateTime > 25 && !player.canSee(this.headPos))) this.despawn();
        break;
      case "scripted":
        break;
    }

    // Hidden in a locker: it listens. A calm heart is nearly silent; a racing one is not.
    if (player.hidden && this.state !== "scripted") {
      const r = director.heartRadius();
      const dl = this.pos.distanceTo(player.hidden.pos);
      if (dl < Math.max(r, 1.4) && dl < 7) {
        this.suspicion += dt * (0.18 + 0.5 * Math.max(0, r - dl) / Math.max(1, r));
        if (this.suspicion > 0.3) { this.target = player.hidden.pos.clone().addScaledVector(player.hidden.face, 1.0); this.path = null; }
      } else this.suspicion = Math.max(0, this.suspicion - dt * 0.15);
      if (this.suspicion >= 1) return "found";
    } else this.suspicion = Math.max(0, this.suspicion - dt * 0.3);

    this.move(dt, t);
    this.animate(dt, t, dist);
    const close = Math.max(0, 1 - dist / 14);
    this.voiceLevel(this.state === "scripted" ? 0.6 : 0.25 + close * 0.6 + (this.state === "chase" ? 0.3 : 0));
    if (this.voice) this.audio.place(this.voice.pan, eye);

    if (this.state === "chase" && dist < 1.05 && !player.hidden) return "caught";
    return null;
  }

  voiceLevel(v) { if (this.voice) this.voice.gain.gain.setTargetAtTime(v * 0.5, this.audio.now, 0.3); }

  arrived() { return this.target && Math.hypot(this.target.x - this.pos.x, this.target.z - this.pos.z) < 0.7; }

  move(dt, t) {
    if (!this.target || this.state === "scripted") return;
    this.repath -= dt;
    if (!this.path || this.repath <= 0) {
      this.path = this.level.path(this.pos, this.target) ?? [];
      this.repath = this.state === "chase" ? 0.35 : 1.5;
    }
    const next = this.path[0];
    if (!next) return;
    const dx = next.x - this.pos.x, dz = next.z - this.pos.z, d = Math.hypot(dx, dz);
    if (d < 0.3) { this.path.shift(); return; }
    // Doors: it opens them (loudly) rather than walking through.
    const door = this.level.doorAt.get(`${Math.floor(next.x / 2)},${Math.floor(next.z / 2)}`);
    if (door && !door.open && !door.locked) {
      this.level.setDoor(door, true);
      this.audio.creak(door.center, 0.8);
    }
    // Moves in surges: a few fast steps, a stop, a few more. Never quite like a person.
    const surge = this.state === "chase" ? 1 : 0.55 + 0.45 * Math.max(0, Math.sin(t * 2.3 + Math.sin(t * 0.7) * 2));
    const sp = SPEED[this.state] ?? 1.2;
    const step = Math.min(d, sp * surge * dt);
    this.pos.x += (dx / d) * step; this.pos.z += (dz / d) * step;
    const want = Math.atan2(dx, dz);
    let dy = want - this.yaw; while (dy > Math.PI) dy -= Math.PI * 2; while (dy < -Math.PI) dy += Math.PI * 2;
    this.yaw += dy * Math.min(1, dt * (this.state === "chase" ? 10 : 4));
    this.speedNow = step / Math.max(dt, 1e-4);
  }

  animate(dt, t, dist) {
    const v = this.speedNow ?? 0;
    this.walk += dt * v * 2.4;
    const w = Math.sin(this.walk), chase = this.state === "chase";
    this.root.position.copy(this.pos);
    this.root.rotation.y = this.yaw;
    // Too tall for the doorways: it folds itself through them.
    const inDoor = this.level.doorAt.has(`${Math.floor(this.pos.x / 2)},${Math.floor(this.pos.z / 2)}`);
    this.duck = (this.duck ?? 0) + ((inDoor ? 1 : 0) - (this.duck ?? 0)) * Math.min(1, dt * 5);
    this.body.position.y = Math.abs(Math.cos(this.walk)) * 0.04 * Math.min(1, v) - this.duck * 0.3;
    this.hips.rotation.x = (chase ? 0.45 : 0.12) + this.duck * 0.35;
    this.chest.rotation.x = (chase ? 0.25 : 0.18 + Math.sin(t * 0.8) * 0.03) + this.duck * 0.5;
    for (const L of this.legs) {
      L.hip.rotation.x = w * 0.55 * L.s * Math.min(1, v) - (chase ? 0.4 : 0.1);
      L.kn.rotation.x = Math.max(0, -w * L.s) * 0.9 * Math.min(1, v) + (chase ? 0.4 : 0.1);
    }
    for (const A of this.arms) {
      A.sh.rotation.x = -w * 0.35 * A.s * Math.min(1, v) + (chase ? -0.9 : 0.05);
      A.sh.rotation.z = A.s * (0.08 + Math.sin(t * 1.3 + A.s) * 0.03);
      A.el.rotation.x = chase ? -0.6 : -0.15 - Math.sin(t * 2 + A.s) * 0.05;
    }
    // The head: still, still, then a snap to a new angle.
    this.twitch -= dt;
    if (this.twitch <= 0) {
      this.twitch = 0.3 + rand() * (chase ? 0.6 : 2.4);
      this.headTarget.set((rand() - 0.5) * 0.6, (rand() - 0.5) * 1.2, (rand() - 0.5) * (rand() < 0.2 ? 1.3 : 0.5));
    }
    const k = Math.min(1, dt * 22);
    this.head.rotation.x += (this.headTarget.x - this.head.rotation.x) * k;
    this.head.rotation.y += (this.headTarget.y - this.head.rotation.y) * k;
    this.head.rotation.z += (this.headTarget.z - this.head.rotation.z) * k;
    const open = this.mouthOpen ?? Math.max(0, 1 - dist / 2.5);
    this.mouth.scale.set(0.55 + open * 0.15, 0.06 + open * 1.9, 0.3);
  }

  // Look straight at a point (scripted moments).
  face(p) { this.yaw = Math.atan2(p.x - this.pos.x, p.z - this.pos.z); this.root.rotation.y = this.yaw; this.headTarget.set(0, 0, 0.3); }
}
