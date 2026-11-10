// level.js -- the third floor of the Halvorsen Apartments, built from a character grid.
//
//   #  wall          .  floor          D  door          Q  Room 309 (locked)
//   E  stairwell exit (locked until the tape has been played)
//
// Every cell is 2 m square and 3 m high. Walls are full cells, so a doorway is a short
// 2 m vestibule with the door hung at its middle.

import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { textTexture, windowTexture, rand } from "./tex.js";

export const CS = 2, WH = 3;

const MAP = [
  "############################",
  "#...#......#......#........#",
  "#...#......#......#........#",
  "#...#......#......#........#",
  "##D####D######D#######D#####",
  "#..........................E",
  "#..........................#",
  "##D####D######Q#####D####D##",
  "#...#......#.....#......#..#",
  "#...#......#.....#......#..#",
  "#...#......#.....#......#..#",
  "############################",
];

export const ROOMS = [
  { name: "301", c0: 1, c1: 3, r0: 1, r1: 3 },
  { name: "302", c0: 5, c1: 10, r0: 1, r1: 3 },
  { name: "303", c0: 12, c1: 17, r0: 1, r1: 3 },
  { name: "304", c0: 19, c1: 26, r0: 1, r1: 3 },
  { name: "corridor", c0: 1, c1: 26, r0: 5, r1: 6 },
  { name: "305", c0: 1, c1: 3, r0: 8, r1: 10 },
  { name: "office", c0: 5, c1: 10, r0: 8, r1: 10 },
  { name: "309", c0: 12, c1: 16, r0: 8, r1: 10 },
  { name: "310", c0: 18, c1: 23, r0: 8, r1: 10 },
  { name: "closet", c0: 25, c1: 26, r0: 8, r1: 10 },
];

const DOOR_LABELS = { "2,4": "301", "7,4": "302", "14,4": "303", "22,4": "304", "2,7": "305", "7,7": "OFFICE", "14,7": "309", "20,7": "310", "25,7": "", "27,5": "STAIRS" };
const LOCKERS = [[1, 3], [12, 1], [26, 3], [9, 6], [19, 6], [1, 10], [16, 10], [26, 8]];
const OPEN_AT_START = new Set(["2,4", "7,4", "14,4", "7,7", "20,7", "25,7"]);

export const cellCenter = (c, r) => new THREE.Vector3(c * CS + CS / 2, 0, r * CS + CS / 2);
export const cellOf = (x, z) => [Math.floor(x / CS), Math.floor(z / CS)];

export class Level {
  constructor(scene, M) {
    this.scene = scene; this.M = M;
    this.rows = MAP.length; this.cols = MAP[0].length;
    this.grid = MAP.map((row) => row.split(""));
    this.doors = []; this.doorAt = new Map(); this.lockers = []; this.boxes = []; this.lights = [];
    this.interactables = []; this.tvs = []; this.windows = [];
    this.group = new THREE.Group(); scene.add(this.group);
    this.buildShell();
    this.buildDoors();
    this.buildLockers();
    this.furnish();
    this.buildLights();
  }

  ch(c, r) { return c < 0 || r < 0 || c >= this.cols || r >= this.rows ? "#" : this.grid[r][c]; }
  isWall(c, r) { return this.ch(c, r) === "#"; }

  room(x, z) {
    const [c, r] = cellOf(x, z);
    for (const R of ROOMS) if (c >= R.c0 && c <= R.c1 && r >= R.r0 && r <= R.r1) return R.name;
    return "door";
  }

  // ------------------------------------------------------------ building

  add(mesh, shadows = true) {
    mesh.castShadow = shadows; mesh.receiveShadow = true;
    this.group.add(mesh);
    return mesh;
  }

  // A box prop. Rotation is a quarter turn at most, so its footprint stays an AABB.
  box(w, h, d, mat, x, y, z, opts = {}) {
    const m = this.add(new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat), opts.shadow ?? true);
    m.position.set(x, y, z);
    if (opts.rx) m.rotation.x = opts.rx;
    if (opts.ry) m.rotation.y = opts.ry;
    if (opts.rz) m.rotation.z = opts.rz;
    if (opts.collide ?? (y - h / 2 < 0.6 && h > 0.25)) {
      const swap = opts.ry && Math.abs(Math.sin(opts.ry)) > 0.7;
      const hw = (swap ? d : w) / 2, hd = (swap ? w : d) / 2;
      this.boxes.push({ x0: x - hw, x1: x + hw, z0: z - hd, z1: z + hd });
    }
    return m;
  }

  buildShell() {
    const M = this.M;
    const walls = [[], [], []], lino = [], boards = [], ceil = [];
    const isLino = (c, r) => r === 5 || r === 6 || "DQE".includes(this.ch(c, r)) || (r <= 3 && c >= 12 && c <= 17);
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const ch = this.ch(c, r), x = c * CS + CS / 2, z = r * CS + CS / 2;
        if (ch === "#") {
          const seen = !this.isWall(c - 1, r) || !this.isWall(c + 1, r) || !this.isWall(c, r - 1) || !this.isWall(c, r + 1);
          if (!seen) continue;
          const g = new THREE.BoxGeometry(CS, WH, CS); g.translate(x, WH / 2, z);
          walls[(c * 7 + r * 13) % 3].push(g);
          continue;
        }
        const f = new THREE.PlaneGeometry(CS, CS); f.rotateX(-Math.PI / 2); f.translate(x, 0, z);
        (isLino(c, r) ? lino : boards).push(f);
        const k = new THREE.PlaneGeometry(CS, CS); k.rotateX(Math.PI / 2); k.translate(x, WH, z);
        ceil.push(k);
      }
    }
    walls.forEach((list, i) => list.length && this.add(new THREE.Mesh(mergeGeometries(list), M[`wall${i}`]), true));
    this.add(new THREE.Mesh(mergeGeometries(lino), M.lino), false);
    this.add(new THREE.Mesh(mergeGeometries(boards), M.boards), false);
    this.add(new THREE.Mesh(mergeGeometries(ceil), M.ceil), false);
  }

  buildDoors() {
    const M = this.M;
    for (let r = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++) {
      const ch = this.ch(c, r);
      if (!"DQE".includes(ch)) continue;
      const key = `${c},${r}`, x = c * CS + CS / 2, z = r * CS + CS / 2;
      const ax = this.isWall(c - 1, r) && this.isWall(c + 1, r) ? "x" : "z";   // the axis the door panel spans
      // Jambs and lintel fill the rest of the doorway.
      const jamb = (dx, dz, w, d) => {
        this.box(w, WH, d, M.wall0, x + dx, WH / 2, z + dz, { collide: true });
      };
      if (ax === "x") { jamb(-0.75, 0, 0.5, CS); jamb(0.75, 0, 0.5, CS); this.box(1.0, WH - 2.15, CS, M.wall1, x, 2.15 + (WH - 2.15) / 2, z, { collide: false }); }
      else { jamb(0, -0.75, CS, 0.5); jamb(0, 0.75, CS, 0.5); this.box(CS, WH - 2.15, 1.0, M.wall1, x, 2.15 + (WH - 2.15) / 2, z, { collide: false }); }
      // Corridor side: rows 5-6. Rooms north of it open towards -z, rooms south towards +z.
      const corridorSign = ax === "x" ? (r < 5 ? 1 : -1) : -1;
      const pivot = new THREE.Group();
      const panelGeo = ax === "x" ? new THREE.BoxGeometry(1.0, 2.12, 0.06) : new THREE.BoxGeometry(0.06, 2.12, 1.0);
      if (ax === "x") { panelGeo.translate(0.5, 1.06, 0); pivot.position.set(x - 0.5, 0, z); }
      else { panelGeo.translate(0, 1.06, 0.5); pivot.position.set(x, 0, z - 0.5); }
      const panel = new THREE.Mesh(panelGeo, M.door); panel.castShadow = panel.receiveShadow = true;
      pivot.add(panel);
      for (const s of [1, -1]) {
        const knob = new THREE.Mesh(new THREE.SphereGeometry(0.035, 12, 8), M.brass);
        if (ax === "x") knob.position.set(0.88, 1.0, s * 0.06); else knob.position.set(s * 0.06, 1.0, 0.88);
        pivot.add(knob);
      }
      const label = DOOR_LABELS[key];
      if (label) {
        const plaque = new THREE.Mesh(new THREE.PlaneGeometry(0.28, 0.08),
          new THREE.MeshStandardMaterial({ map: textTexture(label, { w: 256, h: 72, bg: "#2a2418", color: "#b89a5a", font: '600 44px "IBM Plex Mono", monospace' }), roughness: 0.4, metalness: 0.6 }));
        if (ax === "x") { plaque.position.set(0.5, 1.62, corridorSign * 0.032); if (corridorSign < 0) plaque.rotation.y = Math.PI; }
        else { plaque.position.set(-0.032, 1.62, 0.5); plaque.rotation.y = -Math.PI / 2; }
        pivot.add(plaque);
      }
      this.group.add(pivot);
      // Opening angle: away from the corridor (into the room); the exit swings into the stairwell.
      const openAngle = ch === "E" ? Math.PI * 0.5 : (ax === "x" ? corridorSign : 1) * Math.PI * 0.47;
      const door = {
        key, c, r, ax, x, z, pivot, kind: ch, label, locked: ch !== "D", open: false, angle: 0, openAngle,
        center: new THREE.Vector3(x, 1.1, z), slamTimer: 0,
      };
      if (OPEN_AT_START.has(key)) { door.open = true; door.angle = openAngle; pivot.rotation.y = openAngle; }
      this.doors.push(door); this.doorAt.set(key, door);
    }
    // The stairwell beyond the exit: a dark shaft lit from far below.
    const shaft = new THREE.Mesh(new THREE.BoxGeometry(3, 6, 3), new THREE.MeshBasicMaterial({ color: 0x050605, side: THREE.BackSide }));
    shaft.position.set(57.5, 0.5, 11); this.group.add(shaft);
  }

  blocking(door) { return Math.abs(door.angle) < 0.55; }

  setDoor(door, open) {
    door.open = open;
  }

  update(dt, t) {
    for (const d of this.doors) {
      const target = d.open ? d.openAngle : 0;
      const speed = d.slamTimer > 0 ? 14 : 3.2;
      d.slamTimer = Math.max(0, d.slamTimer - dt);
      d.angle += (target - d.angle) * Math.min(1, dt * speed);
      d.pivot.rotation.y = d.angle;
    }
  }

  buildLockers() {
    const M = this.M;
    for (const [c, r] of LOCKERS) {
      const p = cellCenter(c, r);
      let face = new THREE.Vector3(1, 0, 0), pos = p.clone();
      if (this.isWall(c - 1, r)) { pos.x = c * CS + 0.32; face.set(1, 0, 0); }
      else if (this.isWall(c + 1, r)) { pos.x = (c + 1) * CS - 0.32; face.set(-1, 0, 0); }
      else if (this.isWall(c, r - 1)) { pos.z = r * CS + 0.32; face.set(0, 0, 1); }
      else { pos.z = (r + 1) * CS - 0.32; face.set(0, 0, -1); }
      const ry = Math.atan2(face.x, face.z);
      const g = new THREE.Group(); g.position.copy(pos); g.rotation.y = ry;
      const body = new THREE.Mesh(new THREE.BoxGeometry(0.9, 2.0, 0.6), M.metal); body.position.y = 1.0;
      const doorL = new THREE.Mesh(new THREE.BoxGeometry(0.42, 1.9, 0.03), M.metal); doorL.position.set(-0.215, 1.0, 0.31);
      const doorR = doorL.clone(); doorR.position.x = 0.215;
      const handle = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.14, 0.03), M.chrome); handle.position.set(-0.03, 1.05, 0.34);
      for (const m of [body, doorL, doorR, handle]) { m.castShadow = m.receiveShadow = true; g.add(m); }
      this.group.add(g);
      const sw = Math.abs(face.x) > 0.5;
      this.boxes.push({ x0: pos.x - (sw ? 0.3 : 0.45), x1: pos.x + (sw ? 0.3 : 0.45), z0: pos.z - (sw ? 0.45 : 0.3), z1: pos.z + (sw ? 0.45 : 0.3) });
      this.lockers.push({ c, r, pos, face, ry, group: g, doors: [doorL, doorR], eye: pos.clone().add(new THREE.Vector3(0, 1.58, 0)).addScaledVector(face, 0.12) });
    }
  }

  chair(x, z, ry, opts = {}) {
    const M = this.M, g = new THREE.Group();
    const part = (w, h, d, px, py, pz) => { const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), M.wood); m.position.set(px, py, pz); m.castShadow = m.receiveShadow = true; g.add(m); };
    part(0.46, 0.05, 0.46, 0, 0.46, 0);
    for (const [lx, lz] of [[-0.2, -0.2], [0.2, -0.2], [-0.2, 0.2], [0.2, 0.2]]) part(0.04, 0.46, 0.04, lx, 0.23, lz);
    part(0.46, 0.5, 0.04, 0, 0.74, -0.21);
    part(0.04, 0.5, 0.04, -0.2, 0.74, -0.21); part(0.04, 0.5, 0.04, 0.2, 0.74, -0.21);
    g.position.set(x, opts.y ?? 0, z); g.rotation.set(opts.rx ?? 0, ry, opts.rz ?? 0);
    this.group.add(g);
    this.boxes.push({ x0: x - 0.26, x1: x + 0.26, z0: z - 0.26, z1: z + 0.26 });
    return g;
  }

  bed(x, z, ry = 0) {
    const M = this.M, along = Math.abs(Math.sin(ry)) > 0.7;
    const W = along ? 2.05 : 1.1, D = along ? 1.1 : 2.05;
    this.box(W, 0.28, D, M.wood, x, 0.24, z, { collide: true });
    this.box(along ? 1.95 : 1.0, 0.2, along ? 1.0 : 1.95, M.cloth, x, 0.48, z, { collide: false });
    const hx = along ? -0.95 : 0, hz = along ? 0 : -0.95;
    this.box(along ? 0.06 : 1.1, 0.9, along ? 1.1 : 0.06, M.wood, x + hx * 1.05, 0.55, z + hz * 1.05, { collide: false });
    this.box(along ? 0.35 : 0.6, 0.12, along ? 0.6 : 0.35, M.cloth, x + hx * 0.75, 0.64, z + hz * 0.75, { collide: false });
    this.box(along ? 1.0 : 1.05, 0.07, along ? 1.05 : 1.0, M.cloth, x - hx * 0.4, 0.61, z - hz * 0.4, { collide: false, ry: 0.08 });
  }

  paper(x, y, z, ry = 0) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(0.21, 0.28), this.M.paper);
    m.rotation.set(-Math.PI / 2, 0, ry); m.position.set(x, y + 0.004, z); m.receiveShadow = true;
    this.group.add(m);
    return m;
  }

  decal(text, x, y, z, ry, w = 3, h = 0.75) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshStandardMaterial({
      map: textTexture(text, { w: 1024, h: 256, smear: true, color: "#4a120c", font: '600 150px "Cormorant Garamond", Georgia, serif' }),
      transparent: true, roughness: 0.4, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2,
    }));
    m.position.set(x, y, z); m.rotation.y = ry; this.group.add(m);
    return m;
  }

  window(x, z, ry) {
    const mat = this.windowMat ??= new THREE.MeshStandardMaterial({ map: windowTexture(), emissive: 0xffffff, emissiveMap: null, emissiveIntensity: 0.0, roughness: 0.2 });
    if (!this.windowMat.emissiveMap) { this.windowMat.emissiveMap = this.windowMat.map; this.windowMat.emissiveIntensity = 0.55; }
    const m = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 1.35), mat);
    m.position.set(x, 1.65, z); m.rotation.y = ry; this.group.add(m);
    this.windows.push(m);
  }

  furnish() {
    const M = this.M;
    const cardboard = new THREE.MeshStandardMaterial({ color: 0x5e4a30, roughness: 0.95, bumpMap: M.cloth.bumpMap, bumpScale: 1 });
    const boxes = (x, z, n) => {
      for (let i = 0; i < n; i++) {
        const s = 0.35 + rand() * 0.3;
        this.box(s, s * 0.8, s, cardboard, x + (rand() - 0.5) * 0.5, s * 0.4 + (i > 1 ? 0.5 : 0), z + (rand() - 0.5) * 0.5, { ry: rand() * 0.6 - 0.3, collide: i < 2 });
      }
    };
    const shelf = (x, z, ry) => {
      const along = Math.abs(Math.sin(ry)) > 0.7, W = along ? 0.4 : 1.6, D = along ? 1.6 : 0.4;
      for (const y of [0.05, 0.6, 1.15, 1.7]) this.box(W, 0.04, D, M.metal, x, y, z, { collide: false });
      for (const [a, b] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) this.box(0.04, 1.8, 0.04, M.metal, x + a * (W / 2 - 0.03), 0.9, z + b * (D / 2 - 0.03), { collide: false });
      this.boxes.push({ x0: x - W / 2, x1: x + W / 2, z0: z - D / 2, z1: z + D / 2 });
      for (let i = 0; i < 7; i++) {
        const s = 0.18 + rand() * 0.2, y = [0.07, 0.62, 1.17, 1.72][i % 4];
        this.box(along ? s : s * 1.4, s, along ? s * 1.4 : s, i % 2 ? cardboard : M.wood, x + (along ? 0 : (rand() - 0.5) * 1.1), y + s / 2, z + (along ? (rand() - 0.5) * 1.1 : 0), { collide: false });
      }
    };

    // 301 -- where you wake up.
    this.bed(2.65, 3.25);
    this.box(0.45, 0.55, 0.4, M.wood, 2.35, 0.275, 4.85, { collide: true });
    this.box(0.08, 0.32, 0.08, M.chrome, 2.35, 0.71, 4.85, { collide: false });
    this.lampShade(2.35, 0.95, 4.85);
    this.notes = { mgmt: this.paper(2.33, 0.55, 4.82, 0.3) };
    this.box(1.0, 0.75, 0.6, M.wood, 6.8, 0.375, 2.4, { collide: true });
    this.chair(6.8, 3.2, Math.PI);
    this.box(0.55, 2.0, 1.0, M.wood, 7.7, 1.0, 6.4, { collide: true });
    this.window(5, 2.01, 0);

    // 302 -- living room, the television left on.
    this.box(2.2, 0.45, 0.85, M.cloth, 14, 0.225, 2.55, { collide: true });
    this.box(2.2, 0.5, 0.2, M.cloth, 14, 0.7, 2.2, { collide: false });
    this.box(0.2, 0.6, 0.85, M.cloth, 12.8, 0.3, 2.55, { collide: false });
    this.box(0.2, 0.6, 0.85, M.cloth, 15.2, 0.3, 2.55, { collide: false });
    this.box(1.2, 0.4, 0.6, M.wood, 14, 0.2, 4.4, { collide: true });
    this.box(1.2, 0.5, 0.45, M.wood, 11.5, 0.25, 7.6, { collide: true });
    this.box(0.7, 0.5, 0.45, M.plastic, 11.5, 0.75, 7.62, { collide: false });
    this.tvScreen(11.5, 0.76, 7.385, Math.PI);
    shelf(21.75, 4.6, Math.PI / 2);
    this.box(0.8, 0.45, 0.8, M.cloth, 19.2, 0.225, 3.2, { collide: true, ry: 0.4 });
    this.window(13, 2.01, 0); this.window(19, 2.01, 0);

    // 303 -- kitchen.
    this.box(6, 0.9, 0.6, M.wood, 29, 0.45, 2.3, { collide: true });
    this.box(6.04, 0.04, 0.64, M.ceramic, 29, 0.92, 2.3, { collide: false });
    this.box(0.5, 0.04, 0.38, M.dark, 28, 0.93, 2.3, { collide: false, shadow: false });
    this.box(4, 0.7, 0.35, M.wood, 29.5, 2.0, 2.18, { collide: false });
    this.box(0.8, 1.8, 0.7, M.ceramic, 35.5, 0.9, 2.4, { collide: true });
    this.box(1.4, 0.06, 0.9, M.wood, 30, 0.75, 5.6, { collide: false });
    for (const [a, b] of [[-0.62, -0.38], [0.62, -0.38], [-0.62, 0.38], [0.62, 0.38]]) this.box(0.05, 0.72, 0.05, M.wood, 30 + a, 0.36, 5.6 + b, { collide: false });
    this.boxes.push({ x0: 29.3, x1: 30.7, z0: 5.15, z1: 6.05 });
    this.chair(29.0, 5.6, Math.PI / 2);
    this.chair(32.2, 6.9, 0.7, { rz: Math.PI / 2, y: 0.24 });
    this.notes.diary = this.paper(30.25, 0.78, 5.5, -0.4);
    this.window(33.4, 2.01, 0);

    // 304 -- storage. Something has been sleeping here.
    shelf(40.6, 2.3, 0); shelf(43.2, 2.3, 0); shelf(48.2, 2.3, 0);
    boxes(41, 6.4, 4); boxes(51.5, 3.0, 3); boxes(39.2, 7.0, 2);
    this.box(1.9, 0.18, 0.95, M.cloth, 49.6, 0.09, 5.4, { collide: false, ry: 0.2 });
    this.window(51, 2.01, 0);

    // Corridor.
    this.box(1.5, 0.45, 0.4, M.wood, 11, 0.225, 10.3, { collide: true });
    this.box(1.8, 0.06, 0.6, M.metal, 33, 0.82, 10.55, { collide: false });
    for (const [a, b] of [[-0.85, -0.25], [0.85, -0.25], [-0.85, 0.25], [0.85, 0.25]]) this.box(0.03, 0.8, 0.03, M.chrome, 33 + a, 0.4, 10.55 + b, { collide: false });
    this.boxes.push({ x0: 32.1, x1: 33.9, z0: 10.25, z1: 10.85 });
    this.chair(24.5, 13.2, 1.2, { rz: Math.PI / 2, y: 0.24 });
    boxes(47.5, 13.4, 3);
    this.decal("IT HEARS YOUR HEART", 37.5, 1.75, 10.02, 0);
    this.decal("DON'T RUN", 25.6, 1.55, 13.98, Math.PI, 2.2, 0.6);
    this.exitSign(53.96, 2.5, 11);

    // 305.
    this.bed(7.35, 19.6, Math.PI / 2);
    this.chair(3.3, 17.2, 0.5);
    this.window(5, 21.99, Math.PI);

    // Manager's office: the key to 309.
    this.box(1.6, 0.06, 0.8, M.wood, 16, 0.76, 19.6, { collide: false });
    this.box(0.5, 0.73, 0.78, M.wood, 15.45, 0.365, 19.6, { collide: false });
    this.box(0.5, 0.73, 0.78, M.wood, 16.55, 0.365, 19.6, { collide: false });
    this.boxes.push({ x0: 15.2, x1: 16.8, z0: 19.2, z1: 20.0 });
    this.chair(16, 20.6, Math.PI);
    this.box(0.04, 0.3, 0.04, M.chrome, 16.6, 0.94, 19.75, { collide: false });
    this.lampShade(16.6, 1.12, 19.75);
    this.box(0.5, 1.3, 0.6, M.metal, 21.7, 0.65, 17.0, { collide: true, ry: Math.PI / 2 });
    this.box(0.5, 1.3, 0.6, M.metal, 21.7, 0.65, 17.65, { collide: true, ry: Math.PI / 2 });
    shelf(10.25, 19.2, Math.PI / 2);
    this.notes.log = this.paper(16.35, 0.79, 19.45, 0.2);
    this.keyMesh = this.makeKey(15.75, 0.795, 19.5);
    this.window(13, 21.99, Math.PI); this.window(19, 21.99, Math.PI);

    // 309 -- Mira's room.
    this.bed(24.62, 20.6);
    this.box(0.9, 0.06, 0.6, M.wood, 29, 0.75, 21.55, { collide: false });
    for (const [a, b] of [[-0.4, -0.25], [0.4, -0.25], [-0.4, 0.25], [0.4, 0.25]]) this.box(0.04, 0.72, 0.04, M.wood, 29 + a, 0.36, 21.55 + b, { collide: false });
    this.boxes.push({ x0: 28.55, x1: 29.45, z0: 21.25, z1: 21.85 });
    this.tapeMesh = this.makeTape(29, 0.78, 21.5);
    this.decal("BREATHE SLOW", 29, 1.85, 21.98, Math.PI, 2.4, 0.6);
    this.window(31.2, 21.99, Math.PI);

    // 310 -- chairs facing the wall.
    for (const x of [38, 40, 42, 44, 46]) this.chair(x, 21.2, Math.PI);
    this.chair(42, 18.6, 0);

    // Closet.
    boxes(51, 20.6, 4);
  }

  lampShade(x, y, z) {
    const shade = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.18, 0.2, 20, 1, true),
      new THREE.MeshStandardMaterial({ color: 0xc8b48a, emissive: 0xffb060, emissiveIntensity: 0.6, side: THREE.DoubleSide, roughness: 0.9 }));
    shade.position.set(x, y, z); this.group.add(shade);
  }

  tvScreen(x, y, z, ry) {
    const c = document.createElement("canvas"); c.width = 96; c.height = 72;
    const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
    const m = new THREE.Mesh(new THREE.PlaneGeometry(0.58, 0.42), new THREE.MeshStandardMaterial({ map: t, emissive: 0xffffff, emissiveMap: t, emissiveIntensity: 1.6 }));
    m.position.set(x, y, z); m.rotation.y = ry; this.group.add(m);
    this.tvs.push({ canvas: c, ctx: c.getContext("2d"), tex: t, mesh: m, pos: new THREE.Vector3(x, y, z) });
  }

  exitSign(x, y, z) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(0.7, 0.22), new THREE.MeshStandardMaterial({
      map: textTexture("STAIRS", { w: 256, h: 80, bg: "#0b1a0f", color: "#7dffa0", font: '500 50px "IBM Plex Mono", monospace' }),
      emissive: 0xffffff, emissiveIntensity: 1.4,
    }));
    m.material.emissiveMap = m.material.map;
    m.position.set(x, y, z); m.rotation.y = -Math.PI / 2; this.group.add(m);
    this.exitSignMesh = m;
  }

  makeKey(x, y, z) {
    const g = new THREE.Group();
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.022, 0.006, 8, 20), this.M.brass); ring.rotation.x = Math.PI / 2;
    const shaft = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.006, 0.01), this.M.brass); shaft.position.x = 0.055;
    const bit = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.006, 0.022), this.M.brass); bit.position.set(0.085, 0, 0.012);
    const tag = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.004, 0.03), new THREE.MeshStandardMaterial({ map: textTexture("309", { w: 128, h: 72, bg: "#d6cdb4", color: "#3a2a1a" }) }));
    tag.position.set(-0.05, 0, 0);
    g.add(ring, shaft, bit, tag); g.position.set(x, y, z); g.rotation.y = 0.6; this.group.add(g);
    return g;
  }

  makeTape(x, y, z) {
    const g = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.06, 0.16), this.M.plastic);
    const win = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.005, 0.06), this.M.chrome); win.position.set(-0.03, 0.032, 0);
    const red = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.012, 0.018), new THREE.MeshStandardMaterial({ color: 0x400000, emissive: 0xff2010, emissiveIntensity: 0 }));
    red.position.set(0.09, 0.034, 0.04);
    g.add(body, win, red); g.position.set(x, y, z); g.rotation.y = 0.2; this.group.add(g);
    g.userData.led = red;
    return g;
  }

  buildLights() {
    const shadeMat = new THREE.MeshStandardMaterial({ color: 0x2b2d28, roughness: 0.5, metalness: 0.6, side: THREE.DoubleSide });
    const add = (x, y, z, color, intensity, dist, mode, fixture = true) => {
      const light = new THREE.PointLight(color, intensity, dist, 2);
      light.position.set(x, y, z); this.group.add(light);
      let bulb = null;
      if (fixture) {
        const cord = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, WH - y - 0.12), shadeMat); cord.position.set(x, (WH + y + 0.12) / 2, z);
        const shade = new THREE.Mesh(new THREE.ConeGeometry(0.24, 0.18, 24, 1, true), shadeMat); shade.position.set(x, y + 0.16, z);
        // Unlit: a lit surface this close to its own point light overflows to infinity and the bloom spreads it across the screen.
        bulb = new THREE.Mesh(new THREE.SphereGeometry(0.05, 12, 8), new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(4) }));
        bulb.userData.color = new THREE.Color(color);
        bulb.position.set(x, y + 0.02, z);
        this.group.add(cord, shade, bulb);
      }
      const L = { light, bulb, base: intensity, mode, phase: rand() * 100, override: 1, flickerUntil: 0, on: true };
      this.lights.push(L);
      return L;
    };
    for (const [x, mode] of [[7, "steady"], [19, "buzz"], [31, "broken"], [43, "steady"]]) add(x, 2.55, 12, 0xffc58a, 11, 12, mode);
    add(2.35, 0.95, 4.85, 0xffa860, 2.2, 6, "steady", false);        // 301 bedside lamp
    add(11.5, 0.9, 6.9, 0x9fb6ff, 1.4, 5.5, "tv", false);             // 302 television
    add(16.6, 1.05, 19.75, 0xffcf8a, 2.4, 6, "steady", false);        // office desk lamp
    add(29.6, 2.62, 19.5, 0xff7050, 2.0, 7, "buzz");                  // 309
    add(53.3, 2.4, 11, 0x40ff80, 0.7, 4.5, "steady", false);          // exit sign
  }

  // ------------------------------------------------------------ queries

  // Solid boxes near (x, z): wall cells, door jambs, closed doors and props.
  solidsNear(x, z, out = []) {
    out.length = 0;
    const [c0, r0] = cellOf(x, z);
    for (let r = r0 - 1; r <= r0 + 1; r++) for (let c = c0 - 1; c <= c0 + 1; c++) {
      if (this.isWall(c, r)) out.push({ x0: c * CS, x1: (c + 1) * CS, z0: r * CS, z1: (r + 1) * CS });
      const d = this.doorAt.get(`${c},${r}`);
      if (d && this.blocking(d)) {
        if (d.ax === "x") out.push({ x0: d.x - 0.5, x1: d.x + 0.5, z0: d.z - 0.12, z1: d.z + 0.12 });
        else out.push({ x0: d.x - 0.12, x1: d.x + 0.12, z0: d.z - 0.5, z1: d.z + 0.5 });
      }
    }
    for (const b of this.boxes) if (b.x1 > x - 3 && b.x0 < x + 3 && b.z1 > z - 3 && b.z0 < z + 3) out.push(b);
    return out;
  }

  // Push a circle out of everything solid.
  collide(pos, radius) {
    const boxes = this.solidsNear(pos.x, pos.z, this._tmp ??= []);
    for (let pass = 0; pass < 2; pass++) for (const b of boxes) {
      const cx = Math.max(b.x0, Math.min(pos.x, b.x1)), cz = Math.max(b.z0, Math.min(pos.z, b.z1));
      const dx = pos.x - cx, dz = pos.z - cz, d2 = dx * dx + dz * dz;
      if (d2 >= radius * radius) continue;
      if (d2 > 1e-9) { const d = Math.sqrt(d2), k = (radius - d) / d; pos.x += dx * k; pos.z += dz * k; }
      else {                                   // centre inside the box: leave by the nearest side
        const opts = [[b.x0 - radius - pos.x, 0], [b.x1 + radius - pos.x, 0], [0, b.z0 - radius - pos.z], [0, b.z1 + radius - pos.z]];
        opts.sort((a, c) => Math.abs(a[0] + a[1]) - Math.abs(c[0] + c[1]));
        pos.x += opts[0][0]; pos.z += opts[0][1];
      }
    }
  }

  // Is the cell solid for sight? Walls, and doors that are (nearly) closed.
  opaque(c, r) {
    if (this.isWall(c, r)) return true;
    const d = this.doorAt.get(`${c},${r}`);
    return !!d && this.blocking(d);
  }

  lineOfSight(a, b) {
    const dx = b.x - a.x, dz = b.z - a.z, len = Math.hypot(dx, dz), n = Math.ceil(len / 0.25);
    for (let i = 1; i < n; i++) {
      const t = i / n, x = a.x + dx * t, z = a.z + dz * t;
      const [c, r] = cellOf(x, z);
      if (this.isWall(c, r)) return false;
      const d = this.doorAt.get(`${c},${r}`);
      if (d) {
        if (this.blocking(d)) return false;
        // Through a doorway only via its 1 m opening, not the jambs.
        if (d.ax === "x" ? Math.abs(x - d.x) > 0.5 : Math.abs(z - d.z) > 0.5) return false;
      }
    }
    return true;
  }

  passable(c, r) {
    const ch = this.ch(c, r);
    if (ch === "#") return false;
    if ("DQE".includes(ch)) return !this.doorAt.get(`${c},${r}`).locked;
    return true;
  }

  // Breadth-first path over cells; returns world points (cell centres), excluding the start.
  path(from, to) {
    const [sc, sr] = cellOf(from.x, from.z), [tc, tr] = cellOf(to.x, to.z);
    if (!this.passable(tc, tr)) return null;
    const W = this.cols, prev = new Int32Array(W * this.rows).fill(-1), start = sr * W + sc, goal = tr * W + tc;
    const q = [start]; prev[start] = start;
    while (q.length) {
      const cur = q.shift();
      if (cur === goal) break;
      const c = cur % W, r = (cur / W) | 0;
      for (const [dc, dr] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nc = c + dc, nr = r + dr, ni = nr * W + nc;
        if (nc < 0 || nr < 0 || nc >= W || nr >= this.rows || prev[ni] !== -1 || !this.passable(nc, nr)) continue;
        prev[ni] = cur; q.push(ni);
      }
    }
    if (prev[goal] === -1) return null;
    const out = [];
    for (let cur = goal; cur !== start; cur = prev[cur]) out.push(cellCenter(cur % W, (cur / W) | 0));
    out.reverse();
    if (out.length) { out[out.length - 1].x = to.x; out[out.length - 1].z = to.z; }
    return out;
  }

  // Places the monster can wander to.
  patrolPoints() {
    const pts = [];
    for (let c = 2; c <= 26; c += 4) pts.push(cellCenter(c, 5 + (c % 8 === 2 ? 1 : 0)));
    for (const R of ROOMS) if (R.name !== "corridor") pts.push(cellCenter(Math.round((R.c0 + R.c1) / 2), Math.round((R.r0 + R.r1) / 2)));
    return pts;
  }

  randomFloorNear(p, minD, maxD) {
    for (let i = 0; i < 60; i++) {
      const a = rand() * Math.PI * 2, d = minD + rand() * (maxD - minD);
      const x = p.x + Math.cos(a) * d, z = p.z + Math.sin(a) * d;
      const [c, r] = cellOf(x, z);
      if (this.ch(c, r) === ".") return cellCenter(c, r);
    }
    return null;
  }
}
