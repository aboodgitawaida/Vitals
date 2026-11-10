// audio.js -- every sound is synthesised with Web Audio; nothing is downloaded.
//
// Positional sounds go through HRTF panners, so a whisper behind your left ear is
// behind your left ear on headphones. The stingers are built from distorted, detuned,
// noisy sources (the "nonlinear" acoustics of animal alarm calls, Blumstein et al. 2010).

export class GameAudio {
  constructor() { this.ctx = null; this.heartBpm = 70; this.heartLevel = 0.5; this.doubles = []; }

  init() {
    if (this.ctx) { this.ctx.resume(); return; }
    const ctx = this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.master = ctx.createGain(); this.master.gain.value = 0.9;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -12; comp.knee.value = 8; comp.ratio.value = 5; comp.attack.value = 0.003; comp.release.value = 0.25;
    this.master.connect(comp).connect(ctx.destination);
    this.reverb = ctx.createConvolver(); this.reverb.buffer = this.impulse(2.6, 2.4);
    this.revSend = ctx.createGain(); this.revSend.gain.value = 0.32;
    this.revSend.connect(this.reverb).connect(this.master);
    this.sfx = ctx.createGain(); this.sfx.connect(this.master); this.sfx.connect(this.revSend);
    this.dry = ctx.createGain(); this.dry.connect(this.master);            // close, non-spatial sounds
    this.noise = this.noiseBuffer(4, "white"); this.brown = this.noiseBuffer(6, "brown");
    this.distCurve = this.curve(60);
    this.startAmbient(); this.startRadio(); this.startChase(); this.startHeart();
  }

  get now() { return this.ctx.currentTime; }
  setVolume(v) { if (this.master) this.master.gain.value = v; }

  // ---------------------------------------------------------- building blocks

  noiseBuffer(sec, kind) {
    const n = this.ctx.sampleRate * sec, b = this.ctx.createBuffer(1, n, this.ctx.sampleRate), d = b.getChannelData(0);
    let last = 0;
    for (let i = 0; i < n; i++) {
      const w = Math.random() * 2 - 1;
      if (kind === "brown") { last = (last + 0.02 * w) / 1.02; d[i] = last * 3.5; } else d[i] = w;
    }
    return b;
  }

  impulse(sec, decay) {
    const n = this.ctx.sampleRate * sec, b = this.ctx.createBuffer(2, n, this.ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const d = b.getChannelData(ch);
      for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / n, decay) * (i < 400 ? i / 400 : 1);
    }
    return b;
  }

  curve(k) {
    const n = 1024, c = new Float32Array(n);
    for (let i = 0; i < n; i++) { const x = (i / n) * 2 - 1; c[i] = ((1 + k) * x) / (1 + k * Math.abs(x)); }
    return c;
  }

  src(buf = this.noise, loop = false, rate = 1) {
    const s = this.ctx.createBufferSource(); s.buffer = buf; s.loop = loop; s.playbackRate.value = rate; return s;
  }

  filt(type, freq, q = 1) { const f = this.ctx.createBiquadFilter(); f.type = type; f.frequency.value = freq; f.Q.value = q; return f; }

  env(peak, a, d, t = this.now) {
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.linearRampToValueAtTime(peak, t + a);
    g.gain.exponentialRampToValueAtTime(0.0001, t + a + d);
    return g;
  }

  panner(pos) {
    const p = this.ctx.createPanner();
    p.panningModel = "HRTF"; p.distanceModel = "inverse"; p.refDistance = 1.2; p.maxDistance = 80; p.rolloffFactor = 1.4;
    this.place(p, pos);
    p.connect(this.sfx);
    return p;
  }

  place(p, pos) {
    if (p.positionX) { p.positionX.value = pos.x; p.positionY.value = pos.y; p.positionZ.value = pos.z; }
    else p.setPosition(pos.x, pos.y, pos.z);
  }

  listener(pos, fwd) {
    if (!this.ctx) return;
    const l = this.ctx.listener;
    if (l.positionX) {
      l.positionX.value = pos.x; l.positionY.value = pos.y; l.positionZ.value = pos.z;
      l.forwardX.value = fwd.x; l.forwardY.value = fwd.y; l.forwardZ.value = fwd.z;
      l.upX.value = 0; l.upY.value = 1; l.upZ.value = 0;
    } else { l.setPosition(pos.x, pos.y, pos.z); l.setOrientation(fwd.x, fwd.y, fwd.z, 0, 1, 0); }
  }

  // ---------------------------------------------------------- beds

  startAmbient() {
    const ctx = this.ctx;
    this.amb = ctx.createGain(); this.amb.gain.value = 0.5; this.amb.connect(this.master);
    const rumble = this.src(this.brown, true); const lp = this.filt("lowpass", 160);
    const rg = ctx.createGain(); rg.gain.value = 0.32;
    rumble.connect(lp).connect(rg).connect(this.amb); rumble.start();
    // Two low tones a few Hz apart beat slowly against each other.
    for (const f of [43.6, 46.1, 87.9]) {
      const o = ctx.createOscillator(); o.frequency.value = f;
      const g = ctx.createGain(); g.gain.value = f > 80 ? 0.012 : 0.05;
      const lfo = ctx.createOscillator(); lfo.frequency.value = 0.03 + Math.random() * 0.05;
      const lg = ctx.createGain(); lg.gain.value = g.gain.value * 0.8;
      lfo.connect(lg).connect(g.gain); lfo.start();
      o.connect(g).connect(this.amb); o.start();
    }
    const air = this.src(this.noise, true); const bp = this.filt("bandpass", 3200, 0.6);
    const ag = ctx.createGain(); ag.gain.value = 0.012;
    air.connect(bp).connect(ag).connect(this.amb); air.start();
  }

  startRadio() {
    // The pocket radio: static that rises as the thing gets closer.
    const ctx = this.ctx;
    const s = this.src(this.noise, true, 0.9), bp = this.filt("bandpass", 1900, 0.8), hp = this.filt("highpass", 500);
    this.radioGain = ctx.createGain(); this.radioGain.gain.value = 0;
    this.crackle = ctx.createGain(); this.crackle.gain.value = 1;
    s.connect(bp).connect(hp).connect(this.crackle).connect(this.radioGain).connect(this.dry); s.start();
    this.radioLevel = 0;
    setInterval(() => {
      if (!this.ctx) return;
      const t = this.now, k = 0.5 + Math.random() * (0.5 + this.radioLevel * 1.5);
      this.crackle.gain.setTargetAtTime(k, t, 0.01);
    }, 45);
  }

  setRadio(level) {
    if (!this.radioGain) return;
    this.radioLevel = level;
    this.radioGain.gain.setTargetAtTime(level * level * 0.22, this.now, 0.15);
  }

  startChase() {
    const ctx = this.ctx;
    this.chaseGain = ctx.createGain(); this.chaseGain.gain.value = 0;
    const lp = this.filt("lowpass", 700, 1.5);
    lp.connect(this.chaseGain).connect(this.master);
    for (const f of [110, 116.5, 164.8, 174.6, 233.1]) {
      const o = ctx.createOscillator(); o.type = "sawtooth"; o.frequency.value = f; o.detune.value = (Math.random() - 0.5) * 30;
      const g = ctx.createGain(); g.gain.value = 0.04;
      const trem = ctx.createOscillator(); trem.frequency.value = 5 + Math.random() * 3;
      const tg = ctx.createGain(); tg.gain.value = 0.03; trem.connect(tg).connect(g.gain); trem.start();
      o.connect(g).connect(lp); o.start();
    }
  }

  setChase(level) { this.chaseGain?.gain.setTargetAtTime(level * 0.5, this.now, level > 0.5 ? 0.3 : 1.5); }
  setAmbient(level) { this.amb?.gain.setTargetAtTime(level * 0.5, this.now, 0.8); }

  // ---------------------------------------------------------- the heartbeat

  // Your heart, played back at the rate the camera measures (or the simulation).
  startHeart() {
    this.heartGain = this.ctx.createGain(); this.heartGain.gain.value = 0;
    this.heartGain.connect(this.dry);
    this.nextBeat = this.now + 0.5;
    setInterval(() => this.scheduleHeart(), 40);
  }

  scheduleHeart() {
    if (!this.ctx) return;
    const t = this.now;
    while (this.nextBeat < t + 0.12) {
      const period = 60 / Math.max(40, this.heartBpm);
      this.beat(this.nextBeat, 1, this.heartGain);
      this.nextBeat += period;
      this.onBeat?.(this.nextBeat - period);
    }
    for (const d of this.doubles) {
      while (d.next < t + 0.12 && d.next < d.until) { this.beat(d.next, 0.8, this.heartGain); d.next += 60 / d.bpm; }
    }
    this.doubles = this.doubles.filter((d) => d.next < d.until);
  }

  setHeart(bpm, level) {
    this.heartBpm = bpm;
    this.heartGain?.gain.setTargetAtTime(level, this.now, 0.5);
  }

  // A second heartbeat that is not yours.
  doubleHeart(sec, bpm) { this.doubles.push({ next: this.now + 0.37, until: this.now + sec, bpm }); }

  beat(t, vol, dest) {
    const dub = Math.min(0.3, 0.33 * 60 / this.heartBpm);
    this.thump(t, 58, vol, dest); this.thump(t + dub, 70, vol * 0.65, dest);
  }

  thump(t, f, vol, dest = this.sfx) {
    const o = this.ctx.createOscillator(); o.frequency.setValueAtTime(f * 1.7, t); o.frequency.exponentialRampToValueAtTime(f * 0.7, t + 0.09);
    const g = this.env(vol, 0.006, 0.16, t); const lp = this.filt("lowpass", 140);
    o.connect(lp).connect(g).connect(dest); o.start(t); o.stop(t + 0.3);
  }

  // ---------------------------------------------------------- one-shots

  footstep(vol, pos) {
    const t = this.now, s = this.src(this.noise, false, 0.7 + Math.random() * 0.3);
    const lp = this.filt("lowpass", 650 + Math.random() * 350), hp = this.filt("highpass", 60);
    const g = this.env(vol, 0.006, 0.14, t);
    s.connect(hp).connect(lp).connect(g).connect(pos ? this.panner(pos) : this.dry);
    s.start(t, Math.random() * 3, 0.25);
    this.thump(t, 50, vol * 0.5, pos ? this.panner(pos) : this.dry);
  }

  sting(intensity = 1) {
    const t = this.now, out = this.ctx.createGain(); out.gain.value = intensity;
    const shaper = this.ctx.createWaveShaper(); shaper.curve = this.distCurve;
    const hp = this.filt("highpass", 70);
    shaper.connect(hp).connect(out).connect(this.dry); out.connect(this.revSend);
    for (const f of [98, 103.8, 146.8, 207.7, 220, 311.1, 466]) {
      const o = this.ctx.createOscillator(); o.type = "sawtooth";
      o.frequency.setValueAtTime(f * 1.03, t); o.frequency.exponentialRampToValueAtTime(f * 0.94, t + 1.4);
      const g = this.env(0.18, 0.008, 1.6, t); o.connect(g).connect(shaper); o.start(t); o.stop(t + 1.8);
    }
    // A screech: a high tone frequency-modulated into chaos.
    const sc = this.ctx.createOscillator(); sc.type = "square"; sc.frequency.value = 2300;
    const fm = this.ctx.createOscillator(); fm.frequency.value = 37; const fg = this.ctx.createGain(); fg.gain.value = 900;
    fm.connect(fg).connect(sc.frequency);
    const bp = this.filt("bandpass", 2600, 2.5), sg = this.env(0.22, 0.01, 0.9, t);
    sc.connect(bp).connect(sg).connect(out); sc.start(t); fm.start(t); sc.stop(t + 1.1); fm.stop(t + 1.1);
    const n = this.src(), nb = this.filt("bandpass", 1300, 0.7), ng = this.env(0.7, 0.003, 0.5, t);
    n.connect(nb).connect(ng).connect(shaper); n.start(t, 0, 0.6);
    this.thump(t, 45, 1.2 * intensity, this.dry);
  }

  // Noise shaped by moving formants into something that is almost speech.
  whisper(pos, sec = 2.4, vol = 0.9) {
    const t = this.now, s = this.src(this.noise, true);
    const out = pos ? this.panner(pos) : this.dry, g = this.ctx.createGain(); g.gain.value = 0;
    const f1 = this.filt("bandpass", 600, 6), f2 = this.filt("bandpass", 1600, 8), f3 = this.filt("bandpass", 2600, 6);
    const hp = this.filt("highpass", 300), mixg = this.ctx.createGain(); mixg.gain.value = 2.2;
    s.connect(hp); hp.connect(f1); hp.connect(f2); hp.connect(f3);
    f1.connect(mixg); f2.connect(mixg); f3.connect(mixg); mixg.connect(g).connect(out);
    let k = t;
    while (k < t + sec) {
      const len = 0.08 + Math.random() * 0.2;
      g.gain.setTargetAtTime(vol * (0.4 + Math.random() * 0.6), k, 0.02);
      g.gain.setTargetAtTime(0.0, k + len * 0.8, 0.03);
      f1.frequency.setValueAtTime(300 + Math.random() * 500, k);
      f2.frequency.setValueAtTime(900 + Math.random() * 1300, k);
      k += len + Math.random() * 0.08;
    }
    s.start(t); s.stop(t + sec + 0.3);
  }

  slam(pos, vol = 1) {
    const t = this.now, out = this.panner(pos);
    const n = this.src(), lp = this.filt("lowpass", 500), g = this.env(vol * 1.4, 0.002, 0.5, t);
    n.connect(lp).connect(g).connect(out); n.start(t, Math.random(), 0.7);
    this.thump(t, 40, vol * 1.5, out);
    const c = this.src(), hp = this.filt("highpass", 2500), cg = this.env(vol * 0.5, 0.001, 0.06, t);
    c.connect(hp).connect(cg).connect(out); c.start(t, 0, 0.1);
  }

  knock(pos, n = 3) {
    let t = this.now;
    const out = this.panner(pos);
    for (let i = 0; i < n; i++) {
      this.thump(t, 120, 0.9, out);
      const s = this.src(), lp = this.filt("bandpass", 900, 1.2), g = this.env(0.5, 0.001, 0.07, t);
      s.connect(lp).connect(g).connect(out); s.start(t, Math.random(), 0.1);
      t += 0.28 + Math.random() * 0.12;
    }
  }

  creak(pos, vol = 0.6) {
    const t = this.now, out = this.panner(pos), o = this.ctx.createOscillator(); o.type = "sawtooth";
    o.frequency.setValueAtTime(80, t);
    for (let i = 1; i <= 8; i++) o.frequency.linearRampToValueAtTime(70 + Math.random() * 90, t + i * 0.11);
    const bp = this.filt("bandpass", 700, 9), g = this.env(vol, 0.05, 0.9, t);
    o.connect(bp).connect(g).connect(out); o.start(t); o.stop(t + 1.1);
  }

  phone(pos, rings = 2) {
    const out = this.panner(pos), bp = this.filt("bandpass", 1500, 1.5);
    bp.connect(out);
    for (let r = 0; r < rings; r++) {
      const t0 = this.now + r * 3;
      for (const f of [440, 480]) {
        const o = this.ctx.createOscillator(); o.frequency.value = f;
        const g = this.ctx.createGain(); g.gain.value = 0;
        for (let k = 0; k < 1.6; k += 0.1) g.gain.setValueAtTime(((k * 10) | 0) % 2 ? 0.0 : 0.18, t0 + k);
        g.gain.setValueAtTime(0, t0 + 1.7);
        o.connect(g).connect(bp); o.start(t0); o.stop(t0 + 1.8);
      }
    }
  }

  pickup() {
    const t = this.now;
    for (const [f, d] of [[880, 0], [1320, 0.08]]) {
      const o = this.ctx.createOscillator(); o.type = "triangle"; o.frequency.value = f;
      const g = this.env(0.08, 0.005, 0.5, t + d); o.connect(g).connect(this.dry); o.start(t + d); o.stop(t + d + 0.6);
    }
  }

  click() { this.thump(this.now, 900, 0.15, this.dry); }

  // A continuous positional source (the monster's breathing and clicks, the TV).
  voice(kind) {
    const ctx = this.ctx, pan = this.panner({ x: 0, y: 0, z: 0 }), gain = ctx.createGain(); gain.gain.value = 0;
    gain.connect(pan);
    if (kind === "monster") {
      const s = this.src(this.brown, true), bp = this.filt("bandpass", 320, 3), am = ctx.createGain(); am.gain.value = 0.6;
      const lfo = ctx.createOscillator(); lfo.frequency.value = 0.45; const lg = ctx.createGain(); lg.gain.value = 0.5;
      lfo.connect(lg).connect(am.gain); lfo.start();
      s.connect(bp).connect(am).connect(gain); s.start();
      const clicks = setInterval(() => {
        if (gain.gain.value < 0.02) return;
        const t = this.now + Math.random() * 0.05, n = 1 + ((Math.random() * 4) | 0);
        for (let i = 0; i < n; i++) {
          const c = this.src(), hp = this.filt("bandpass", 2000 + Math.random() * 1500, 4), g = this.env(1.2, 0.001, 0.025, t + i * 0.05);
          c.connect(hp).connect(g).connect(gain); c.start(t + i * 0.05, Math.random() * 3, 0.05);
        }
      }, 380);
      return { pan, gain, stop: () => clearInterval(clicks) };
    }
    const s = this.src(this.noise, true), bp = this.filt("bandpass", 3000, 0.5);
    s.connect(bp).connect(gain); s.start();
    return { pan, gain, stop: () => s.stop() };
  }
}
