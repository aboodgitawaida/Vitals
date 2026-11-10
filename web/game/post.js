// post.js -- the camera's look: bloom on the few lights there are, then a film grade:
// desaturated green-grey, heavy vignette, grain, a little chromatic aberration, and a
// squeeze of the frame on every heartbeat that grows with fear.

import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const Grade = {
  uniforms: {
    tDiffuse: { value: null }, uTime: { value: 0 }, uRes: { value: new THREE.Vector2(1, 1) },
    uFear: { value: 0 }, uPulse: { value: 0 }, uStatic: { value: 0 }, uRed: { value: 0 }, uBlack: { value: 0 },
    uGrain: { value: 0.07 }, uBright: { value: 1 },
  },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float uTime, uFear, uPulse, uStatic, uRed, uBlack, uGrain, uBright; uniform vec2 uRes;
    varying vec2 vUv;
    float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main(){
      vec2 c = vUv - 0.5;
      float r2 = dot(c, c);
      vec2 uv = vUv - c * uPulse * (0.006 + uFear * 0.012);
      // Tearing lines while the signal breaks up.
      float line = step(0.985 - uStatic * 0.2, hash(vec2(floor(uv.y * 120.0), floor(uTime * 24.0))));
      uv.x += line * uStatic * 0.04 * (hash(vec2(uTime, uv.y)) - 0.5);
      float ab = (0.0015 + uFear * 0.004 + uStatic * 0.01) * (0.3 + r2 * 3.0);
      vec3 col = vec3(texture2D(tDiffuse, uv + c * ab).r, texture2D(tDiffuse, uv).g, texture2D(tDiffuse, uv - c * ab).b);
      col = clamp(col, 0.0, 4.0);
      col *= uBright;
      float l = dot(col, vec3(0.299, 0.587, 0.114));
      col = mix(col, vec3(l) * vec3(0.93, 1.0, 0.94), 0.38 + uFear * 0.25);
      col = col * vec3(0.97, 1.0, 0.97) + vec3(0.006, 0.008, 0.007);
      float vig = 1.0 - smoothstep(0.12, 0.62, r2 * (1.15 + uFear * 0.7 + uPulse * uFear * 0.6));
      col *= mix(0.12, 1.0, vig);
      float g = hash(vUv * uRes + fract(uTime * 7.13) * 91.7) - 0.5;
      col += g * (uGrain + uFear * 0.04) * (0.6 + l);
      col = mix(col, vec3(hash(vUv * uRes + uTime * 13.0)) * 0.8, uStatic * 0.55);
      col = mix(col, vec3(0.35, 0.0, 0.0) * (0.5 + l), uRed);
      col *= 1.0 - uBlack;
      gl_FragColor = vec4(col, 1.0);
    }`,
};

// A single NaN or infinite pixel (a lit surface millimetres from a light, a degenerate bump
// normal) is invisible on its own, but bloom blurs it over the whole frame and the screen
// goes black. Clean the frame before anything blurs it.
const Sanitize = {
  uniforms: { tDiffuse: { value: null } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `
    uniform sampler2D tDiffuse; varying vec2 vUv;
    void main(){
      vec4 c = texture2D(tDiffuse, vUv);
      if (any(isnan(c)) || any(isinf(c))) c = vec4(0.0, 0.0, 0.0, 1.0);
      gl_FragColor = vec4(min(c.rgb, vec3(48.0)), 1.0);
    }`,
};

export class Post {
  constructor(renderer, scene, camera) {
    this.renderer = renderer;
    this.composer = new EffectComposer(renderer);
    this.composer.addPass(new RenderPass(scene, camera));
    this.composer.addPass(new ShaderPass(Sanitize));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.55, 0.7, 0.82);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.grade = new ShaderPass(Grade);
    this.composer.addPass(this.grade);
    this.u = this.grade.uniforms;
    this.resize();
  }

  resize() {
    const w = innerWidth, h = innerHeight, pr = this.renderer.getPixelRatio();
    this.composer.setPixelRatio(pr);
    this.composer.setSize(w, h);
    this.u.uRes.value.set(w * pr, h * pr);
  }

  render(t) { this.u.uTime.value = t; this.composer.render(); }
}
