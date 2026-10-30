// Helper for test_web_features.py: runs web/features.js in Node on frames sent as JSON on stdin.
import { readFileSync } from "node:fs";
import { FeatureExtractor } from "../web/features.js";

const input = JSON.parse(readFileSync(0, "utf8"));
const fx = new FeatureExtractor(input.cfg);
const out = input.frames.map((f) => {
  const frame = { data: Buffer.from(f.rgba, "base64"), width: f.width, height: f.height };
  const r = fx.process(frame, f.landmarks);
  if (!r.face) return { face: false };
  return { face: true, colors: r.colors, bg_rgb: r.bg_rgb, pose: r.pose, movement: r.movement,
           fast_motion: r.fast_motion, stats: r.stats };
});
process.stdout.write(JSON.stringify(out));
