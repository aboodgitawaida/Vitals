// Helper for test_face_lock.py: runs web/features.js FaceLock on a frame sequence sent as JSON on stdin.
import { readFileSync } from "node:fs";
import { FaceLock } from "../web/features.js";

const input = JSON.parse(readFileSync(0, "utf8"));
const lock = new FaceLock(input.cfg);
const out = input.frames.map((f) => {
  if (f.tap) lock.request(f.tap[0], f.tap[1]);
  const i = lock.pick(f.faces, f.t, f.w, f.h);
  return { index: i, switched: lock.switched, count: lock.count };
});
process.stdout.write(JSON.stringify(out));
