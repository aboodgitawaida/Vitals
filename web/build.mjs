// build.mjs -- assembles the static site in dist/ (what Vercel serves).
//
//   node web/build.mjs
//
// Copies the page from web/ and the Python modules the browser needs (live_session.py
// and everything it imports from this repo) into dist/py/, with a manifest the
// worker reads. Nothing is compiled: the .py files are served as they are.

import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "dist");

// live_session.py plus every repo module it imports, directly or indirectly.
function localModules(entry) {
  const seen = new Set(), todo = [entry];
  while (todo.length) {
    const name = todo.pop();
    if (seen.has(name)) continue;
    seen.add(name);
    const src = readFileSync(join(root, `${name}.py`), "utf8");
    for (const m of src.matchAll(/^\s*(?:from\s+(\w+)\s+import|import\s+(\w+))/gm)) {
      const mod = m[1] || m[2];
      if (existsSync(join(root, `${mod}.py`))) todo.push(mod);
    }
  }
  return [...seen].sort().map((n) => `${n}.py`);
}

rmSync(out, { recursive: true, force: true });
mkdirSync(join(out, "py"), { recursive: true });
cpSync(join(root, "web"), out, { recursive: true, filter: (src) => !src.endsWith("build.mjs") });
const files = localModules("live_session");
for (const f of files) cpSync(join(root, f), join(out, "py", f));
writeFileSync(join(out, "py", "manifest.json"), JSON.stringify(files));
console.log(`dist/ ready (python: ${files.join(", ")})`);
