// worker.js -- runs the project's Python (live_session.py and the DSP it uses) in
// Pyodide, off the page's main thread so the camera view never stalls.
// Started as a module worker (new Worker(url, {type: "module"})).
//
// Messages in:  {type: "init"}
//               {type: "frames", frames: [...]}   per-frame features since the last batch
//               {type: "control", action: "method" | "background" | "reset", value}
// Messages out: {type: "status", text}            while loading
//               {type: "ready", cfg}              regions and thresholds from face_layout.py
//               {type: "info", info, ran, ms}     after each batch (ran = an analysis ran)
//               {type: "error", text}

const PYODIDE_VERSION = "314.0.7";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let py = null;

const GLUE = `
import json
import face_layout
from live_session import LiveSession

SESSION = LiveSession()

def config():
    keys = ("REGIONS", "GRID", "PATCHES", "MIN_PATCH_PIXELS", "DARK_FRACTION", "CLIP_LEVEL",
            "MOTION_LIMIT", "SMALL_MOTION", "SMOOTHING")
    return json.dumps({k: getattr(face_layout, k) for k in keys})

def frames(batch_json):
    batch = json.loads(batch_json)
    ran = False
    for k, f in enumerate(batch):
        ran |= SESSION.frame(f["t"], f["face"], f.get("colors"), f.get("bg"), f.get("pose"),
                             f.get("fast", False), f.get("small", False), f.get("stats"),
                             f.get("fps", 0.0), analyze=k == len(batch) - 1)
    return json.dumps({"info": SESSION.info(), "ran": ran})

def control(action, value):
    if action == "method":
        SESSION.set_method(value)
    elif action == "background":
        SESSION.set_background(value)
    elif action == "reset":
        SESSION.reset()
`;

async function init() {
  postMessage({ type: "status", text: "Loading Python…" });
  const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
  py = await loadPyodide({ indexURL: PYODIDE_URL });
  postMessage({ type: "status", text: "Loading numpy and scipy…" });
  await py.loadPackage(["numpy", "scipy"]);
  postMessage({ type: "status", text: "Loading the signal processing…" });
  // The project's own .py files, copied next to this page by build.mjs.
  const files = await (await fetch("py/manifest.json")).json();
  for (const name of files) {
    const res = await fetch(`py/${name}`);
    if (!res.ok) throw new Error(`could not load ${name}`);
    py.FS.writeFile(`/home/pyodide/${name}`, await res.text());
  }
  py.runPython(GLUE);
  postMessage({ type: "ready", cfg: JSON.parse(py.globals.get("config")()) });
}

onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") {
      await init();
    } else if (msg.type === "frames") {
      const t0 = performance.now();
      const out = JSON.parse(py.globals.get("frames")(JSON.stringify(msg.frames)));
      postMessage({ type: "info", ...out, ms: performance.now() - t0 });
    } else if (msg.type === "control") {
      py.globals.get("control")(msg.action, msg.value);
    }
  } catch (err) {
    postMessage({ type: "error", text: String(err && err.message ? err.message : err) });
  }
};
