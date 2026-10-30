// worker.js -- runs the project's Python (live_session.py and the DSP it uses) in
// Pyodide, off the page's main thread so the camera view never stalls.
// Started as a module worker (new Worker(url, {type: "module"})).
//
// Messages in:  {type: "init"}
//               {type: "frames", frames: [...]}   per-frame features since the last batch
//               {type: "control", action: "method" | "background" | "reset", value}
// Messages out: {type: "status", text}            while loading
//               {type: "ready"}
//               {type: "info", info, ran, ms}     after each batch (ran = an analysis ran)
//               {type: "error", text}

const PYODIDE_VERSION = "314.0.7";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let py = null;

const GLUE = `
import json
from live_session import LiveSession

SESSION = LiveSession()

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

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not load ${url}`);
  return res.text();
}

async function init() {
  postMessage({ type: "status", text: "Loading Python…" });
  // Download numpy and our .py files WHILE the Python core downloads, instead of after it.
  // (Fetching the numpy wheel warms the browser cache; loadPackage then reads it from there.)
  const numpyPrefetch = fetch(`${PYODIDE_URL}pyodide-lock.json`)
    .then((r) => r.json())
    .then((lock) => fetch(`${PYODIDE_URL}${lock.packages.numpy.file_name}`))
    .then((r) => r.arrayBuffer())
    .catch(() => {});                         // only a speed-up: loadPackage fetches it anyway
  const sources = fetch("py/manifest.json").then((r) => r.json())
    .then((files) => Promise.all(files.map(async (name) => [name, await fetchText(`py/${name}`)])));
  const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
  py = await loadPyodide({ indexURL: PYODIDE_URL });
  postMessage({ type: "status", text: "Loading numpy…" });
  // No scipy: vitals_dsp falls back to signal_lite.py (numpy only), ~14 MB less to download.
  await numpyPrefetch;
  await py.loadPackage(["numpy"]);
  postMessage({ type: "status", text: "Loading the signal processing…" });
  // The project's own .py files, copied next to this page by build.mjs.
  for (const [name, text] of await sources) py.FS.writeFile(`/home/pyodide/${name}`, text);
  py.runPython(GLUE);
  postMessage({ type: "ready" });
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
