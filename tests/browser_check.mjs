// browser_check.mjs -- run the real web app in headless Chromium with a video file as the webcam.
//
// Checks the whole browser path end to end (camera, MediaPipe in JS, features.js, the Python
// analysis in Pyodide, the dashboard) and prints what the page shows once a second.
//
//   node web/build.mjs
//   ffmpeg -i mcd_rppg/1020_before/vid.avi -t 50 -pix_fmt yuv420p -r 30 cam.y4m
//   node tests/browser_check.mjs cam.y4m [seconds]
//
// Needs Playwright with Chromium (npm i playwright && npx playwright install chromium) and
// internet access (MediaPipe and Pyodide load from CDNs). Chromium loops the video, so a
// reading may drop briefly where the clip restarts.

import { createRequire } from "node:module";
import { readFile } from "node:fs/promises";
import http from "node:http";
import path from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const [video, seconds = "45"] = process.argv.slice(2);
if (!video) {
  console.error("usage: node tests/browser_check.mjs video.y4m [seconds]");
  process.exit(2);
}
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "dist");
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
                ".py": "text/plain", ".png": "image/png" };

const server = http.createServer(async (req, res) => {
  let file = path.join(root, decodeURIComponent(req.url.split("?")[0]));
  if (file.endsWith("/")) file += "index.html";
  try {
    const body = await readFile(file);
    res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404);
    res.end();
  }
}).listen(0);
const url = `http://localhost:${server.address().port}/`;

const browser = await chromium.launch({
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
         `--use-file-for-fake-video-capture=${path.resolve(video)}`],
});
const page = await browser.newPage({ viewport: { width: 1366, height: 900 } });
page.on("pageerror", (e) => console.log("page error:", e.message));
await page.goto(url);
await page.click("#start");
const t0 = Date.now();
const shown = [];
while (Date.now() - t0 < Number(seconds) * 1000) {
  await page.waitForTimeout(1000);
  const s = await page.evaluate(() => ({
    bpm: document.getElementById("bpm").textContent,
    level: document.getElementById("level-text").textContent,
    details: document.getElementById("details").textContent,
  }));
  console.log(`${((Date.now() - t0) / 1000).toFixed(0).padStart(3)} s  ${s.bpm.padStart(3)} BPM  ${s.level.padEnd(22)} ${s.details}`);
  if (/^\d+$/.test(s.bpm)) shown.push(Number(s.bpm));
}
await browser.close();
server.close();
if (!shown.length) {
  console.log("No heart rate was shown.");
  process.exit(1);
}
shown.sort((a, b) => a - b);
console.log(`Shown ${shown.length} times, median ${shown[Math.floor(shown.length / 2)]} BPM, range ${shown[0]}-${shown.at(-1)}.`);
