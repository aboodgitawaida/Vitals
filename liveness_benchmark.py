"""
liveness_benchmark.py -- does the liveness check tell live faces from fakes?

Simulates the two phases of a check for real people and for attacks, many
times each, and counts the verdicts.

What the simulated camera does (so the check is not tested on an easy world):
  * the screen's light reaches the face (strength depends on distance and room
    light) and, much more weakly, the wall behind
  * auto-exposure pulls overall brightness back within ~0.7 s
  * auto white-balance pulls the overall tint back within ~2 s
  * screen + camera delay of 70-330 ms, camera noise, a real (or no) pulse

Scenarios:
  live, close          person 30-50 cm from the screen, normal room light
  live, far/bright     person far away or in a bright room: little screen light
  printed photo        reflects the flashes like skin, but has no pulse
  mask                 reflects the flashes, no pulse
  phone replay         a phone showing a pre-recorded video (which HAS a pulse);
                       our flashes only glint on its glass in 1-3 spots
  virtual camera       video injected as a fake webcam: has a pulse, never sees the screen
  live deepfake        a real person behind real-time face-swap software that
                       carries their lighting and pulse through: the known blind spot

Run:  python liveness_benchmark.py          (~2-4 minutes, all CPU cores)

HONEST CAVEAT: these attack models are simplified. Real attacks (a good OLED
phone, a matte photo under a lamp) must be tested in person; record them with
the app and re-score with `python liveness.py recordings/liveness_...`.
"""

import numpy as np

from face_regions import PATCHES, patch_region
from liveness import colour_at, flash_check, make_challenge, progressive_pulse_check, verdict
from session_benchmark import make_session
from synthetic_benchmark import PULSE_DIRECTION, SKIN_TONES, WALL

SCENARIOS = ["live, close", "live, far/bright", "printed photo", "mask", "phone replay",
             "virtual camera", "live deepfake"]


def _lowpass(x, t, tau):
    """First-order lag (how auto-exposure / white-balance catch up)."""
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        a = 1 - np.exp(-(t[i] - t[i - 1]) / tau)
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def simulate_flash(scenario, tone, rng, noise=0.6):
    """One flash phase. Returns times, emitted RGB (N,3), {patch: (N,3)}, background (N,3)."""
    t = np.cumsum(rng.normal(1 / 30, 0.003, int(7.0 * 30)))
    t -= t[0]
    start = 0.5
    challenge = make_challenge(rng)
    emitted = np.array([colour_at(challenge, x - start) or (255, 255, 255) for x in t], dtype=float)

    delay = rng.uniform(0.07, 0.33)
    seen = np.array([colour_at(challenge, x - start - delay) or (255, 255, 255) for x in t], dtype=float) / 255

    if scenario in ("live, close", "printed photo", "mask", "live deepfake"):
        g_face = rng.uniform(0.15, 0.6)          # screen light vs room light on the face
    elif scenario == "live, far/bright":
        g_face = rng.uniform(0.01, 0.06)
    else:
        g_face = 0.0                             # replay / virtual camera: no real face there
    g_wall = g_face * rng.uniform(0.03, 0.2)
    ambient = np.array([1.0, 0.97, 0.9])         # slightly warm room light

    light_face = ambient + g_face * seen
    light_wall = ambient + g_wall * seen

    skin, strength = SKIN_TONES[tone]
    if scenario == "mask":
        skin = np.array([200.0, 170.0, 150.0])   # pale silicone
    if scenario == "printed photo":
        skin = skin * 0.9
    beats = rng.uniform(55, 95) / 60
    has_pulse = scenario not in ("printed photo", "mask")
    wave = np.sin(2 * np.pi * beats * t) if has_pulse else np.zeros_like(t)

    patches = {}
    glint_patches = set(rng.choice(PATCHES, size=rng.integers(1, 4), replace=False))
    for name in PATCHES:
        amp = 0.0012 * strength * (1.0 if patch_region(name) == "forehead" else 0.7)
        if scenario in ("phone replay", "virtual camera"):
            base = (skin / 255) * ambient                     # the video's own lighting, fixed
            colour = 255 * (base + amp * wave[:, None] * PULSE_DIRECTION * 0.5)
            if scenario == "phone replay" and name in glint_patches:
                colour = colour + 255 * rng.uniform(0.02, 0.08) * seen   # gloss: colour of the screen itself
        else:
            colour = skin * light_face * (1 + amp * wave[:, None] * PULSE_DIRECTION / 0.77)
        patches[name] = colour
    wall = WALL * light_wall

    if scenario != "virtual camera":
        # Our camera's auto-exposure and auto white-balance react to the whole picture.
        scene = np.mean([p for p in patches.values()], axis=0) * 0.4 + wall * 0.6
        lum = scene.mean(axis=1)
        exposure = lum.mean() / _lowpass(lum, t, 0.7)
        tint_now = scene / lum[:, None]                   # per-channel share of the light
        balance = tint_now.mean(axis=0) / _lowpass(tint_now, t, 2.0)   # slowly undo any tint shift
        gain = exposure[:, None] * balance
        patches = {n: p * gain for n, p in patches.items()}
        wall = wall * gain

    patches = {n: p + rng.normal(0, noise, p.shape) for n, p in patches.items()}
    wall = wall + rng.normal(0, noise / 3, wall.shape)
    return t, emitted, patches, wall


def run_trial(args):
    scenario, tone, seed = args
    rng = np.random.default_rng(seed)
    noise = float(rng.choice([0.3, 0.6, 1.1]))             # bright / normal / dim room
    has_pulse = scenario not in ("printed photo", "mask")
    s = make_session(tone, noise, rng, seconds=30.5, pulse=has_pulse, events=False)
    pulse, when = progressive_pulse_check(s["t"], s["patches"], s["background"], s["pose"])
    flash = flash_check(*simulate_flash(scenario, tone, rng, noise))
    return scenario, tone, verdict(pulse, flash), pulse.verdict, flash.verdict, when, noise


def main(trials=60, seed=11):
    from multiprocessing import Pool
    tones = list(SKIN_TONES)
    jobs = [(sc, tones[i % 3], seed * 10000 + k * 1000 + i) for k, sc in enumerate(SCENARIOS) for i in range(trials)]
    with Pool() as pool:
        res = pool.map(run_trial, jobs)

    print(f"Simulated liveness checks, {trials} per scenario (skin tones and bright/normal/dim light mixed).\n")
    print(f"{'scenario':<18}{'LIVE':>7}{'NOT LIVE':>10}{'NOT CONF.':>11}   {'pulse pass':>10}{'flash pass':>11}   should be")
    for sc in SCENARIOS:
        r = [x for x in res if x[0] == sc]
        share = lambda v: np.mean([x[2] == v for x in r])
        expect = {"live, close": "LIVE", "live, far/bright": "LIVE or NOT CONFIRMED",
                  "printed photo": "never LIVE", "mask": "never LIVE", "phone replay": "NOT LIVE",
                  "virtual camera": "NOT LIVE", "live deepfake": "LIVE (blind spot)"}[sc]
        print(f"{sc:<18}{share('LIVE'):>7.0%}{share('NOT LIVE'):>10.0%}{share('NOT CONFIRMED'):>11.0%}"
              f"   {np.mean([x[3] == 'pass' for x in r]):>10.0%}{np.mean([x[4] == 'pass' for x in r]):>11.0%}   {expect}")

    attacks = [x for x in res if x[0] in ("printed photo", "mask", "phone replay", "virtual camera")]
    live = [x for x in res if x[0] == "live, close"]
    print(f"\nAttacks wrongly accepted as LIVE: {np.mean([x[2] == 'LIVE' for x in attacks]):.1%} "
          f"(photo, mask, phone replay, virtual camera)")
    real = [x for x in res if x[0] in ("live, close", "live, far/bright")]
    print(f"Real people called NOT LIVE (accused of being fake): {np.mean([x[2] == 'NOT LIVE' for x in real]):.1%}")
    whens = [x[5] for x in live if x[2] == "LIVE" and x[5] is not None]
    print(f"Real people (close): median time to confirm the pulse {np.median(whens):.0f} s")
    print("\nReal people (close), share confirmed LIVE, by skin tone and room light:")
    print(f"{'':<10}" + "".join(f"{lab:>9}" for lab in ("bright", "normal", "dim")))
    extra = [run_trial(("live, close", tn, 777000 + i * 7 + j)) for tn in tones for i in range(20) for j in range(3)]
    for tn in tones:
        row = f"{tn:<10}"
        for nz in (0.3, 0.6, 1.1):
            g = [x for x in live + extra if x[1] == tn and x[6] == nz]
            row += f"{np.mean([x[2] == 'LIVE' for x in g]):>9.0%}" if g else f"{'-':>9}"
        print(row)
    print("(Not confirmed = asked to add light or move closer; never called fake.)")


if __name__ == "__main__":
    main()
