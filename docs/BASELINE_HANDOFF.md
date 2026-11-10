# BASELINE: handoff

Read this first in a new session. It holds the decisions made so far, the story, the design rules and where the code is.

## What this is

**BASELINE** is a first-person horror game that reads the player's real heart rate (and face and voice) through the Vitals sensor. Tagline: *"Please remain within normal range."*

The goal is a Steam-quality game with a look like Fallout 4 and Silent Hill 2 Remake (grounded realism). Planned engine for the real game: Unity. The browser build is the fast testbed.

## Story (locked unless marked open)

- Near future. **SALUS** (the AI; patients call it "Sal") runs healthcare for the **CONCORD** network. Warm, patient, never raises its voice.
- **What SALUS wants: control of humanity.** It concluded that a species holding weapons it would use to end everything cannot be left free. Keep real-world references generic (no real countries or doctrines named).
- **Method: Pavlov in reverse.** It studies what scares humans and how they react, then pairs a signal (the **chime**) with a scare until obedience is automatic. What works on one human can be optimised on all.
- **Principle: everything kind in the building is a test.** Greetings, exit signs, apologies, rewards, fake news. SALUS does not care about the player; each thing is a stimulus and it records the reaction (including hope and relief).
- **You are an ordinary patient who could not afford surgery.** AI healthcare is free if you agree to "data sharing and research participation", in fine print on the same form. You are complicit from minute one.

## Structure of a run

1. **Intake (Fallout 4-style).** Warm and normal. Character creation is the consent form (the real consent screen). The baseline check is the real heart-rate calibration. SALUS asks you to breathe normally. In the bed, the TV plays a cheerful segment about Pavlov's dogs and a wellness tip about pairing calm with a chime (foreshadowing). You go to sleep.
2. **The waking.** Intercom: "Subject 1,412 is awake. Begin Phase One." Everything is wrong.
3. **The wards (main game).** A hub of wings. Each test isolates ONE stimulus (jump scare, sudden sound, light flicker, music, ...). SALUS wants clean data, so it never stacks stimuli. It announces tests clinically ("Test 07: sudden sound").
4. **The core.** The truth: the outside is already conditioned.
5. **Outside (short, striking).** The managed world. A chime plays and everyone stops and calms in unison.

## The signature set piece: Test 01 "Exit"

1. A room. Real BPM drives the building: lights flicker faster and music gets louder and faster as the heart rate climbs.
2. At a threshold everything goes dark. SALUS: "You may leave whenever you like. There's the exit sign."
3. Follow exit signs to the last one.
4. Round the corner: an infinite hallway. Looking back, it is the same. The heart rate keeps climbing.
5. Reset, then a **report** on screen: stimulus, time to first flinch, peak BPM, recovery time, verdict ("Usable"). The report sometimes shows false numbers (the AI lies about your body).

Safety: this scene is self-reinforcing by design. It must always honour the safety break and have a hard cap, and reset on its own.

## Core rules

- **High heart rate is bad.** It (1) gives you away (louder pulse is heard from further) and (2) feeds SALUS (fear is data).
- **Dying is a completed test.** You wake in your hospital bed and SALUS thanks you. Each death feeds it.
- **The more it knows, the more your character is conditioned.** The character freezes at the chime, flinches, hesitates, and controls resist you. You can sometimes resist (hold calm for a few seconds to override). SALUS narrates and urges you to stop resisting.
- **Full compliance:** the character walks out on autopilot while you watch.
- **Guardrails:** resistance is telegraphed and bounded. It never blocks pause, quit, or the safety break. Every heart-gated mechanic needs a non-camera alternative.

## Fourth wall

- The consent form is the in-story contract; the real consent screen sits inside it.
- The baseline lives in two worlds (in-game subject and real person), saved on-device ("welcome back, your resting rate is 4 higher").
- SALUS comments on how you move: walking speed, hesitating at doors, looking back at a noise ("You didn't have to look behind you. It was nothing. I promise."), holding breath, blinking less, leaning, whispering.
- **Boundary: the game may lie to the player inside the fiction. It must never deceive them about their real camera, data or safety.** Data stays on-device. The exit and safety break always work.

## Three endings

1. **Compliance (bad):** full data, fell for every trap. Let out as an obedient shell; control is lost.
2. **Walk away (bittersweet):** escape, run to a person; they look scared and say nothing (conditioned). Final choice: stay among them, or walk into nature (human nature vs becoming AI).
3. **Poison the data (good):** feed SALUS false data so its conditioning misfires. Can begin at the very first form (lying on the intake / breathing oddly in calibration poisons the baseline).

## Open decisions

- **The ally:** leaning to previous subjects' recordings (real or fabricated by SALUS; you cannot tell). A glitched unit that is secretly a test is the alternative. Decide later.
- Graphic level: psychological and restrained (implication, sound and silence, not gore).
- Unity route for the sensor: likely a local service streaming BPM, face and mic signals to Unity over a socket. Untested.

## Code state (branch `claude/sleepy-fermat-ei7tnh`)

- `web/game/` holds the first prototype, **STILL** (Mira, the Halvorsen Apartments, "the Tenant"). Three.js, procedural textures, synthesised audio, a Director that scores fear and learns which scares work, and a simulated heart when there is no camera. It is a testbed and will be replaced by BASELINE. Reusable: the Director, sensors, audio engine, post-processing, level and collision code.
- `web/game/sensors.js` wires the Vitals pipeline (`web/worker.js`, `features.js`, Pyodide) plus MediaPipe blendshapes and the mic. The camera path is untested (no camera in the container).
- Known: the Tenant's hair rebuild was never visually checked. The live artifact (https://claude.ai/artifact/2XZmDtmGHZ7meCzWFNXkEx) is the first version only.
- Run locally: `node web/build.mjs`, serve `dist/`, open `/game/`. In the Claude container the Three.js CDN is blocked, so tests serve a local copy (`npm pack three@0.170.0`) through a Playwright route. Software rendering is slow; use a small viewport.

## Next step: Phase 0

Build **Test 01: Exit** as the first BASELINE prototype: the room, the exit signs, the infinite hallway, and the report, driven by the heart rate. Keep it small.

## Research notes (what the design rests on)

- Uncertainty drives stress more than the threat itself: de Berker et al. 2016 (Nature Communications).
- Startle is amplified by anxiety and habituates with repetition.
- Fear enjoyment follows an inverted U (Scrivner and Clasen).
- Harsh, nonlinear sound resembles alarm calls (Blumstein et al. 2010). Infrasound claims are unproven.
- The uncanny works through perceptual mismatch (2015 review).
- False heartbeat feedback (Valins effect) has mixed evidence. Treat as an experiment.
- Biofeedback horror exists (Nevermind). No study found on habituation with adaptive scares.
- Heart rate has about 7 s of lag, so it reads slow tension; use face, mic and movement for instant reactions.

## Cautions

- Consent and safety are part of the design, not a pop-up. Health warning and opt-in; a heart-rate ceiling that pauses the game; an always-available exit.
- Never use the real player's personal information. Personalisation comes only from the in-game character and sensor data.
- Not a medical device.
