# First real-camera test

Everything in this project has been verified against synthetic pose fixtures
and unit tests — never against a real human in front of a real camera.
That's the single highest-priority gap right now: no amount of
additional heuristic tuning matters until we know where the real gap between
"passes the test suite" and "actually works" is.

This is the smallest test that closes that gap. Not a benchmark, not a demo —
a diagnostic. The goal is to generate one real session report and find out
what's wrong with it.

## 1. Before you start

```bash
cd combat-vision
.venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/mypy   # confirm the baseline is green
```

If `import combat_vision` fails despite an editable install succeeding, see
the macOS `.pth`-hidden-flag note in the README — a known environment quirk,
not a code problem.

Camera check: on macOS, the first `combat-vision live` run will prompt for
camera permission for your terminal app. Grant it before starting, or the
webcam source will silently return no frames.

## 2. Calibrate — don't skip this

Every speed/power number produced so far in development is unitless px/s
dressed up as if it meant something. Without calibration, `combat-vision`
still runs and still reports numbers, but "12.3 m/s" is not a real
measurement of anything — it's pixels per second with a unit label attached.

In live mode, press `c`, click two points a known real-world distance apart
in frame (e.g. shoulder width if you know it, or place a tape measure/ring
rope in shot), type the distance in metres, press enter. Do this **before**
throwing anything you want measured.

If you skip this, the session still produces a valid report — just in px/s,
and the strike-fault thresholds (guard height, elbow flare ratio, rotation
angles) are unitless/angle-based anyway and work identically either way.
Calibration only matters for speed/power numbers being physically real.

## 3. The script

Stand where your whole body is in frame (the HUD will tell you if it isn't).
Run through this in order — each part is designed to deliberately trigger one
specific thing, so we can tell whether that detector actually fires on a real
body, not just on synthetic fixtures shaped exactly like its own threshold
logic:

1. **30s stillness** — establishes a clean idle baseline. Nothing should fire.
2. **10 relaxed jabs and crosses**, guard up between each — should register as
   clean hip-turn/leg-drive/guard-up the whole way, no faults.
3. **Deliberately drop your guard** after a punch and hold it down for 2+
   seconds — should trigger the guard-drop cue within ~0.6s.
4. **Deliberately flare an elbow** out from your ribs and hold it — should
   trigger the elbow-flare cue.
5. **Throw 5 crosses/hooks arm-only, no hip turn** (stand square, punch with
   just the shoulder/arm) — should trigger the no-hip-turn fault.
6. **Throw 5 crosses/hooks with real hip drive** — should *not* fault, and
   should log as clean.
7. **Throw 5 punches standing flat-footed, knees locked straight** — should
   trigger the no-leg-drive fault.
8. **Throw 5 punches with normal knee bend** — should log clean.
9. **A short combo** (jab-cross-hook or similar) — check it chains correctly
   in the combination engine and doesn't double-count.
10. **Move around, switch stance if you can** — footwork/stance-switch check.
11. **If you can, have a second person step into frame** — checks two-fighter
    tracking, and walk one person fully out of frame for 3+ seconds then back
    in, to exercise the relabel/re-identification path for real.
12. **Try the drill mode** (`d` key) — follow the on-screen combo prompt and
    check the clean/broken grading matches what you actually threw.

Total: under 5 minutes. Don't warm up first — the point is to see raw,
slightly imperfect technique, not your best reps.

## 4. What to capture

- Review mode is easiest for analysis: `combat-vision review your_clip.mp4
  --sport boxing --output report.json`. This also saves to the local SQLite
  DB, so trend/baseline features have something to read.
- Or live mode: run `combat-vision live --sport boxing`, press `q` when done
  — it prints the same report to stdout and saves it.
- Keep the raw video if you can share it. The report tells us *what the
  pipeline concluded*; the video is the only way to check *whether it was
  right*.

## 5. What to check when reviewing the report

For each of the 12 steps above, compare what you actually did against what
the report says happened. Specifically flag:

- **False negatives**: you dropped your guard/flared an elbow/locked your
  knees and nothing fired.
- **False positives**: you did something correctly and it got flagged anyway.
- **Missed strikes**: punches you know you threw that don't appear in the
  count at all (see the note below — this is a known possible failure mode).
- **Misclassified strikes**: a jab reported as a cross, a hook reported as
  unknown, etc.
- **Identity problems**: if two people were in frame, did labels swap at any
  point, especially around the walk-out/walk-back-in test?

## 6. Known limitations going in — don't rediscover these, build on them

- **A punch that happens entirely while pose tracking loses your wrist
  (a clinch, a fast crossing motion, motion blur) will not be counted at
  all** — not misclassified, just silently absent. See
  `tests/test_adversarial_conditions.py` for what's already confirmed about
  this.
- **Low-confidence keypoints are trusted exactly the same as high-confidence
  ones.** No engine currently reads MediaPipe's per-keypoint confidence
  score. If false positives cluster around moments where you were
  partially out of frame or motion-blurred, this is very likely why —
  and it's a fixable, scoped change, not a mystery.
- **Guard height, elbow flare, and hip rotation are read from one 2D camera
  angle.** Camera placement will materially affect accuracy: side-on reads
  hip/shoulder rotation well and guard height less reliably; front-on is
  the reverse. Try to note (or vary) your camera angle when reviewing
  results — mismatches between steps 5/6 and 7/8 above are the most likely
  place camera angle shows up as the actual cause of a wrong reading,
  not a broken detector.
- **All fault thresholds in `config/default.yaml` are unvalidated
  defaults** — reasonable engineering guesses, never tuned against a real
  body. If something fires too eagerly or not eagerly enough, that's a
  config number to adjust, not necessarily a logic bug.

The output of this test is the input to the next round of work — whatever
breaks here is the actual priority list, not whatever seemed important from
reading the code.
