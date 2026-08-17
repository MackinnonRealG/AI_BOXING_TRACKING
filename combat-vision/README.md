# Combat Vision

AI camera analysis for **boxing and kickboxing**: point one or more cameras at the fighters and measure punch speed, estimated power, strike types, footwork, stance switches, combinations, and inter-fighter distance — live or from recorded footage — with per-fighter progression tracked across sessions.

Beyond logging *what* happened, it also coaches *how well*: live, actionable technique cues — guard dropping, punching without hip rotation, locked-knee/no-leg-drive punches — plus a guided drill mode (`d` in live mode) that prompts a combo and grades whether you threw it clean, and personal-best tracking that compares each session against your own history instead of a fixed threshold.

## Setup

Requires Python 3.11 or 3.12 (MediaPipe does not yet ship wheels for newer versions).

```bash
cd combat-vision
python3.12 -m venv .venv          # or: uv venv --python 3.12
.venv/bin/pip install -e ".[dev]" # or: uv pip install -p .venv/bin/python -e ".[dev]"
```

The MediaPipe pose model (`pose_landmarker_lite.task`, ~5 MB) is downloaded automatically to `~/.cache/combat_vision/` on first run.

> **macOS note:** if `import combat_vision` fails with `ModuleNotFoundError` even though the editable install succeeded, some macOS setups re-apply the `hidden` file flag to `.pth` files, which Python 3.12 skips. Workaround: `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`, or add a `sitecustomize.py` in site-packages appending `<repo>/src` to `sys.path`.

## Running

```bash
# Live mode: webcam + real-time overlay
# keys: q quits, h toggles the foot heat map, t toggles the tracker backend
# (supervision/ByteTrack <-> built-in centroid; also settable via tracking.backend in config)
# s flips between boxing and kickboxing live -- every engine picks up which
#   strikes/faults it should monitor immediately, no restart needed
# d starts/stops a guided drill for fighter A, cycling through built-in combos
combat-vision live --sport boxing --camera 0
combat-vision live --sport kickboxing --rtsp rtsp://192.168.1.20/stream

# Review mode: analyse recorded sparring into a session report
combat-vision review sparring.mp4 --sport kickboxing --output report.json
```

Every tunable (thresholds, smoothing, camera, calibration) lives in `config/default.yaml`; pass `--config my.yaml` to override. To get **metric speeds (m/s)** instead of px/s, set `calibration.reference_length_m` / `reference_length_px` from a known length in frame (e.g. the ring rope span).

```bash
# Quality gates
.venv/bin/pytest && .venv/bin/ruff check . && .venv/bin/mypy
```

Everything above is verified against synthetic pose fixtures only — this has
never been run against a real camera in development. Before trusting any of
it, see [`docs/first_real_test.md`](docs/first_real_test.md) for a short,
structured first session and what to check in the results.

## Architecture

One pipeline serves both modes; **mode only changes the source and the sink**:

```
capture → detection & pose → tracking (IDs) → smoothing → metrics engines → event bus → sinks
(webcam/file/rtsp) (PoseBackend)  (Fighter A/B)  (One-Euro)  (speed, power, …)          (overlay UI | report | SQLite)
```

Design rules that keep the system extensible:

- **Typed contracts everywhere** (`events/types.py`): stages exchange frozen dataclasses (`TrackedPose`, `StrikeEvent`, `StepEvent`, …), never raw arrays. Coordinates are normalized [0,1]; physical units come from the `Calibration` object, with graceful px/s fallback when uncalibrated.
- **Sport as configuration** (`sports/`): a `SportProfile` declares active strike classes, striking limbs, and target zones. The pipeline never branches on sport name — adding Muay Thai means adding one profile file.
- **Pluggable pose backend** (`pose/`): `PoseBackend` maps any model's landmarks onto one canonical `KeypointName` vocabulary. Default is MediaPipe PoseLandmarker (tasks API, multi-person); a YOLOv8-pose stub documents the swap path.
- **Engines are bus-isolated** (`engines/`): each engine consumes the tracked-pose stream (or other engines' events — the combination engine consumes `StrikeEvent`s) and publishes typed events. They never call each other, so each is unit-testable by replaying recorded pose sequences (`tests/fixtures/`).
- **Analytics reads storage only** (`analytics/`): trends and pattern recognition work from persisted events with no camera attached. Raw events are stored per session (SQLite/SQLAlchemy, alembic scaffolded), so aggregates can always be recomputed.

### Module map

| Module | Status | Purpose |
|---|---|---|
| `capture/` | ✅ | `CameraSource` interface; webcam, video file, RTSP; multi-camera-ready frames |
| `pose/` | ✅ mediapipe / ✅ yolov8 (optional extra) | Pose backends behind one interface; YOLOv8 needs `pip install "combat-vision[yolo]"` |
| `tracking/` | ✅ | ByteTrack (supervision) by default with live-toggleable centroid fallback; stable A/B identities through occlusion; appearance-based re-ID (color-histogram descriptor) disambiguates which label to recycle onto a reappearing fighter after a long occlusion — supervision tracker only, see Roadmap |
| `filtering/` | ✅ | One-Euro filter per keypoint (speed-adaptive smoothing) |
| `sports/` | ✅ | Boxing + kickboxing profiles, hot-swappable at runtime (`SwitchableSportProfile`, live `s` key) |
| `engines/speed` | ✅ tested | Limb velocity (wrists; +ankles/knees in kickboxing), hysteresis stroke detection → candidates |
| `engines/strike_classifier` | ✅ tested | Heuristic classification: jab/cross/hook/uppercut + kicks/knees, landed detection vs opponent zones |
| `engines/power` | ✅ tested | 0–100 *estimated* power: speed + limb extension + torso rotation |
| `engines/stance` | ✅ tested | Orthodox/southpaw/square with debounce; switch log with timestamps |
| `engines/footwork` | ✅ tested | Step detection, stance width, weight shift, per-fighter heat map |
| `engines/distance` | ✅ tested | Decimated inter-fighter distance samples |
| `engines/combination` | ✅ tested | Gap-based strike chaining → most-used sequences |
| `engines/guard` | ✅ tested | Per-hand guard-height fault: sustained hand drop below chin line → live cue |
| `engines/elbow` | ✅ tested | Per-arm elbow-tuck fault: elbow flared out from the torso centerline → live cue |
| `engines/rotation` | ✅ tested | Hip-shoulder separation: shoulders turned without matching hip turn → "arm punch" fault |
| `engines/knee_bend` | ✅ tested | Locked-knee posture + no-leg-drive punches (both knees straight at stroke start) |
| `engines/kick_balance` | ✅ tested | Base (standing) leg lateral wobble during kicks/knees — kickboxing only |
| `engines/head_posture` | ✅ tested | Head-roll (eye line vs shoulder line) — measurement only, not graded as a fault |
| `engines/depth_posture` | ✅ tested | *Approximate* elbow-flare / torso-lean from MediaPipe's unused `z` channel — measurement only |
| `calibration/` | ✅ v1 | Reference-length px→m scale; two-camera DLT triangulation math (`triangulation.py`, tested against synthetic geometry — not yet wired to live capture, see Roadmap) |
| `events/` | ✅ | Typed events + synchronous bus |
| `storage/` | ✅ | Fighters/sessions/rounds/events schema + repository + alembic |
| `analytics/` | ✅ tested | Session reports (incl. technique-fault counts + coaching notes), cross-session trends (incl. fault rate), per-fighter personal-best baselines, pattern recognition (median-split association mining over labelled rounds) |
| `review/` | ✅ | Video → full pipeline → JSON + text report + persistence + personal-best notes |
| `drills.py` / `ui/drill_coach.py` | ✅ tested | Guided drill mode: built-in combo library + countdown/active/result state machine |
| `ui/` | ✅ overlay / 🔲 web | OpenCV overlay (skeleton incl. eyes/ears, strike/power/stance/guard/knee stats, live fault cues, drill prompts, heat map); FastAPI `/stats` websocket stub |

## Speed engine (the reference implementation)

`engines/speed.py` is the template for the remaining engines:

1. Wrist positions → pixel space → frame-to-frame speed, smoothed by a short moving average (window in config).
2. A **hysteresis state machine** turns the speed signal into strokes: open above `start_speed`, track the peak, close below `end_speed_ratio × peak`. One punch = one event, even with noisy speed.
3. Guards: minimum peak (`peak_min_speed`), maximum stroke duration (rejects tracking glides), and a debounce interval (the jab's retraction never double-counts).

Its tests replay synthetic fixtures with known ground truth (a jab whose true peak is exactly 6.0 m/s), asserting detection count, peak accuracy, jitter rejection, and the uncalibrated px/s fallback. Regenerate fixtures with `python tests/fixtures/generate_fixtures.py`.

## Roadmap

1. **Learned strike classifier** — replace the geometric heuristics with a small temporal model (1D-CNN over keypoint windows) trained on labelled sparring clips; today's classifier generates exactly that labelled data. **Blocked on data**: no labelled clip corpus exists yet — this needs real sparring footage run through review mode and hand-corrected before any training code is worth writing.
2. **Two-fighter hardening (appearance-based re-ID)** — done for the default tracker: `pose/appearance.py` computes a normalized HSV hue histogram per detection, `PersonDetection.appearance` carries it, and `SupervisionTracker` prefers the closest appearance match when more than one label is simultaneously eligible for recycling (both fighters lost, then both reappear) instead of picking whichever slot comes first. This is a color histogram, not a learned embedding — it will **not** reliably distinguish two fighters in near-identical kit (same-color rash guards, etc.), and the centroid fallback tracker (`FighterTracker`) doesn't attempt recycling at all (different design, untouched). Both are known, accepted limits of this v1, not bugs.
3. **Web UI** — replace the OpenCV window via the existing FastAPI websocket stub. A substantial separate frontend effort, not attempted here.
4. **Pattern recognition v2** — gradient-boosted trees + SHAP once ~100 labelled rounds exist (v1 median-split mining ships now).
5. **Multi-camera calibration/fusion** — the triangulation math (two-view DLT) is done and tested against synthetic camera geometry: see `calibration/triangulation.py`. Live fusion remains incomplete and requires physical cameras (not just code) to build and validate: per-camera intrinsic calibration from real checkerboard captures, extrinsics from shared scene references, and synchronized dual-camera capture wired into the pipeline. RTSP reconnection policy is also still open.
6. **Personal-baseline coverage** — currently speed-only per strike type; extending it to power score and technique-fault rate (not just hand speed) would sharpen the "am I actually improving" signal further.
