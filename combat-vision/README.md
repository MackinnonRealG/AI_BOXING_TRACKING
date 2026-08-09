# Combat Vision

AI camera analysis for **boxing and kickboxing**: point one or more cameras at the fighters and measure punch speed, estimated power, strike types, footwork, stance switches, combinations, and inter-fighter distance — live or from recorded footage — with per-fighter progression tracked across sessions.

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
# Live mode: webcam + real-time overlay (q quits, h toggles the foot heat map)
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
| `tracking/` | ✅ v1 | Greedy centroid tracker, occlusion tolerance, stable A/B identities |
| `filtering/` | ✅ | One-Euro filter per keypoint (speed-adaptive smoothing) |
| `sports/` | ✅ | Boxing + kickboxing profiles |
| `engines/speed` | ✅ tested | Limb velocity (wrists; +ankles/knees in kickboxing), hysteresis stroke detection → candidates |
| `engines/strike_classifier` | ✅ tested | Heuristic classification: jab/cross/hook/uppercut + kicks/knees, landed detection vs opponent zones |
| `engines/power` | ✅ tested | 0–100 *estimated* power: speed + limb extension + torso rotation |
| `engines/stance` | ✅ tested | Orthodox/southpaw/square with debounce; switch log with timestamps |
| `engines/footwork` | ✅ tested | Step detection, stance width, weight shift, per-fighter heat map |
| `engines/distance` | ✅ tested | Decimated inter-fighter distance samples |
| `engines/combination` | ✅ tested | Gap-based strike chaining → most-used sequences |
| `calibration/` | ✅ v1 | Reference-length px→m scale; multi-camera fusion stub |
| `events/` | ✅ | Typed events + synchronous bus |
| `storage/` | ✅ | Fighters/sessions/rounds/events schema + repository + alembic |
| `analytics/` | ✅ tested | Session reports, cross-session trends, pattern recognition (median-split association mining over labelled rounds) |
| `review/` | ✅ | Video → full pipeline → JSON + text report + persistence |
| `ui/` | ✅ overlay / 🔲 web | OpenCV overlay (skeleton, strike/power/stance stats, heat map); FastAPI `/stats` websocket stub |

## Speed engine (the reference implementation)

`engines/speed.py` is the template for the remaining engines:

1. Wrist positions → pixel space → frame-to-frame speed, smoothed by a short moving average (window in config).
2. A **hysteresis state machine** turns the speed signal into strokes: open above `start_speed`, track the peak, close below `end_speed_ratio × peak`. One punch = one event, even with noisy speed.
3. Guards: minimum peak (`peak_min_speed`), maximum stroke duration (rejects tracking glides), and a debounce interval (the jab's retraction never double-counts).

Its tests replay synthetic fixtures with known ground truth (a jab whose true peak is exactly 6.0 m/s), asserting detection count, peak accuracy, jitter rejection, and the uncalibrated px/s fallback. Regenerate fixtures with `python tests/fixtures/generate_fixtures.py`.

## Roadmap

1. **Learned strike classifier** — replace the geometric heuristics with a small temporal model (1D-CNN over keypoint windows) trained on labelled sparring clips; today's classifier generates exactly that labelled data.
2. **Two-fighter hardening** — appearance-based re-ID through clinches and long occlusions.
3. **Web UI** — replace the OpenCV window via the existing FastAPI websocket stub.
4. **Pattern recognition v2** — gradient-boosted trees + SHAP once ~100 labelled rounds exist (v1 median-split mining ships now).
5. RTSP reconnection policy, multi-camera calibration/fusion, homography-based calibration (current is a single scalar scale).
