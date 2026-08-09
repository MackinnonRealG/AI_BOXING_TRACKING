# AI Boxing Tracking

**Combat Vision** — an AI system that analyses boxing and kickboxing through a camera: punch speed, estimated power, strike classification (jab/cross/hook/uppercut + kicks and knees), footwork and foot-placement heat maps, stance switches, two-fighter tracking and distance, combination mapping, session reports, and progression tracking across sessions.

➡️ **The full project, documentation, setup instructions, and architecture guide live in [`combat-vision/`](combat-vision/).**

Quick start:

```bash
cd combat-vision
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"

# Live camera analysis (q quits, h toggles the foot heat map)
.venv/bin/combat-vision live --sport boxing

# Analyse recorded sparring footage into a session report
.venv/bin/combat-vision review sparring.mp4 --sport kickboxing --output report.json
```

Built with Python 3.11/3.12, MediaPipe PoseLandmarker (YOLOv8-pose optional), OpenCV, SQLAlchemy, and FastAPI.
