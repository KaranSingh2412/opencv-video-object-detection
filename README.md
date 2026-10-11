# OpenCV Video Object Detection

This project runs object detection on a video file or camera stream with OpenCV and a TensorFlow SavedModel. It uses the SSD MobileNet V2 FPNLite 640x640 COCO model, draws detections and timing information on each frame, and publishes an annotated MP4 together with machine-readable run artifacts.

## Features

- Video-file and camera input modes.
- COCO class filtering, confidence thresholds, detection limits, frame limits, and duration limits.
- Optional preview window for camera or video processing.
- Streamed per-frame CSV and JSONL telemetry so long videos do not require all detections in memory.
- Atomic result publication and a readable-output verification step.
- SHA-256 verification for the model archive and OpenCV sample video.

## Prerequisites

- Python 3.10 or newer is recommended.
- A working C/C++ runtime supported by TensorFlow.
- OpenCV video codec support for MP4 output.
- For camera mode, an accessible camera and OS camera permission.

TensorFlow and OpenCV can be resource-intensive. The default model and sample video are downloaded locally and are intentionally not tracked by Git.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_detection_assets.py
```

The last command downloads the pinned SSD MobileNet model and the OpenCV `vtest.avi` sample, verifies both SHA-256 checksums, and stores them in:

```text
models/ssd_mobilenet_v2_fpnlite_640x640/
data/videos/vtest.avi
```

Download only one asset when needed:

```bash
python scripts/download_detection_assets.py --only model
python scripts/download_detection_assets.py --only video
```

## Usage

Process the bundled sample video without opening a preview window:

```bash
python detect_video.py --source data/videos/vtest.avi --no-display
```

Process only the first 100 frames and keep people and cars:

```bash
python detect_video.py \
  --source data/videos/vtest.avi \
  --classes person car \
  --max-frames 100 \
  --no-display
```

Use a custom video and output path:

```bash
python detect_video.py \
  --source /path/to/input.mp4 \
  --output results/videos/custom_detected.mp4 \
  --score-threshold 0.5 \
  --no-display
```

Use camera index `0`. Camera mode displays a preview by default; press `Q` or `Esc` to stop and save, or provide a frame/time limit for non-interactive use:

```bash
python detect_video.py --camera 0 --duration 30
python detect_video.py --camera 0 --max-frames 300 --no-display
```

Run `python detect_video.py --help` for all options.

## Outputs

When `--output` is omitted, results are written under `results/videos/`. For an input named `vtest.avi`, the default output is:

```text
results/videos/vtest_detected.mp4
results/videos/vtest_detected.frames.csv
results/videos/vtest_detected.detections.jsonl
results/videos/vtest_detected.summary.json
```

The MP4 contains the annotated frames. The CSV contains frame timing and object counts; the JSONL contains detections in frame coordinates; and the summary JSON records settings, environment, timing statistics, stop reason, and artifact paths. OpenCV does not copy source audio.

## Validation

The repository has no heavyweight test suite. These focused checks validate the Python syntax and the CLI parser without downloading or loading the model:

```bash
python -m compileall -q detect_video.py scripts src
python detect_video.py --help
```

Asset download and end-to-end inference require network access, installed dependencies, the downloaded model, and (for video processing) a readable input video.

## Project layout

```text
.
├── detect_video.py                  # Command-line entry point
├── requirements.txt                # Runtime dependencies
├── scripts/
│   └── download_detection_assets.py # Pinned, checksum-verified asset setup
├── src/
│   ├── validation.py                # Shared input/config validation
│   └── object_detection/
│       ├── config.py                # Defaults and DetectionConfig
│       ├── detector.py              # TensorFlow SavedModel backend
│       ├── labels.py                # COCO category mapping
│       ├── overlay.py               # Frame annotation
│       ├── postprocessing.py         # Detection parsing/filtering
│       └── video.py                  # Video orchestration and artifacts
├── data/videos/                     # Downloaded input samples (ignored)
├── models/                          # Downloaded SavedModel (ignored)
└── results/videos/                  # Generated outputs (ignored)
```

The `data/videos/`, `models/`, and `results/videos/` directories contain only
`.gitkeep` placeholders in the repository. Downloaded assets and generated
outputs remain local and are excluded from Git by design.

The model and sample-video sources, URLs, checksums, and license references are recorded in `scripts/download_detection_assets.py`. This repository does not add a project license.
