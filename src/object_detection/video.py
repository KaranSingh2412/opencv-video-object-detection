"""Video orchestration: stream frames, save artifacts, and release resources."""

from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from numbers import Integral
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import cv2
import numpy as np

from ..validation import validate_integer, validate_nonnegative
from .config import DetectionConfig
from .detector import SavedModelDetector
from .overlay import annotate_frame


STAGES = ("preprocess_ms", "inference_ms", "postprocess_ms", "detection_ms", "processing_ms")


# --------------------------------------
# 1. Calculate Timing Statistics
# --------------------------------------

def summarize_times(values):

    array = np.asarray(values, dtype=float)

    if array.size == 0:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "fps": None,
        }

    mean = float(array.mean())

    return {
        "count": int(array.size),
        "mean_ms": mean,
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "fps": 1000 / mean if mean > 0 else None,
    }


# --------------------------------------
# 2. Finalize Camera Recording
# --------------------------------------

def finalize_camera_video(capture_path, output_path, frame_count, elapsed_seconds):

    """Encode at the measured recording rate so playback matches its duration."""

    measured_fps = frame_count / elapsed_seconds

    # Keep the MPEG-4 timebase within the encoder's supported range.

    fps = min(65535.0, max(0.001, measured_fps))

    precision = max(0, min(3, int(np.floor(np.log10(65534 / fps)))))

    fps = round(fps, precision)

    capture = cv2.VideoCapture(str(capture_path))

    writer = None

    try:
        if not capture.isOpened():
            raise OSError("Unable to reopen the camera recording")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise OSError("Unable to finalize the camera result video")

        saved = 0

        while True:
            ok, frame = capture.read()

            if not ok:
                break

            writer.write(frame)

            saved += 1

        if saved != frame_count:
            raise OSError("Camera recording lost frames during final encoding")
    finally:
        capture.release()

        if writer is not None:
            writer.release()

    return fps


# --------------------------------------
# 3. Run Camera or Video Object Detection
# --------------------------------------

def run_video(
    source,
    output,
    config: DetectionConfig,
    *,
    max_frames=None,
    duration=None,
    display=False,
    detector=None,
):

    """Save MP4 + CSV + JSONL + JSON; publish files only after a successful run.

    ``detector`` may be injected for tests. Frame memory is bounded: detections
    and rows are streamed to disk, and only timing samples are retained.
    """

    # Validate Input and Recording Settings

    is_camera = isinstance(source, Integral)

    if is_camera:
        validate_integer(source, "Camera index", minimum=0)

        source = int(source)

        if not display and max_frames is None and duration is None:
            raise ValueError("Camera mode needs a preview or a limit: use --max-frames or --duration")
    else:
        source = Path(source).expanduser().resolve()

        if not source.is_file():
            raise FileNotFoundError(f"Video not found: {source}")

    output = Path(output).expanduser().resolve()

    if max_frames is not None:
        validate_integer(max_frames, "Maximum frames")

    if duration is not None:
        validate_nonnegative(duration, "Duration")

        if duration == 0:
            raise ValueError("Duration must be greater than zero")

    if output.suffix.lower() != ".mp4":
        raise ValueError("Output video must use an .mp4 extension")

    artifact_paths = {
        "video": output,
        "frames": output.with_suffix(".frames.csv"),
        "detections": output.with_suffix(".detections.jsonl"),
        "summary": output.with_suffix(".summary.json"),
    }

    if source in artifact_paths.values():
        raise ValueError("Output would overwrite the source video")

    if any(path.exists() and not path.is_file() for path in artifact_paths.values()):
        raise ValueError("An output path is a directory; choose another output filename")

    # Open Camera or Source Video

    capture = cv2.VideoCapture(source if is_camera else str(source))

    writer = None

    try:
        if not capture.isOpened():
            if is_camera:
                raise ValueError(f"Unable to open camera {source}. Check camera permissions or try another camera index.")

            raise ValueError(f"Unable to open video: {source}")

        decode_start = perf_counter()

        ok, frame = capture.read()

        decode_ms = (perf_counter() - decode_start) * 1000

        first_decode_ms = decode_ms

        if not ok or frame is None:
            raise ValueError(f"{'Camera' if is_camera else 'Video'} contains no readable frames: {source}")

        # Read Video Dimensions and Playback Rate

        height, width = frame.shape[:2]

        source_fps = float(capture.get(cv2.CAP_PROP_FPS))

        fps_fallback = not np.isfinite(source_fps) or source_fps <= 0

        fps = 30.0 if fps_fallback else source_fps

        if fps_fallback:
            if is_camera:
                print("[INFO] Camera FPS unavailable; playback will follow measured recording speed")
            else:
                print("[INFO] Source FPS unavailable; saving video at 30 FPS")

        # Load and Warm Up Detector

        detector = detector if detector is not None else SavedModelDetector(config)

        print(f"[INFO] Loading model: {config.model_dir}", flush=True)

        detector.load()

        detector.warmup(frame, config.warmup)

        # Use a fresh camera frame after model loading and warmup.

        if is_camera:
            decode_start = perf_counter()

            ok, frame = capture.read()

            decode_ms = (perf_counter() - decode_start) * 1000

            first_decode_ms = decode_ms

            if not ok or frame is None:
                raise ValueError("Camera stopped delivering frames during setup")

            if frame.shape != (height, width, 3):
                raise ValueError("Camera frame dimensions changed during setup")

        first_frame_time = perf_counter()

        frame_time = first_frame_time

        output.parent.mkdir(parents=True, exist_ok=True)

        # Prepare Temporary Output Files

        with tempfile.TemporaryDirectory(prefix=".ed-detection-", dir=output.parent) as temporary:
            temporary = Path(temporary)

            staged = {name: temporary / path.name for name, path in artifact_paths.items()}

            recording_path = temporary / "camera_capture.avi" if is_camera else staged["video"]

            writer = cv2.VideoWriter(
                str(recording_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

            if not writer.isOpened():
                raise OSError("Unable to create MP4 output; check OpenCV codec support and output permissions")

            # Initialize Timing and Object Counts

            timings = {stage: [] for stage in STAGES}

            counts = Counter()

            processed = 0

            processing_total_ms = 0.0

            stop_reason = "camera_stream_ended" if is_camera else "end_of_video"

            loop_start = perf_counter()

            with staged["frames"].open("w", newline="", encoding="utf-8") as csv_file, \
                    staged["detections"].open("w", encoding="utf-8") as detection_file:
                csv_writer = csv.DictWriter(
                    csv_file,
                    fieldnames=["frame", "timestamp_s", "objects", *STAGES],
                )

                csv_writer.writeheader()

                # Process Each Video Frame

                try:
                    while True:
                        started = perf_counter()

                        result = detector.detect(frame)

                        processing_fps = 1000 * processed / processing_total_ms if processing_total_ms > 0 else 0.0

                        annotated = annotate_frame(
                            frame,
                            result,
                            processed,
                            processing_fps,
                            config.score_threshold,
                        )

                        writer.write(annotated)

                        processing_ms = decode_ms + (perf_counter() - started) * 1000

                        processing_total_ms += processing_ms

                        # Save Per-Frame Timing and Detections

                        row = {
                            "frame": processed,
                            "timestamp_s": frame_time - first_frame_time if is_camera else processed / fps,
                            "objects": len(result.detections),
                            **result.timings_ms,
                            "processing_ms": processing_ms,
                        }

                        csv_writer.writerow(row)

                        detection_file.write(json.dumps({
                            "frame": processed,
                            "timestamp_s": row["timestamp_s"],
                            "detections": [detection.to_dict() for detection in result.detections],
                        }, allow_nan=False) + "\n")

                        counts.update(detection.class_name for detection in result.detections)

                        for stage in STAGES:
                            timings[stage].append(row[stage])

                        processed += 1

                        if processed == 1 or processed % 25 == 0:
                            print(
                                f"[INFO] Frame {processed}: {len(result.detections)} objects, "
                                      f"inference {result.timings_ms['inference_ms']:.1f} ms",
                                flush=True,
                            )

                        # Display Preview When Requested

                        if display:
                            cv2.imshow("EdgeLab Video Detection", annotated)

                            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                                stop_reason = "user_stopped"

                                break

                        # Check Frame Limit and Read Next Frame

                        if max_frames is not None and processed >= max_frames:
                            stop_reason = "frame_limit"

                            break

                        if duration is not None and perf_counter() - loop_start >= duration:
                            stop_reason = "duration_limit"

                            break

                        decode_start = perf_counter()

                        ok, frame = capture.read()

                        frame_time = perf_counter()

                        decode_ms = (perf_counter() - decode_start) * 1000

                        if not ok:
                            break

                        if frame is None or frame.shape != (height, width, 3):
                            raise ValueError("Video frame dimensions changed during processing")

                except KeyboardInterrupt:
                    if not is_camera or processed == 0:
                        raise

                    stop_reason = "user_stopped"

                    print("\n[INFO] Stopping camera and saving recorded frames...", flush=True)

            recording_elapsed = perf_counter() - loop_start + first_decode_ms / 1000

            capture.release()

            # Finish Encoding and Measure Total Time

            writer.release()

            writer = None

            loop_ms = (perf_counter() - loop_start) * 1000 + first_decode_ms

            if is_camera:
                print("[INFO] Finalizing camera result file...", flush=True)

                fps = finalize_camera_video(
                    recording_path,
                    staged["video"],
                    processed,
                    recording_elapsed,
                )
            # Confirm the encoder produced a readable video before publishing.

            # Verify Saved Video

            check = cv2.VideoCapture(str(staged["video"]))

            try:
                readable, _ = check.read()

                encoded_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))

                if not readable or encoded_frames != processed:
                    raise OSError(f"Saved video verification failed: expected {processed} frames, got {encoded_frames}")
            finally:
                check.release()

            # Build Run Summary

            settings = asdict(config)

            settings["model_dir"] = str(config.model_dir)

            settings["classes"] = list(config.classes)

            summary = {
                "source": f"camera:{source}" if is_camera else str(source),
                "source_type": "camera" if is_camera else "video",
                "recording_elapsed_seconds": recording_elapsed,
                "duration_limit_seconds": duration,
                "frames_processed": processed,
                "frame_size": {
                    "width": width,
                    "height": height,
                },
                "output_fps": fps,
                "fps_fallback_used": bool(fps_fallback),
                "stop_reason": stop_reason,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "settings": settings,
                "warmup_excluded": True,
                "environment": {
                    "python": platform.python_version(),
                    "opencv": cv2.__version__,
                    "numpy": np.__version__,
                    "platform": platform.platform(),
                    **getattr(detector, "environment", {}),
                },
                "timings": {stage: summarize_times(values) for stage, values in timings.items()},
                "loop_elapsed_ms": loop_ms,
                "loop_fps": processed * 1000 / loop_ms,
                "timing_scope": {
                    "detection_ms": "Preprocessing + synchronized inference + postprocessing; excludes drawing and video I/O",
                    "processing_ms": "Decode + detection + overlay + writer.write; excludes telemetry and display",
                    "loop_fps": "Processing loop including telemetry, optional display, and initial encoder flush; excludes load/warmup and camera re-encoding",
                    "hud_fps": "Cumulative processing FPS through the preceding frame; first frame shows 0",
                },
                "detections_per_class": dict(counts.most_common()),
                "count_scope": "Detection occurrences across frames, not distinct tracked objects",
                "artifacts": {name: str(path) for name, path in artifact_paths.items()},
                "video_notes": (
                    "Camera dimensions preserved; playback FPS follows measured recording speed; output is silent"
                    if is_camera else
                    "Original dimensions and source playback FPS; output is silent (OpenCV does not copy audio)"
                ),
            }

            staged["summary"].write_text(
                json.dumps(summary, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )

            # Publish Completed Output Files

            for name, path in artifact_paths.items():
                staged[name].replace(path)

            return summary
    finally:

        # Release Video Resources

        capture.release()

        if writer is not None:
            writer.release()

        if display:
            cv2.destroyAllWindows()
