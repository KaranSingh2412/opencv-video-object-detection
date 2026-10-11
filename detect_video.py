"""Choose camera or video input, then save object-detection results."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2

from src.object_detection.config import (
    DEFAULT_MODEL,
    DEFAULT_VIDEO,
    PROJECT_ROOT,
    DetectionConfig,
)
from src.validation import validate_integer, validate_nonnegative


# --------------------------------------
# 1. Define Command-Line Options
# --------------------------------------

def build_parser():

    parser = argparse.ArgumentParser(
        description="Choose camera or video input and save annotated TensorFlow SSD detections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inputs = parser.add_mutually_exclusive_group()

    inputs.add_argument(
        "--source",
        type=Path,
        help="Original video file; skip the startup menu",
    )

    inputs.add_argument(
        "--camera",
        type=int,
        help="Camera index, usually 0; skip the startup menu",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Model folder or SavedModel folder",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Annotated .mp4; automatically named in results/videos when omitted",
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Minimum confidence, 0 to 1",
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        help="Keep selected COCO names, e.g. person car",
    )

    parser.add_argument(
        "--max-detections",
        type=int,
        default=100,
        help="Maximum detections per frame",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional positive frame limit",
    )

    parser.add_argument(
        "--duration",
        type=float,
        help="Stop recording or processing after this many seconds",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Untimed warmup calls; 0 disables warmup",
    )

    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Preview window: on for camera, off for video; --no-display disables it",
    )

    return parser


# --------------------------------------
# 2. Ask for Camera or Video Input
# --------------------------------------

def choose_source(args):

    if args.source is not None:
        return args.source

    if args.camera is not None:
        validate_integer(args.camera, "Camera index", minimum=0)

        return args.camera

    print("\nChoose your input:")
    print("  1. Camera")
    print("  2. Video file")

    while True:
        choice = input("Enter 1 or 2: ").strip().lower()

        if choice in ("1", "camera", "webcam"):
            while True:
                camera_text = input("Camera index [0]: ").strip()

                try:
                    camera_index = int(camera_text or "0")

                    validate_integer(camera_index, "Camera index", minimum=0)
                except ValueError:
                    print("Enter a camera number such as 0 or 1.")

                    continue

                return camera_index

        if choice in ("2", "video", "video file"):
            while True:
                video_text = input("Video path [Enter to use the sample video]: ").strip()

                # Accept a quoted path pasted from a file manager.

                if len(video_text) >= 2 and video_text[0] == video_text[-1] and video_text[0] in ("'", '"'):
                    video_text = video_text[1:-1]

                source = Path(video_text).expanduser() if video_text else DEFAULT_VIDEO

                if not source.is_file():
                    print(f"Video not found: {source}. Enter an existing video file.")

                    continue

                return source

        print("Please choose 1 for Camera or 2 for Video file.")


# --------------------------------------
# 3. Choose Result File Name
# --------------------------------------

def choose_output(source, requested_output):

    if requested_output is not None:
        return requested_output

    if isinstance(source, int):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")

        name = f"camera_{source}_{timestamp}.mp4"
    else:
        name = f"{source.stem}_detected.mp4"

    return PROJECT_ROOT / "results/videos" / name


# --------------------------------------
# 4. Run Main Program
# --------------------------------------

def main(argv=None):

    args = build_parser().parse_args(argv)

    try:

        # Prepare Detector Settings

        config = DetectionConfig(
            model_dir=args.model,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            warmup=args.warmup,
            classes=tuple(args.classes or ()),
        )

        if args.max_frames is not None:
            validate_integer(args.max_frames, "Maximum frames")

        if args.duration is not None:
            validate_nonnegative(args.duration, "Duration")

            if args.duration == 0:
                raise ValueError("Duration must be greater than zero")

        # Select Input and Prepare Result File

        source = choose_source(args)

        is_camera = isinstance(source, int)

        display = is_camera if args.display is None else args.display

        output = choose_output(source, args.output)

        if is_camera:
            if display:
                print("\nCamera mode: press Q or Esc in the preview to stop and save.")

            print("Ctrl+C also stops and saves an active camera recording.")

        print(f"Result file: {output}")

        # Process Input and Save Results

        from src.object_detection.video import run_video

        report = run_video(
            source,
            output,
            config,
            max_frames=args.max_frames,
            duration=args.duration,
            display=display,
        )
    except EOFError:
        print(
            "Error: No startup choice received. Use --source /path/to/video.mp4 or --camera 0.",
            file=sys.stderr,
        )

        return 1
    except (OSError, ValueError, TypeError, RuntimeError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)

        return 1
    except KeyboardInterrupt:
        print("Cancelled before a result could be saved.", file=sys.stderr)

        return 130

    print(f"\nCompleted: {report['frames_processed']} frames")

    print(f"Video: {report['artifacts']['video']}")

    print(f"Report: {report['artifacts']['summary']}")

    return 0


# --------------------------------------
# 5. Start Program
# --------------------------------------

if __name__ == "__main__":
    sys.exit(main())
