"""Draw readable object labels and timing without changing detector inputs."""

import cv2
import numpy as np


# --------------------------------------
# 1. Choose Object Class Color
# --------------------------------------

def class_color(class_id):

    hue = int(class_id * 137.508) % 180

    hsv = np.array([[[hue, 200, 255]]], dtype=np.uint8)

    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


# --------------------------------------
# 2. Draw Boxes, Labels, and Timing
# --------------------------------------

def annotate_frame(frame, result, frame_index, pipeline_fps, threshold):

    # Copy Frame and Set Text Style

    annotated = frame.copy()

    height, width = frame.shape[:2]

    scale = max(0.35, min(0.7, width / 1400))

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw Object Boxes and Labels

    for detection in result.detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]

        x1, x2 = np.clip([x1, x2], 0, width - 1).tolist()

        y1, y2 = np.clip([y1, y2], 0, height - 1).tolist()

        color = class_color(detection.class_id)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{detection.class_name} {detection.confidence:.2f}"

        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, 1)

        label_x = max(0, min(x1, width - text_width - 4))

        label_y = max(text_height + baseline + 4, y1)

        cv2.rectangle(
            annotated,
            (label_x, label_y - text_height - baseline - 4),
            (min(width - 1, label_x + text_width + 4), label_y),
            color,
            -1,
        )

        cv2.putText(
            annotated,
            label,
            (label_x + 2, label_y - baseline - 2),
            font,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Prepare Timing Panel Text

    lines = [
        "SSD MobileNet V2 FPNLite | TensorFlow",
        f"Inference {result.timings_ms['inference_ms']:.1f} ms | Detection {result.timings_ms['detection_ms']:.1f} ms",
        f"Processing FPS {pipeline_fps:.1f} | Objects {len(result.detections)}",
        f"Frame {frame_index} | Confidence threshold {threshold:.2f}",
    ]

    # Draw Timing Panel

    line_height = max(16, int(35 * scale))

    panel_height = min(height, line_height * len(lines) + 12)

    panel_width = min(
        width,
        max(cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines) + 16,
    )

    roi = annotated[:panel_height, :panel_width]

    roi[:] = cv2.addWeighted(roi, 0.4, np.zeros_like(roi), 0.6, 0)

    for index, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (8, 8 + (index + 1) * line_height - 4),
            font,
            scale,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    return annotated
