"""Convert TensorFlow SSD output into validated frame-coordinate detections."""

from dataclasses import asdict, dataclass

import numpy as np

from .config import DetectionConfig
from .labels import COCO_LABEL_MAP


# --------------------------------------
# 1. Store Object Detection Data
# --------------------------------------

@dataclass(frozen=True)
class Detection:

    bbox: tuple[float, float, float, float]

    class_id: int

    class_name: str

    confidence: float

    # --------------------------------------
    # 1. Convert Detection to Dictionary
    # --------------------------------------

    def to_dict(self):

        return asdict(self)


# --------------------------------------
# 2. Process Model Detections
# --------------------------------------

def parse_detections(outputs, frame_shape, config: DetectionConfig):

    """Convert normalized yxyx boxes to clipped xyxy pixel coordinates.

    Invalid/nonfinite outputs are discarded. Filtering precedes top-k and
    scores are sorted explicitly instead of relying on backend ordering.
    The SavedModel already applies non-maximum suppression.
    """

    # Check Required Model Outputs

    required = ("detection_boxes", "detection_classes", "detection_scores", "num_detections")

    missing = set(required) - set(outputs)

    if missing:
        raise ValueError(f"Model output is missing: {', '.join(sorted(missing))}")

    # Read Output Arrays

    boxes = np.asarray(outputs["detection_boxes"])

    classes = np.asarray(outputs["detection_classes"])

    scores = np.asarray(outputs["detection_scores"])

    if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[2] != 4:
        raise ValueError("Expected detection_boxes with shape [1, N, 4]")

    if scores.ndim != 2 or classes.ndim != 2 or classes.shape != scores.shape or scores.shape != boxes.shape[:2]:
        raise ValueError("Expected matching detection scores/classes with shape [1, N]")

    # Read Number of Detections

    count = np.asarray(outputs["num_detections"]).ravel()

    if count.size != 1 or not np.isfinite(count[0]) or count[0] < 0 or count[0] != int(count[0]):
        raise ValueError("Invalid num_detections output")

    count = min(int(count[0]), boxes.shape[1])

    # Prepare Frame Dimensions and Class Filter

    height, width = frame_shape[:2]

    allowed = set(config.classes)

    detections = []

    # Validate and Convert Each Detection

    for box, class_value, score in zip(boxes[0, :count], classes[0, :count], scores[0, :count]):
        if not np.isfinite(box).all() or not np.isfinite(class_value) or not np.isfinite(score):
            continue

        if not 0 <= score <= 1 or score < config.score_threshold or class_value != int(class_value):
            continue

        class_id = int(class_value)

        class_name = COCO_LABEL_MAP.get(class_id)

        if class_name is None or (allowed and class_name not in allowed):
            continue

        y1, x1, y2, x2 = np.clip(box, 0, 1)

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(Detection(
            bbox=(float(x1 * width), float(y1 * height), float(x2 * width), float(y2 * height)),
            class_id=class_id, class_name=class_name, confidence=float(score),
        ))

    return sorted(detections, key=lambda detection: detection.confidence, reverse=True)[:config.max_detections]
