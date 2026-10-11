"""Validated settings for the SSD MobileNet video workflow."""

from dataclasses import dataclass
from pathlib import Path

from ..validation import validate_integer, validate_nonnegative
from .labels import COCO_LABEL_MAP


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "ssd_mobilenet_v2_fpnlite_640x640"
DEFAULT_MODEL = PROJECT_ROOT / "models" / MODEL_NAME
DEFAULT_VIDEO = PROJECT_ROOT / "data/videos/vtest.avi"


# --------------------------------------
# 1. Define Detector Settings
# --------------------------------------

@dataclass(frozen=True)
class DetectionConfig:

    model_dir: Path = DEFAULT_MODEL

    score_threshold: float = 0.3

    max_detections: int = 100

    warmup: int = 5

    classes: tuple[str, ...] = ()

    # --------------------------------------
    # 1. Validate Detector Settings
    # --------------------------------------

    def __post_init__(self):

        object.__setattr__(
            self,
            "model_dir",
            Path(self.model_dir).expanduser().resolve(),
        )

        if isinstance(self.classes, str):
            raise ValueError("Classes must be a sequence of class names")

        object.__setattr__(self, "classes", tuple(self.classes))

        validate_nonnegative(self.score_threshold, "Score threshold")

        if self.score_threshold > 1:
            raise ValueError("Score threshold must be between 0 and 1")

        validate_integer(self.max_detections, "Maximum detections")

        validate_integer(self.warmup, "Warmup", minimum=0)

        unknown = set(self.classes) - set(COCO_LABEL_MAP.values())

        if unknown:
            raise ValueError(f"Unknown COCO classes: {', '.join(sorted(unknown))}")
