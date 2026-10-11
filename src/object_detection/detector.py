"""Lazy TensorFlow SavedModel backend for the reference SSD MobileNet model."""

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from ..validation import validate_image, validate_integer
from .config import DetectionConfig
from .postprocessing import Detection, parse_detections


# --------------------------------------
# 1. Store Frame Detection Results
# --------------------------------------

@dataclass(frozen=True)
class FrameResult:

    detections: list[Detection]

    timings_ms: dict[str, float]


# --------------------------------------
# 2. Preprocess Video Frame
# --------------------------------------

def preprocess_frame(frame, input_size=(640, 640)):

    validate_image(frame, "Video frame")

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Video frame must have three BGR channels")

    resized = cv2.resize(frame, input_size, interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    return np.ascontiguousarray(rgb[np.newaxis])


# --------------------------------------
# 3. TensorFlow Object Detector
# --------------------------------------

class SavedModelDetector:

    """Load once, then return detections and timings without drawing or I/O."""

    # --------------------------------------
    # 1. Initialize Detector
    # --------------------------------------

    def __init__(self, config: DetectionConfig):

        self.config = config

        self._model = None

        self._signature = None

        self._tf = None

        self.input_size = (640, 640)

        self.input_name = None

        self.environment = {}

    # --------------------------------------
    # 2. Load TensorFlow Model
    # --------------------------------------

    def load(self):

        if self._signature is not None:
            return

        # Find SavedModel Folder

        model_dir = self.config.model_dir

        if not (model_dir / "saved_model.pb").is_file():
            model_dir = model_dir / "saved_model"

        if not (model_dir / "saved_model.pb").is_file():
            raise FileNotFoundError(
                f"SavedModel not found at {self.config.model_dir}. "
                "Run: python scripts/download_detection_assets.py --only model"
            )

        try:
            import tensorflow as tf
        except ImportError as error:
            raise RuntimeError("Install dependencies: python -m pip install -r requirements.txt") from error

        try:
            model = tf.saved_model.load(str(model_dir))
        except tf.errors.OpError as error:
            raise RuntimeError(f"Unable to load TensorFlow model: {error}") from error

        signature = model.signatures.get("serving_default")

        if signature is None:
            raise ValueError("SavedModel has no serving_default signature")

        # Check Model Input and Output

        specs = signature.structured_input_signature[1]

        if len(specs) != 1:
            raise ValueError("Expected one named SSD image input")

        name, spec = next(iter(specs.items()))

        if spec.dtype != tf.uint8 or spec.shape.rank != 4:
            raise ValueError("Expected a uint8 SSD image input with shape [1, H, W, 3]")

        batch, height, width, channels = spec.shape.as_list()

        if batch not in (None, 1) or channels != 3:
            raise ValueError("SSD signature must accept one three-channel image")

        required = {"detection_boxes", "detection_scores", "detection_classes", "num_detections"}

        if not required.issubset(signature.structured_outputs):
            raise ValueError("SavedModel does not expose standard SSD detection outputs")

        # Store Loaded Model Information

        self.input_size = (width or 640, height or 640)

        self._tf, self._model, self._signature, self.input_name = tf, model, signature, name

        self.environment = {
            "tensorflow": tf.__version__,
            "input_name": name,
            "input_shape": spec.shape.as_list(),
            "input_dtype": spec.dtype.name,
            "devices": [device.name for device in tf.config.list_logical_devices()],
        }

    # --------------------------------------
    # 3. Detect Objects and Measure Time
    # --------------------------------------

    def detect(self, frame):

        if self._signature is None:
            raise RuntimeError("Load the detector before calling detect")

        # Preprocess Input Frame

        start = perf_counter()

        batch = preprocess_frame(frame, self.input_size)

        tensor = self._tf.convert_to_tensor(batch, dtype=self._tf.uint8)

        preprocessed = perf_counter()

        try:

            # Run Model Inference

            raw = self._signature(**{self.input_name: tensor})
            # Materialize tensors before stopping the clock, including device sync.

            outputs = {name: value.numpy() for name, value in raw.items()}
        except self._tf.errors.OpError as error:
            raise RuntimeError(f"TensorFlow inference failed: {error}") from error

        inferred = perf_counter()

        # Convert Model Outputs to Detections

        detections = parse_detections(outputs, frame.shape, self.config)

        finished = perf_counter()

        # Return Detections and Timing

        return FrameResult(
            detections,
            {
                "preprocess_ms": (preprocessed - start) * 1000,
                "inference_ms": (inferred - preprocessed) * 1000,
                "postprocess_ms": (finished - inferred) * 1000,
                "detection_ms": (finished - start) * 1000,
            },
        )

    # --------------------------------------
    # 4. Warm Up Detector
    # --------------------------------------

    def warmup(self, frame, iterations):

        validate_integer(iterations, "Warmup", minimum=0)

        for _ in range(iterations):
            self.detect(frame)
