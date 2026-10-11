"""Fetch the official model and video with fixed SHA-256 checksums.

Model: TensorFlow Models, Apache-2.0.
https://github.com/tensorflow/models/blob/master/LICENSE
Sample: OpenCV 4.10.0 samples/data/vtest.avi.
https://github.com/opencv/opencv/blob/4.10.0/LICENSE
https://github.com/opencv/opencv/blob/4.10.0/COPYRIGHT
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "ssd_mobilenet_v2_fpnlite_640x640"
ARCHIVE_ROOT = MODEL_NAME + "_coco17_tpu-8"
MODEL_URL = f"https://download.tensorflow.org/models/object_detection/tf2/20200711/{ARCHIVE_ROOT}.tar.gz"
MODEL_SHA256 = "22d8c26d49218a8da9d650914b16e8d1cff3a7bc40df342011db582ac9f4ed92"
VIDEO_URL = "https://raw.githubusercontent.com/opencv/opencv/4.10.0/samples/data/vtest.avi"
VIDEO_SHA256 = "45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf"


# --------------------------------------
# 1. Calculate File Checksum
# --------------------------------------

def sha256(path):

    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# --------------------------------------
# 2. Download and Verify File
# --------------------------------------

def download_verified(url, destination, expected_hash):

    digest = hashlib.sha256()

    with urlopen(url, timeout=60) as response, Path(destination).open("wb") as handle:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(chunk)

            digest.update(chunk)

    if digest.hexdigest() != expected_hash:
        raise ValueError(f"Checksum mismatch for {url}")


# --------------------------------------
# 3. Extract Model Files
# --------------------------------------

def extract_model(archive_path, destination):

    """Extract only regular files in the expected model root; reject links."""

    destination = Path(destination)

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()

        for member in members:
            path = PurePosixPath(member.name)

            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != ARCHIVE_ROOT:
                raise ValueError("Model archive contains an unexpected path")

            if not member.isfile() and not member.isdir():
                raise ValueError("Model archive contains unsupported links or special files")

        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)

                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)

    model = destination / ARCHIVE_ROOT

    if not (model / "saved_model/saved_model.pb").is_file():
        raise ValueError("Model archive is missing saved_model.pb")

    return model


# --------------------------------------
# 4. Download Detection Model
# --------------------------------------

def download_model():

    destination = PROJECT_ROOT / "models" / MODEL_NAME

    if (destination / "saved_model/saved_model.pb").is_file():
        print(f"Model already present: {destination}")

        return

    if destination.exists():
        raise FileExistsError(f"Incomplete model folder exists: {destination}; move it aside before retrying")

    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        temporary = Path(temporary)

        archive = temporary / "model.tar.gz"

        print("Downloading SSD MobileNet V2 FPNLite 640 model...")

        download_verified(MODEL_URL, archive, MODEL_SHA256)

        model = extract_model(archive, temporary / "extracted")

        (model / "PROVENANCE.json").write_text(
            json.dumps({
                "name": MODEL_NAME,
                "url": MODEL_URL,
                "tarball_sha256": MODEL_SHA256,
                "license": "Apache-2.0 (TensorFlow Models)",
            }, indent=2) + "\n",
            encoding="utf-8",
        )

        model.rename(destination)

    print(f"Model saved: {destination}")


# --------------------------------------
# 5. Download Sample Video
# --------------------------------------

def download_video():

    destination = PROJECT_ROOT / "data/videos/vtest.avi"

    if destination.is_file():
        if sha256(destination) != VIDEO_SHA256:
            raise ValueError(f"Existing sample video has an unexpected checksum: {destination}")

        print(f"Sample video already present: {destination}")

        return

    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "vtest.avi"

        print("Downloading OpenCV sample video...")

        download_verified(VIDEO_URL, staged, VIDEO_SHA256)

        staged.replace(destination)

    print(f"Sample video saved: {destination}")


# --------------------------------------
# 6. Run Main Program
# --------------------------------------

def main(argv=None):

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--only", choices=("model", "video", "both"), default="both")

    args = parser.parse_args(argv)

    try:
        if args.only in ("model", "both"):
            download_model()

        if args.only in ("video", "both"):
            download_video()
    except (OSError, ValueError, tarfile.TarError) as error:
        print(f"Error: {error}", file=sys.stderr)

        return 1

    return 0


# --------------------------------------
# 7. Start Program
# --------------------------------------

if __name__ == "__main__":
    sys.exit(main())
