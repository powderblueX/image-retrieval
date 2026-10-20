"""Download and extract the TensorFlow Inception v3 graph."""

import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from inception import MODEL_PATH

MODEL_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/models/"
    "image/imagenet/inception-2015-12-05.tgz"
)


def main() -> None:
    if MODEL_PATH.exists():
        print(f"Model already exists: {MODEL_PATH}")
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / "inception.tgz"
        print(f"Downloading model from {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, archive_path)
        with tarfile.open(archive_path) as archive:
            member = next(
                (
                    item
                    for item in archive.getmembers()
                    if Path(item.name).name == MODEL_PATH.name and item.isfile()
                ),
                None,
            )
            if member is None:
                raise RuntimeError("Downloaded archive does not contain the graph file.")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("Could not read the graph file from the archive.")
            with MODEL_PATH.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

