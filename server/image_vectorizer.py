"""Generate compressed Inception features for a local image dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

from inception import SERVER_DIR, extract_feature, load_inception_graph

DEFAULT_DATASET = SERVER_DIR / "database" / "dataset"
DEFAULT_OUTPUT = SERVER_DIR / "data" / "features.npz"
DEFAULT_INDEX = SERVER_DIR / "data" / "image_paths.json"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    image_paths = sorted(
        path
        for path in dataset.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No supported images found in {dataset}")

    graph, bottleneck_tensor, jpeg_tensor = load_inception_graph()
    features: list[np.ndarray] = []
    with tf.Session(graph=graph, config=tf.ConfigProto(device_count={"GPU": 0})) as session:
        for number, image_path in enumerate(image_paths, start=1):
            features.append(
                extract_feature(session, image_path, jpeg_tensor, bottleneck_tensor)
            )
            if number % 100 == 0 or number == len(image_paths):
                print(f"Processed {number}/{len(image_paths)} images")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=np.asarray(features, dtype=np.float32))
    relative_paths = []
    for path in image_paths:
        try:
            stored_path = path.relative_to(SERVER_DIR).as_posix()
        except ValueError:
            stored_path = str(path)
        relative_paths.append(stored_path)
    args.index.write_text(
        json.dumps(relative_paths, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Features saved to {args.output}")
    print(f"Image index saved to {args.index}")


if __name__ == "__main__":
    main()

