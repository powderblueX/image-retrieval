"""Inception-feature image similarity search."""

import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

from inception import SERVER_DIR, extract_feature, load_inception_graph

FEATURES_PATH = SERVER_DIR / "data" / "features.npz"
INDEX_PATH = SERVER_DIR / "data" / "image_paths.json"
RESULT_DIR = SERVER_DIR / "static" / "result"


class ImageSearchEngine:
    def __init__(self) -> None:
        if not FEATURES_PATH.exists() or not INDEX_PATH.exists():
            raise FileNotFoundError(
                "Feature index not found. Run `python server/image_vectorizer.py`."
            )

        with np.load(FEATURES_PATH) as archive:
            self.features = archive["features"].astype(np.float32)
        paths = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        self.image_paths = [
            path if (path := Path(value)).is_absolute() else SERVER_DIR / path
            for value in paths
        ]
        if len(self.features) != len(self.image_paths):
            raise ValueError("Feature count does not match the image index.")

        self.graph, self.bottleneck_tensor, self.jpeg_tensor = load_inception_graph()
        self.session = tf.Session(
            graph=self.graph,
            config=tf.ConfigProto(device_count={"GPU": 0}),
        )

    def recommend(self, query_path: Path, count: int = 9) -> list[Path]:
        query = extract_feature(
            self.session,
            query_path,
            self.jpeg_tensor,
            self.bottleneck_tensor,
        )
        feature_norms = np.linalg.norm(self.features, axis=1)
        query_norm = np.linalg.norm(query)
        similarities = (self.features @ query) / np.maximum(
            feature_norms * query_norm, 1e-12
        )
        indices = np.argsort(-similarities)[: min(count, len(self.image_paths))]

        if RESULT_DIR.exists():
            shutil.rmtree(RESULT_DIR)
        RESULT_DIR.mkdir(parents=True)

        results: list[Path] = []
        for position, index in enumerate(indices):
            source = self.image_paths[int(index)]
            destination = RESULT_DIR / f"{position:02d}_{source.name}"
            shutil.copy2(source, destination)
            results.append(destination)
        return results

    def close(self) -> None:
        self.session.close()

