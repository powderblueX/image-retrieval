"""TensorFlow v1 compatibility helpers for the Inception feature extractor."""

from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

SERVER_DIR = Path(__file__).resolve().parent
MODEL_PATH = SERVER_DIR / "imagenet" / "classify_image_graph_def.pb"

BOTTLENECK_TENSOR_NAME = "pool_3/_reshape:0"
JPEG_DATA_TENSOR_NAME = "DecodeJpeg/contents:0"


def load_inception_graph() -> tuple[tf.Graph, tf.Tensor, tf.Tensor]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run `python server/download_model.py`."
        )

    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(MODEL_PATH.read_bytes())
        bottleneck, jpeg_data = tf.import_graph_def(
            graph_def,
            name="",
            return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME],
        )
    return graph, bottleneck, jpeg_data


def extract_feature(
    session: tf.Session,
    image_path: Path,
    jpeg_tensor: tf.Tensor,
    bottleneck_tensor: tf.Tensor,
) -> np.ndarray:
    values = session.run(
        bottleneck_tensor,
        feed_dict={jpeg_tensor: image_path.read_bytes()},
    )
    return np.squeeze(values).astype(np.float32)

