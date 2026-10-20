"""Flask web application for image similarity search."""

import atexit
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from inception import SERVER_DIR
from search import ImageSearchEngine

UPLOAD_DIR = SERVER_DIR / "uploads"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@lru_cache(maxsize=1)
def get_search_engine() -> ImageSearchEngine:
    return ImageSearchEngine()


def close_search_engine() -> None:
    if get_search_engine.cache_info().currsize:
        get_search_engine().close()


atexit.register(close_search_engine)


@app.get("/")
def index():
    return render_template("main.html")


@app.post("/imgUpload")
def upload_image():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify(error="No image was selected."), 400

    suffix = Path(secure_filename(uploaded.filename)).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify(error="Only PNG and JPEG images are supported."), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    uploaded.save(upload_path)
    try:
        results = get_search_engine().recommend(upload_path)
    finally:
        upload_path.unlink(missing_ok=True)

    return jsonify(
        {
            f"image{position}": url_for(
                "static", filename=f"result/{result.name}"
            )
            for position, result in enumerate(results)
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

