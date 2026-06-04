"""
Violence Detection — Flask Backend
=====================================
Serves the web UI and exposes a /predict endpoint that accepts a video
file upload, runs the ConvLSTM model, and returns a JSON result.

Usage (local):
    python app.py

Usage (Colab / Kaggle via localtunnel):
    See 3_deploy.ipynb
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH      = "model.keras"
IMG_H, IMG_W    = 64, 64
SEQ_LEN         = 60
CLASSES         = ["NonViolence", "Violence"]
UPLOAD_FOLDER   = "uploads"
ALLOWED_EXTS    = {"mp4", "avi", "mov", "mkv", "webm"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app   = Flask(__name__, static_folder="static")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")


# ── Frame extraction (same pipeline as training) ────────────────────────────
def extract_frames(video_path, seq_len=SEQ_LEN, img_h=IMG_H, img_w=IMG_W):
    """Uniformly sample seq_len frames from a video and return as a float32 array."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration    = frame_count / fps if fps > 0 else 0

    if duration < 1.0:
        cap.release()
        return None

    interval_ms = (duration * 1000) / seq_len
    frames      = []
    timestamp   = 0.0

    while timestamp < duration * 1000 and len(frames) < seq_len:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp)
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (img_w, img_h))
        frames.append(frame)
        timestamp += interval_ms

    cap.release()

    if len(frames) == seq_len - 1:
        frames.append(frames[-1])
    if len(frames) != seq_len:
        return None

    arr = np.asarray(frames, dtype=np.float32) / 255.0
    return arr   # (60, 64, 64, 3)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    ext = video_file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported format: .{ext}"}), 400

    # Save upload temporarily
    save_path = os.path.join(UPLOAD_FOLDER, "input." + ext)
    video_file.save(save_path)

    # Extract frames
    frames = extract_frames(save_path)
    if frames is None:
        return jsonify({"error": "Could not extract frames — video may be too short or corrupted"}), 422

    # Predict
    x     = frames[np.newaxis, ...]          # (1, 60, 64, 64, 3)
    probs = model.predict(x, verbose=0)[0]   # (2,)

    result = {
        "label":      CLASSES[int(np.argmax(probs))],
        "confidence": float(np.max(probs)),
        "scores": {
            "NonViolence": float(probs[0]),
            "Violence":    float(probs[1]),
        }
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
