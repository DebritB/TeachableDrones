"""
server.py
---------
GestureFly 4.0 – Offline Python bridge server (Flask)
Runs on localhost:5000. All ML and drone logic lives here.

Endpoints:
  GET  /status                    → server health
  GET  /webcam                    → MJPEG stream (laptop cam)
  POST /capture                   → capture one sample {label}
  GET  /samples                   → current sample counts
  POST /clear_samples             → wipe gesture_data.csv
  GET  /train                     → train all classifiers (SSE progress)
  GET  /predict_stream            → SSE: live predictions from all models
  POST /tello/connect             → connect to Tello
  GET  /tello/stream              → MJPEG stream from Tello cam
  POST /tello/command             → send drone command {cmd}
  POST /tello/disconnect          → land + disconnect
"""

import csv
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import cv2
import joblib
import numpy as np
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
CSV_FILE   = ROOT / "gesture_data.csv"
MODEL_DIR  = ROOT / "models"
VENV_PY    = ROOT / "venv" / "Scripts" / "python.exe"

sys.path.insert(0, str(ROOT))
from gesture_recognition.vector_extractor import HandVectorExtractor

VECTOR_DIM = 63
GESTURE_CLASSES = [
    "TAKEOFF", "LAND", "UP", "DOWN",
    "LEFT", "RIGHT", "FORWARD", "BACKWARD",
    "CLOCKWISE_90", "ANTICLOCKWISE_90",
]

# ── App ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Shared state ──────────────────────────────────────────────────────────
_lock          = threading.Lock()
_webcam_cap    = None
_tello         = None
_tello_cap     = None
_extractor     = None
_models        = {}
_label_encoder = None
_last_frame    = None          # latest BGR frame from webcam
_last_vector   = None          # latest 63-D vector


# ── Startup helpers ────────────────────────────────────────────────────────

def _get_webcam():
    global _webcam_cap
    if _webcam_cap is None or not _webcam_cap.isOpened():
        _webcam_cap = cv2.VideoCapture(0)
    return _webcam_cap


def _get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = HandVectorExtractor()
    return _extractor


def _load_models():
    global _models, _label_encoder
    le_path = MODEL_DIR / "label_encoder.pkl"
    if not le_path.exists():
        return False
    _label_encoder = joblib.load(str(le_path))
    _models = {}
    for name in ["LR", "KNN", "SVM", "RF", "XGB", "ANN"]:
        p = MODEL_DIR / f"gesture_clf_{name}.pkl"
        if p.exists():
            data = joblib.load(str(p))
            _models[name] = data["pipeline"]
    return len(_models) > 0


# ── Frame grabber thread ───────────────────────────────────────────────────

def _frame_loop():
    global _last_frame, _last_vector
    ext = _get_extractor()
    while True:
        cap = _get_webcam()
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.03)
            continue
        frame = cv2.flip(frame, 1)
        vector, annotated = ext.extract(frame)
        with _lock:
            _last_frame  = annotated
            _last_vector = vector
        time.sleep(0.01)


threading.Thread(target=_frame_loop, daemon=True).start()
_load_models()


# ── MJPEG helpers ──────────────────────────────────────────────────────────

def _encode_jpeg(frame) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()


def _mjpeg_generator(frame_source_fn):
    while True:
        frame = frame_source_fn()
        if frame is None:
            time.sleep(0.05)
            continue
        data = _encode_jpeg(frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        )
        time.sleep(0.033)  # ~30 fps cap


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    return jsonify({
        "ok": True,
        "models_loaded": list(_models.keys()),
        "model_ready": len(_models) > 0,
        "classes": GESTURE_CLASSES,
    })


@app.get("/webcam")
def webcam_stream():
    def src():
        with _lock:
            return _last_frame.copy() if _last_frame is not None else None
    return Response(
        stream_with_context(_mjpeg_generator(src)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/samples")
def get_samples():
    counts = {c: 0 for c in GESTURE_CLASSES}
    if CSV_FILE.exists():
        with open(CSV_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lbl = row.get("label", "")
                if lbl in counts:
                    counts[lbl] += 1
    return jsonify(counts)


@app.post("/capture")
def capture():
    data  = request.json or {}
    label = data.get("label", "").strip().upper()
    if label not in GESTURE_CLASSES:
        return jsonify({"error": f"Unknown label '{label}'"}), 400

    with _lock:
        vector = _last_vector

    if vector is None:
        return jsonify({"error": "No hand detected"}), 400

    # Write to CSV
    write_header = not CSV_FILE.exists()
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["label"] + [f"v{i}" for i in range(VECTOR_DIM)])
        writer.writerow([label] + vector.tolist())

    return jsonify({"ok": True, "label": label})


@app.post("/clear_samples")
def clear_samples():
    if CSV_FILE.exists():
        CSV_FILE.unlink()
    return jsonify({"ok": True})


@app.post("/clear_class")
def clear_class():
    data  = request.json or {}
    label = data.get("label", "").strip().upper()
    if label not in GESTURE_CLASSES:
        return jsonify({"error": f"Unknown label '{label}'"}), 400

    if not CSV_FILE.exists():
        return jsonify({"ok": True, "removed": 0})

    rows = []
    removed = 0
    fieldnames = None
    with open(CSV_FILE, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row.get("label", "") == label:
                removed += 1
            else:
                rows.append(row)

    if rows and fieldnames:
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    elif CSV_FILE.exists():
        CSV_FILE.unlink()

    return jsonify({"ok": True, "removed": removed})


@app.get("/train")
def train_models():
    """SSE endpoint — streams training progress line by line."""
    def generate():
        global _models, _label_encoder
        try:
            import pandas as pd
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.neural_network import MLPClassifier
            from sklearn.svm import SVC
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from xgboost import XGBClassifier

            def send(msg):
                return f"data: {json.dumps({'msg': msg})}\n\n"

            if not CSV_FILE.exists():
                yield send("ERROR: gesture_data.csv not found")
                return

            df = pd.read_csv(str(CSV_FILE))
            yield send(f"Loaded {len(df)} samples across {df['label'].nunique()} classes")

            X  = df[[f"v{i}" for i in range(VECTOR_DIM)]].values.astype(np.float32)
            le = LabelEncoder()
            y  = le.fit_transform(df["label"].values)

            MODEL_DIR.mkdir(exist_ok=True)
            joblib.dump(le, str(MODEL_DIR / "label_encoder.pkl"))
            yield send("Label encoder saved")

            clfs = {
                "LR":  LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
                "KNN": KNeighborsClassifier(n_neighbors=7, metric="minkowski", weights="distance"),
                "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True),
                "RF":  RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
                "XGB": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                                     subsample=0.8, colsample_bytree=0.8,
                                     eval_metric="mlogloss", random_state=42, verbosity=0),
                "ANN": MLPClassifier(hidden_layer_sizes=(512, 256, 256, 128, 64, 32),
                                     activation="relu", solver="adam",
                                     learning_rate_init=0.001, max_iter=1000,
                                     early_stopping=True, validation_fraction=0.1,
                                     random_state=42),
            }

            results = []
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for name, clf in clfs.items():
                yield send(f"Training {name}...")
                pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
                scores   = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
                pipeline.fit(X, y)
                out = str(MODEL_DIR / f"gesture_clf_{name}.pkl")
                joblib.dump({"pipeline": pipeline, "label_encoder": le}, out)
                results.append((name, float(scores.mean()), float(scores.std())))
                yield send(f"  {name}: CV={scores.mean():.4f} ± {scores.std():.4f}  ✓ saved")

            results.sort(key=lambda r: -r[1])
            yield send("─── Results ───")
            for name, mean, std in results:
                yield send(f"  {name:<4}  {mean:.4f} ± {std:.4f}")
            best = results[0]
            yield send(f"Best: {best[0]} ({best[1]:.4f})")

            # Reload models into server memory BEFORE sending DONE
            # so the new models are live the instant the client receives DONE.
            yield send("Reloading models into memory…")
            with _lock:
                le_path = MODEL_DIR / "label_encoder.pkl"
                _label_encoder = joblib.load(str(le_path))
                _models = {}
                for mname in ["LR", "KNN", "SVM", "RF", "XGB", "ANN"]:
                    p = MODEL_DIR / f"gesture_clf_{mname}.pkl"
                    if p.exists():
                        _models[mname] = joblib.load(str(p))["pipeline"]
            yield send(f"✓ {len(_models)} models active in memory")
            yield send("DONE")

        except Exception as e:
            yield f"data: {json.dumps({'msg': f'ERROR: {e}'})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.get("/predict_stream")
def predict_stream():
    """SSE — streams predictions for all loaded models ~10fps."""
    def generate():
        while True:
            with _lock:
                vector = _last_vector

            if vector is None or not _models:
                payload = {"hand": False, "predictions": {}}
            else:
                preds = {}
                v = vector.reshape(1, -1)
                for name, pipeline in _models.items():
                    try:
                        proba = pipeline.predict_proba(v)[0]
                        idx   = int(proba.argmax())
                        preds[name] = {
                            "label": _label_encoder.classes_[idx],
                            "confidence": round(float(proba[idx]), 3),
                        }
                    except Exception:
                        pred = pipeline.predict(v)[0]
                        preds[name] = {
                            "label": _label_encoder.classes_[pred],
                            "confidence": 1.0,
                        }
                payload = {"hand": True, "predictions": preds}

            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.1)

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# ── Tello routes ───────────────────────────────────────────────────────────

@app.post("/tello/connect")
def tello_connect():
    global _tello, _tello_cap
    try:
        from djitellopy import Tello
        if _tello is not None:
            return jsonify({"ok": True, "msg": "Already connected"})
        t = Tello()
        t.connect()
        battery = t.get_battery()
        t.streamon()
        _tello     = t
        _tello_cap = t.get_frame_read()
        return jsonify({"ok": True, "battery": battery})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/tello/battery")
def tello_battery():
    """Get current drone battery level (or -1 if not connected)."""
    global _tello
    if _tello is None:
        return jsonify({"ok": False, "battery": -1, "error": "Not connected"}), 400
    try:
        battery = _tello.get_battery()
        return jsonify({"ok": True, "battery": battery})
    except Exception as e:
        return jsonify({"ok": False, "battery": -1, "error": str(e)}), 500


@app.get("/tello/stream")
def tello_stream():
    def src():
        if _tello_cap is None:
            return None
        frame = _tello_cap.frame
        if frame is None:
            return None
        return frame.copy()

    return Response(
        stream_with_context(_mjpeg_generator(src)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/tello/command")
def tello_command():
    global _tello
    if _tello is None:
        return jsonify({"ok": False, "error": "Not connected"}), 400

    cmd = (request.json or {}).get("cmd", "").upper()
    CMD_MAP = {
        "TAKEOFF":          lambda: _tello.takeoff(),
        "LAND":             lambda: _tello.land(),
        "UP":               lambda: _tello.move_up(30),
        "DOWN":             lambda: _tello.move_down(30),
        "LEFT":             lambda: _tello.move_left(30),
        "RIGHT":            lambda: _tello.move_right(30),
        "FORWARD":          lambda: _tello.move_forward(30),
        "BACKWARD":         lambda: _tello.move_back(30),
        "CLOCKWISE_90":     lambda: _tello.rotate_clockwise(90),
        "ANTICLOCKWISE_90": lambda: _tello.rotate_counter_clockwise(90),
    }
    if cmd not in CMD_MAP:
        return jsonify({"ok": False, "error": f"Unknown command: {cmd}"}), 400
    try:
        CMD_MAP[cmd]()
        return jsonify({"ok": True, "cmd": cmd})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/config")
def get_config():
    """Expose non-secret config to renderer (localhost only)."""
    import os
    return jsonify({
        "ollama_host":  os.environ.get("OLLAMA_HOST",  "http://localhost:11434"),
        "ollama_model": os.environ.get("OLLAMA_MODEL", "llava"),
    })


@app.get("/tello/snapshot")
def tello_snapshot():
    """Return a single JPEG frame (drone cam if connected, else webcam) as base64."""
    import base64
    frame = None
    source = "drone"
    if _tello_cap is not None:
        frame = _tello_cap.frame
    if frame is None:
        source = "webcam"
        with _lock:
            frame = _last_frame
    if frame is None:
        return jsonify({"ok": False, "error": "No frame available"}), 400

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jsonify({
        "ok": True,
        "source": source,
        "image_b64": base64.b64encode(buf.tobytes()).decode(),
    })


@app.post("/tello/disconnect")
def tello_disconnect():
    global _tello, _tello_cap
    if _tello:
        try:
            _tello.land()
            _tello.streamoff()
            _tello.end()
        except Exception:
            pass
        _tello     = None
        _tello_cap = None
    return jsonify({"ok": True})


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("GestureFly 4.0 – Bridge Server running on http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
