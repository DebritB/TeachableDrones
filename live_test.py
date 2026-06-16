"""
live_test.py
------------
Live gesture recognition test — all 6 classifiers run simultaneously.

Show your hand to the webcam. Every frame:
  • All 6 models predict your gesture + confidence
  • A consensus vote is shown at the top
  • Press 1-6 to highlight a specific model
  • Press A  to return to "show all" mode
  • Press Q  to quit

Usage:
    python live_test.py
    python live_test.py --camera 1
"""

import argparse
import collections
import os
import sys
import time

import cv2
import joblib
import numpy as np

from gesture_recognition.vector_extractor import HandVectorExtractor

# ─── Constants ───────────────────────────────────────────────────────────────

MODEL_DIR   = "models"
WINDOW      = "GestureFly 4.0 – Live Classifier Test"
SMOOTH_LEN  = 8          # prediction smoothing buffer (frames)
PANEL_H     = 265        # height of bottom info panel (px)

MODEL_KEYS = ["LR", "KNN", "SVM", "RF", "XGB", "ANN"]
KEY_MAP    = {ord(str(i + 1)): name for i, name in enumerate(MODEL_KEYS)}  # 1-6 → name

# Colours (BGR)
C_GREEN   = (0,   220,  80)
C_YELLOW  = (0,   220, 220)
C_RED     = (60,   60, 220)
C_WHITE   = (240, 240, 240)
C_GRAY    = (130, 130, 130)
C_DARK    = (25,   25,  25)
C_PANEL   = (35,   35,  35)
C_HILIGHT = (255, 180,   0)


# ─── Model loading ───────────────────────────────────────────────────────────

def load_models() -> tuple[dict, object]:
    models = {}
    missing = []
    for name in MODEL_KEYS:
        path = os.path.join(MODEL_DIR, f"gesture_clf_{name}.pkl")
        if not os.path.exists(path):
            missing.append(path)
            continue
        data = joblib.load(path)
        models[name] = data["pipeline"]

    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    if not os.path.exists(le_path) or missing:
        sys.exit(f"[ERROR] Missing model files: {missing or [le_path]}. Run train_classifier.py first.")

    le = joblib.load(le_path)
    print(f"Loaded {len(models)} models.  Classes: {list(le.classes_)}\n")
    return models, le


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict_all(
    models: dict,
    le,
    vector: np.ndarray,
) -> dict[str, tuple[str, float]]:
    """Returns {model_name: (gesture_label, confidence)} for all models."""
    results = {}
    v = vector.reshape(1, -1)
    for name, pipeline in models.items():
        try:
            proba = pipeline.predict_proba(v)[0]
            idx   = int(proba.argmax())
            results[name] = (le.classes_[idx], float(proba[idx]))
        except Exception:
            pred  = pipeline.predict(v)[0]
            results[name] = (le.classes_[pred], 1.0)
    return results


def consensus(preds: dict[str, tuple[str, float]]) -> tuple[str, int]:
    """Majority-vote label + how many models agree."""
    votes = collections.Counter(v[0] for v in preds.values())
    label, count = votes.most_common(1)[0]
    return label, count


# ─── Drawing helpers ─────────────────────────────────────────────────────────

def _bar(frame, x, y, w, h, fraction: float, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * fraction), y + h), color, -1)


def draw_panel(
    canvas: np.ndarray,
    preds:  dict[str, tuple[str, float]] | None,
    smoothed: dict[str, str],
    consensus_label: str,
    consensus_count: int,
    selected: str | None,
    fps: float,
    hand_visible: bool,
):
    h, w = canvas.shape[:2]
    panel_y = h - PANEL_H

    # Panel background
    cv2.rectangle(canvas, (0, panel_y), (w, h), C_PANEL, -1)
    cv2.line(canvas, (0, panel_y), (w, panel_y), C_GRAY, 1)

    if not hand_visible or preds is None:
        cv2.putText(canvas, "No hand detected – show your hand",
                    (20, panel_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_RED, 2, cv2.LINE_AA)
        _draw_legend(canvas, panel_y, w, selected, fps)
        return

    # ── Consensus banner ──
    agree_color = C_GREEN if consensus_count >= 5 else C_YELLOW if consensus_count >= 3 else C_RED
    cv2.putText(canvas,
                f"CONSENSUS: {consensus_label}   ({consensus_count}/6 agree)",
                (20, panel_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, agree_color, 2, cv2.LINE_AA)

    # ── Per-model rows ──
    row_h  = 30
    bar_w  = 120
    name_x = 20
    pred_x = 80
    bar_x  = 290
    conf_x = 420
    y0     = panel_y + 55

    for i, name in enumerate(MODEL_KEYS):
        y = y0 + i * row_h
        if y + row_h > h - 10:
            break

        label, conf = preds.get(name, ("?", 0.0))
        smooth_label = smoothed.get(name, label)

        is_selected  = (selected == name)
        is_consensus = (smooth_label == consensus_label)
        row_color    = C_HILIGHT if is_selected else (C_GREEN if is_consensus else C_RED)

        # Highlight selected row background
        if is_selected:
            cv2.rectangle(canvas, (name_x - 4, y - 18), (w - 10, y + 8), (55, 45, 20), -1)

        # [N] Name
        cv2.putText(canvas, f"[{i+1}] {name:<4}", (name_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_HILIGHT if is_selected else C_GRAY, 1, cv2.LINE_AA)

        # Arrow + smoothed label
        cv2.putText(canvas, f"→  {smooth_label}", (pred_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, row_color, 1 + int(is_selected), cv2.LINE_AA)

        # Confidence bar
        _bar(canvas, bar_x, y - 14, bar_w, 16, conf, row_color)

        # Confidence %
        cv2.putText(canvas, f"{conf * 100:5.1f}%", (conf_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1, cv2.LINE_AA)

    _draw_legend(canvas, panel_y, w, selected, fps)


def _draw_legend(canvas, panel_y, w, selected, fps):
    h = canvas.shape[0]
    sel_text = f"Highlighted: [{MODEL_KEYS.index(selected)+1}] {selected}" if selected else "All models shown"
    cv2.putText(canvas,
                f"Keys: 1-6 highlight model | A all models | Q quit      "
                f"FPS:{fps:4.0f}   {sel_text}",
                (20, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1, cv2.LINE_AA)


def draw_top_banner(frame, consensus_label: str, consensus_count: int, selected: str | None):
    """Large prediction shown on the video frame itself."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)
    color = C_GREEN if consensus_count >= 5 else C_YELLOW
    text  = f"{consensus_label}" if selected is None else f"[{selected}]  {consensus_label}"
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    models, le = load_models()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {args.camera}")

    # Smoothing buffers: deque of last SMOOTH_LEN labels per model
    buffers: dict[str, collections.deque] = {
        name: collections.deque(maxlen=SMOOTH_LEN) for name in MODEL_KEYS
    }

    selected: str | None = None   # None = show all highlighted equally
    fps_timer = time.time()
    fps = 0.0
    frame_count = 0

    print("GestureFly 4.0 – Live Classifier Test")
    print("Show your hand. Press 1-6 to highlight a model, A for all, Q to quit.\n")

    with HandVectorExtractor() as extractor:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            vector, annotated = extractor.extract(frame)

            # FPS
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

            preds: dict | None    = None
            smoothed: dict[str, str] = {}
            con_label = "—"
            con_count = 0

            if vector is not None:
                # Run all classifiers
                if selected is None:
                    preds = predict_all(models, le, vector)
                else:
                    # Still compute all, but only selected is highlighted
                    preds = predict_all(models, le, vector)

                # Update smoothing buffers
                for name, (label, _) in preds.items():
                    buffers[name].append(label)

                # Smoothed label = majority in buffer
                for name in MODEL_KEYS:
                    if buffers[name]:
                        smoothed[name] = collections.Counter(buffers[name]).most_common(1)[0][0]
                    else:
                        smoothed[name] = "—"

                # If a model is selected, use its smoothed label for the banner
                if selected and selected in smoothed:
                    con_label = smoothed[selected]
                    con_count = sum(1 for v in smoothed.values() if v == con_label)
                else:
                    con_label, con_count = consensus(
                        {k: (v, preds[k][1]) for k, v in smoothed.items()}
                    )

                draw_top_banner(annotated, con_label, con_count, selected)

            # Build canvas: video frame on top, panel below
            cam_h, cam_w = annotated.shape[:2]
            canvas = np.zeros((cam_h + PANEL_H, cam_w, 3), dtype=np.uint8)
            canvas[:cam_h] = annotated

            draw_panel(
                canvas, preds, smoothed,
                con_label, con_count,
                selected, fps,
                hand_visible=(vector is not None),
            )

            cv2.imshow(WINDOW, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("a") or key == ord("A"):
                selected = None
            elif key in KEY_MAP:
                selected = KEY_MAP[key]

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
