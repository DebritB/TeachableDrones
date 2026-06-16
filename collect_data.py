"""
collect_data.py
---------------
Collect labeled gesture samples for GestureFly classifier training.

Workflow:
  1. Press a number key  0-7  to select the gesture class to record.
  2. Hold your hand in the desired pose in front of the camera.
  3. Press  SPACE  to capture one sample (63-D vector + label).
  4. Repeat steps 2-3 until you have enough samples for that gesture
     (aim for 100+ per class for a reliable classifier).
  5. Press  S  to save all collected samples and quit.
     Press  Q  to quit without saving.

Output file:  gesture_data.csv   (appended if it already exists)
Columns:  label, v0, v1, …, v62

Gesture class map:
  0 → TAKEOFF          4 → LEFT
  1 → LAND             5 → RIGHT
  2 → UP               6 → CLOCKWISE_90
  3 → DOWN             7 → ANTICLOCKWISE_90

Usage:
    python collect_data.py
    python collect_data.py --camera 1
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

from gesture_recognition.vector_extractor import HandVectorExtractor

# -------------------------------------------------------------------------
GESTURE_MAP: dict[str, str] = {
    "0": "TAKEOFF",
    "1": "LAND",
    "2": "UP",
    "3": "DOWN",
    "4": "LEFT",
    "5": "RIGHT",
    "6": "CLOCKWISE_90",
    "7": "ANTICLOCKWISE_90",
}

VECTOR_DIM = 63
CSV_FILE   = "gesture_data.csv"
WINDOW     = "GestureFly 4.0 – Data Collector"

# -------------------------------------------------------------------------

def _overlay_ui(
    frame: np.ndarray,
    current_key: str | None,
    counts: dict[str, int],
    status: str,
):
    h, w = frame.shape[:2]

    # Dark top bar
    cv2.rectangle(frame, (0, 0), (w, 120), (25, 25, 25), -1)

    # Active gesture
    if current_key:
        label_text = f"Active: [{current_key}]  {GESTURE_MAP[current_key]}"
        label_color = (0, 255, 100)
    else:
        label_text  = "Active: None  –  press 0-7 to select a gesture"
        label_color = (80, 80, 80)

    cv2.putText(frame, label_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2, cv2.LINE_AA)

    # Status line
    cv2.putText(frame, status, (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 1, cv2.LINE_AA)

    # Per-class sample counts
    total   = sum(counts.values())
    summary = "  ".join(
        f"{GESTURE_MAP[k][:3]}:{counts[k]}"
        for k in sorted(GESTURE_MAP)
    )
    cv2.putText(frame, f"Samples  {summary}  |  total:{total}", (10, 83),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    # Key legend
    cv2.putText(
        frame,
        "Keys:  0-7 select class   |   SPACE capture   |   S save+quit   |   Q quit",
        (10, 108),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA,
    )


# -------------------------------------------------------------------------

def _save(records: list[list]) -> None:
    if not records:
        print("No samples were collected.")
        return

    cols = ["label"] + [f"v{i}" for i in range(VECTOR_DIM)]
    new_df = pd.DataFrame(records, columns=cols)

    if os.path.exists(CSV_FILE):
        existing = pd.read_csv(CSV_FILE)
        combined = pd.concat([existing, new_df], ignore_index=True)
        print(f"Appended {len(records)} new samples to  {CSV_FILE}")
    else:
        combined = new_df
        print(f"Created  {CSV_FILE}  with {len(records)} samples")

    combined.to_csv(CSV_FILE, index=False)
    print("\nSamples per class:")
    print(combined["label"].value_counts().to_string())


# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GestureFly – gesture data collector")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera index {args.camera}")

    records: list[list]  = []
    counts:  dict[str, int] = {k: 0 for k in GESTURE_MAP}
    current_key: str | None = None
    status = "Press 0-7 to select a gesture class, then SPACE to capture a sample."

    print("GestureFly 4.0 – Data Collector")
    print("Gesture map:")
    for k, v in GESTURE_MAP.items():
        print(f"  {k} → {v}")
    print()

    with HandVectorExtractor() as extractor:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            vector, annotated = extractor.extract(frame)

            if vector is None and "No hand" not in status:
                status = "No hand detected – show your hand to the camera."

            _overlay_ui(annotated, current_key, counts, status)
            cv2.imshow(WINDOW, annotated)

            key = cv2.waitKey(1) & 0xFF
            ch  = chr(key) if key < 128 else ""

            if ch in ("q", "Q"):
                print("Quit without saving.")
                break

            elif ch in ("s", "S"):
                _save(records)
                break

            elif ch in GESTURE_MAP:
                current_key = ch
                status = (
                    f"Selected [{current_key}] {GESTURE_MAP[current_key]}.  "
                    f"Hold the pose and press SPACE to capture."
                )

            elif key == 32:   # SPACE
                if current_key is None:
                    status = "No gesture selected!  Press 0-7 first."
                elif vector is None:
                    status = "No hand in frame!  Show your hand, then press SPACE."
                else:
                    row = [GESTURE_MAP[current_key]] + vector.tolist()
                    records.append(row)
                    counts[current_key] += 1
                    status = (
                        f"Captured!  [{current_key}] {GESTURE_MAP[current_key]}"
                        f"  →  {counts[current_key]} sample(s) so far."
                    )

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------------------------------------------------

if __name__ == "__main__":
    main()
