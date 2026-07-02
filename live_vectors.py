"""
live_vectors.py
---------------
Real-time MediaPipe hand landmark vector viewer.

Show your hand to the webcam — the 63-D normalized feature vector is
displayed on screen and printed to the console in real time.

This is the first stage of GestureFly 4.0:
  GESTURE  →  63-D vector  →  (classifier)  →  drone command

Drone commands that will be trained:
  TAKEOFF | LAND | UP | DOWN | LEFT | RIGHT | CLOCKWISE_90 | ANTICLOCKWISE_90

Usage:
    python live_vectors.py
    python live_vectors.py --camera 1        # use a different camera index
"""

import argparse
import sys

import cv2

from gesture_recognition.vector_extractor import HandVectorExtractor

# -------------------------------------------------------------------------
WINDOW = "GestureFly 4.0 – Live Vector Monitor"

COMMAND_LEGEND = [
    "--- Drone commands (to train) ---",
    "0: TAKEOFF        1: LAND",
    "2: UP             3: DOWN",
    "4: LEFT           5: RIGHT",
    "6: CLOCKWISE_90   7: ANTICLOCKWISE_90",
    "",
    "Q – quit",
]


# -------------------------------------------------------------------------
def _draw_text_block(
    frame,
    lines: list[str],
    x: int,
    y_start: int,
    scale: float = 0.42,
    color: tuple = (180, 180, 180),
    thickness: int = 1,
):
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line,
            (x, y_start + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, thickness, cv2.LINE_AA,
        )


def _draw_vector_overlay(frame, vector):
    """Print first 9 values (3 landmarks) on the frame bottom."""
    h = frame.shape[0]
    lines = ["63-D vector extracted  (showing first 3 landmarks):"]
    for i in range(3):
        b = i * 3
        lines.append(
            f"  LM{i:02d}  x={vector[b]:+.3f}  y={vector[b+1]:+.3f}  z={vector[b+2]:+.3f}"
        )
    lines.append(f"  ... (+ {63 - 9} more values)")
    _draw_text_block(frame, lines, x=10, y_start=h - 90, color=(0, 255, 90))


# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GestureFly – live vector monitor")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera index {args.camera}")

    print("GestureFly 4.0 – Live Vector Monitor")
    print("Show your hand. Press Q to quit.\n")

    with HandVectorExtractor() as extractor:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARNING] Empty frame — retrying…")
                continue

            frame = cv2.flip(frame, 1)  # mirror so it feels natural
            vector, annotated = extractor.extract(frame)

            h, w = annotated.shape[:2]

            if vector is not None:
                # Console: live preview of first 9 values
                preview = "  ".join(f"{v:+.3f}" for v in vector[:9])
                print(f"\r[v0-v8] {preview} …", end="", flush=True)

                # On-frame overlay
                _draw_vector_overlay(annotated, vector)

            else:
                print("\r[waiting for hand]                              ", end="", flush=True)
                cv2.putText(
                    annotated,
                    "No hand detected – show your hand",
                    (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 60, 255), 2, cv2.LINE_AA,
                )

            # Legend (top-left)
            _draw_text_block(annotated, COMMAND_LEGEND, x=10, y_start=22)

            cv2.imshow(WINDOW, annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
