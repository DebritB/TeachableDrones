"""
vector_extractor.py
-------------------
Extracts a normalized 63-D hand landmark vector from a webcam frame
using MediaPipe Tasks API (HandLandmarker).

Vector layout  (63 values = 21 landmarks × 3 coords):
    [lm0_x, lm0_y, lm0_z,  lm1_x, lm1_y, lm1_z,  ...,  lm20_x, lm20_y, lm20_z]

Normalization (makes it position- and scale-invariant):
    1. Shift origin to wrist (landmark 0).
    2. Divide by the L2 distance from wrist → middle-finger MCP (landmark 9).

MediaPipe landmark index reference:
    0  Wrist
    1-4   Thumb  (CMC → TIP)
    5-8   Index  (MCP → TIP)
    9-12  Middle (MCP → TIP)
    13-16 Ring   (MCP → TIP)
    17-20 Pinky  (MCP → TIP)

Model file: models/hand_landmarker.task  (downloaded automatically if missing)
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (
    HandLandmarksConnections,
    drawing_styles,
    drawing_utils,
)

# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).parent.parent / "models"
_MODEL_PATH = _MODEL_DIR / "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS


def _ensure_model() -> None:
    """Download the model file if it is not already present."""
    if _MODEL_PATH.exists():
        return
    import urllib.request
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[HandVectorExtractor] Downloading model → {_MODEL_PATH}")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("[HandVectorExtractor] Download complete.")


# ---------------------------------------------------------------------------
class HandVectorExtractor:
    """
    Wraps MediaPipe HandLandmarker (Tasks API) and returns normalized 63-D vectors.

    Usage (context manager):
        with HandVectorExtractor() as extractor:
            vector, annotated_frame = extractor.extract(bgr_frame)

    Usage (manual):
        extractor = HandVectorExtractor()
        vector, annotated_frame = extractor.extract(bgr_frame)
        extractor.close()
    """

    WRIST      = 0   # landmark used as origin
    SCALE_REF  = 9   # landmark whose distance from wrist sets the scale
    VECTOR_DIM = 63  # 21 landmarks × 3 (x, y, z)

    def __init__(
        self,
        max_hands: int = 1,
        detection_conf: float = 0.7,
        tracking_conf: float = 0.7,
    ):
        _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    def extract(
        self, bgr_frame: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """
        Parameters
        ----------
        bgr_frame : np.ndarray
            BGR image from OpenCV (e.g. cap.read()).

        Returns
        -------
        vector : np.ndarray of shape (63,)  or  None
            Normalized landmark vector.
            None when no hand is visible.
        annotated : np.ndarray
            Copy of the frame with hand landmarks drawn.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        annotated = bgr_frame.copy()

        if not result.hand_landmarks:
            return None, annotated

        hand_lm = result.hand_landmarks[0]  # list of NormalizedLandmark

        # Draw landmarks using Tasks drawing utilities
        drawing_utils.draw_landmarks(
            annotated,
            hand_lm,
            _CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )

        # Raw coordinates  →  (21, 3)
        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_lm],
            dtype=np.float32,
        )

        # Shift origin to wrist
        pts -= pts[self.WRIST]

        # Scale normalisation
        scale = float(np.linalg.norm(pts[self.SCALE_REF]))
        if scale > 1e-6:
            pts /= scale

        return pts.flatten(), annotated  # (63,)

    # ------------------------------------------------------------------
    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
