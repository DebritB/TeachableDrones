# ✈ GestureFly 4.0

Control a **DJI Tello drone** with hand gestures — no controller required. GestureFly lets you record your own custom gestures, train a local machine learning model, and fly in real time. An optional AI scene analysis feature uses a locally-running **Ollama** vision model to describe what the drone sees.

---

## Features

- **Teachable gestures** — Record your own hand poses for any of 8 drone commands
- **6 ML classifiers** — Trains LR, KNN, SVM, Random Forest, XGBoost, and ANN in parallel; live prediction uses a majority vote
- **Fully offline** — All inference runs locally; no cloud dependency
- **AI scene analysis** — Uses a local Ollama vision model (e.g. `llava`) to describe the drone's view
- **Electron desktop UI** — Clean 3-step workflow: Collect → Train → Deploy
- **MJPEG streams** — Live webcam and drone camera feeds in the UI

---

## Gesture Classes

| Key | Gesture | Drone Command |
|-----|---------|---------------|
| `0` | TAKEOFF | Take off |
| `1` | LAND | Land |
| `2` | UP | Ascend |
| `3` | DOWN | Descend |
| `4` | LEFT | Strafe left |
| `5` | RIGHT | Strafe right |
| `6` | CLOCKWISE_90 | Rotate right 90° |
| `7` | ANTICLOCKWISE_90 | Rotate left 90° |

---

## Architecture

```
GestureFly 4.0
├── server.py                  # Flask bridge (localhost:5000) — ML + drone logic
├── collect_data.py            # CLI tool for recording gesture samples
├── train_classifier.py        # Trains all 6 classifiers offline
├── gesture_recognition/
│   └── vector_extractor.py   # MediaPipe hand landmark → 63-D normalized vector
├── models/
│   ├── hand_landmarker.task   # MediaPipe model (auto-downloaded)
│   └── gesture_clf_*.pkl      # Trained classifiers (generated after training)
└── app/                       # Electron desktop UI
    ├── electron/main.js
    ├── electron/preload.js
    └── renderer/              # HTML/CSS/JS frontend
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| Ollama | Latest |
| DJI Tello drone | Optional (webcam works without it) |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/gesturefly4.0.git
cd gesturefly4.0
```

### 2. Python environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Node / Electron

```bash
cd app
npm install
cd ..
```

### 4. Ollama (AI scene analysis)

Install Ollama from [ollama.com](https://ollama.com), then pull a vision model:

```bash
ollama pull llava
```

### 5. Environment configuration

Copy the example env file and edit as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `llava` | Vision model to use for scene analysis |

---

## Running

### Start the Python server

```bash
# With venv active
python server.py
```

Server starts at `http://localhost:5000`.

### Start the Electron UI

```bash
cd app
npm start
```

---

## Workflow

### ① Collect gesture samples

Open the **Collect** tab in the UI, or run the CLI tool:

```bash
python collect_data.py
```

- Press `0`–`7` to select a gesture class
- Hold your hand in the desired pose
- Press `SPACE` to capture a sample
- Aim for **100+ samples per gesture** for reliable accuracy
- Press `S` to save and quit

Samples are saved to `gesture_data.csv`.

### ② Train

Click **Train All Models** in the **Train** tab (or run `python train_classifier.py`).

Training runs 5-fold cross-validation on all 6 classifiers and saves them to `models/`.

### ③ Deploy

1. Connect your Tello to the drone's Wi-Fi network
2. Open the **Deploy** tab and click **Connect Drone**
3. Live gesture predictions appear in real time — the majority-vote result is sent as a drone command
4. Click **Analyze Scene (AI)** to get an Ollama-powered description of what the drone sees

---

## Project Dependencies

**Python**

| Package | Purpose |
|---------|---------|
| `mediapipe` | Hand landmark detection |
| `opencv-python` | Webcam / video frame capture |
| `scikit-learn` | ML classifiers and preprocessing |
| `xgboost` | XGBoost classifier |
| `flask` + `flask-cors` | Local REST/SSE bridge server |
| `python-dotenv` | `.env` configuration loading |
| `numpy` / `pandas` | Numerical and data utilities |

**JavaScript**

| Package | Purpose |
|---------|---------|
| `electron` | Desktop app shell |

---

## License

MIT
