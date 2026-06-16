# ASL Sequence Translator v2.0 — Project Context

## What this project is
A real-time American Sign Language translator that captures 30-frame hand pose sequences using MediaPipe, classifies them with a Conv1D-BiLSTM model, and streams predictions via a FastAPI WebSocket to a React frontend — deployed live on HuggingFace Spaces.

## My background
- CS student at University of Toronto Scarborough
- Know Python, C, Git, React, Flask
- This is a portfolio project targeting software engineering and ML internships

---

## Tech Stack
- Python 3.11
- MediaPipe (hand landmark extraction)
- OpenCV (webcam capture)
- TensorFlow/Keras (Conv1D-BiLSTM model)
- scikit-learn (preprocessing, baselines, evaluation)
- FastAPI + Uvicorn (backend inference server)
- WebSockets (real-time frame streaming)
- React (frontend webcam UI)
- Weights & Biases (experiment tracking)
- Gradio (HuggingFace Spaces demo)
- Docker + Docker Compose (containerized deployment)
- HuggingFace Spaces (public live demo)

---

## File Structure
```
asl-translator/
├── src/
│   ├── landmark_extractor.py   # MediaPipe init + 21-landmark extraction
│   ├── data_collector.py       # Webcam loop, 30-frame sequence capture
│   ├── augmentation.py         # Jitter, scale, time_warp augmentations
│   ├── preprocessor.py         # Normalize, scale, 70/15/15 split
│   ├── model.py                # Conv1D-BiLSTM Keras architecture
│   ├── train.py                # Training loop with W&B logging
│   ├── evaluate.py             # Baselines + confusion matrix + OOD test
│   ├── inference.py            # Local real-time prediction loop
│
├── backend/
│   ├── main.py                 # FastAPI app: /predict + /ws endpoints
│   ├── test_ws_client.py       # Python asyncio WebSocket test client
│
├── frontend/                   # React app (create-react-app or Vite)
│   └── src/App.jsx             # Webcam + MediaPipe WASM + WebSocket UI
│
├── demo/
│   └── app.py                  # Gradio app for HuggingFace Spaces
│
├── data/
│   ├── raw/                    # .npy files per gesture (one folder per class)
│   ├── processed/              # X_train, y_train, X_val, y_val, X_test, y_test
│
├── models/
│   ├── gesture_classifier.keras
│   ├── gesture_classifier.tflite
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│
├── scripts/
│   └── dataset_summary.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Model Architecture
```
Input(shape=(30, 63))
→ Conv1D(64, kernel_size=3, activation='relu')
→ BatchNormalization()
→ Bidirectional(LSTM(128, return_sequences=False))
→ Dropout(0.3)
→ Dense(64, activation='relu')
→ Dropout(0.3)
→ Dense(num_classes, activation='softmax')
```
- Loss: categorical crossentropy
- Optimizer: Adam (lr=0.001)
- Callbacks: ModelCheckpoint (best val_accuracy), EarlyStopping (patience=15), WandbCallback
- Epochs: up to 60, batch size 32

---

## Dataset
- 20 classes: 10 ASL letters (A, B, C, D, E, F, G, H, I, L) + 10 words (HELLO, THANKS, YES, NO, PLEASE, SORRY, HELP, MORE, GOOD, BAD)
- 100 sequences per class = ~2,000 total sequences
- Each sequence: (30, 63) numpy array — 30 frames × 21 landmarks × 3 coordinates (x,y,z)
- Wrist normalization: subtract landmark[0] from all 21 landmarks per frame (translational invariance)
- Additional ~10 sequences per class collected with other hand / different angle → stored in data/raw/varied/ as out-of-distribution test set

---

## Preprocessing
- Stratified 70/15/15 train/val/test split
- StandardScaler fitted ONLY on training set, applied to val and test (no leakage)
- Scaler and LabelEncoder saved as .pkl files

---

## Evaluation Plan
- 4-model comparison on held-out test set: BiLSTM vs KNN vs SVM vs LogisticRegression
- Metrics: test accuracy + macro F1 for all 4 models
- Confusion matrix heatmap (seaborn) for BiLSTM
- Per-class precision/recall/F1 (sklearn classification_report)
- Out-of-distribution accuracy on data/raw/varied/ — honest generalization estimate

---

## Data Augmentation (applied only to training)
- add_jitter(seq, std=0.01): Gaussian noise on all landmark coordinates
- scale_sequence(seq, factor_range=(0.9, 1.1)): random hand size scaling
- time_warp(seq, warp_rate=0.1): temporal stretch/compress via np.interp
- Implemented as tf.data.Dataset generator

---

## Backend (FastAPI)
- POST /predict: accepts SequenceRequest {landmarks: List[List[float]]} (30×63), returns {label, confidence, all_probs}
- WebSocket /ws: rolling deque(maxlen=30) accumulates per-frame landmarks, fires inference when full, returns {label, confidence}
- CORS middleware enabled (allow origins=['*'])
- Model, scaler, encoder loaded on server startup

---

## Frontend (React)
- MediaPipe Hands running in-browser via WASM (@mediapipe/hands)
- Landmarks drawn on canvas overlay over webcam feed
- WebSocket connection to ws://localhost:8000/ws
- Per-frame: extract 63 floats → send JSON → receive {label, confidence} → display
- Sentence buffer: same prediction held 20 frames at confidence > 0.85 → append to word
- Web Speech API (window.speechSynthesis) for text-to-speech output
- Clear button to reset buffer, prediction history panel (last 10 predictions)

---

## Deployment
- Gradio demo: demo/app.py → HuggingFace Spaces (Gradio SDK, free tier)
- Full stack: docker-compose.yml (FastAPI backend + nginx-served React frontend)
- TFLite export: models/gesture_classifier.tflite for mobile deployment angle

---

## 14-Day Execution Plan

### Day 1 — Foundation: live hand landmarks on screen (2h)
Focus: Get MediaPipe running and visualize 21 hand landmarks on a live webcam feed
Tasks:
1. Create venv, install mediapipe, opencv-python, numpy, tensorflow, scikit-learn, matplotlib, pin in requirements.txt
2. Write src/landmark_extractor.py: init MediaPipe Hands, capture with cv2.VideoCapture, extract 21 (x,y,z) landmarks
3. Draw landmarks with mp.solutions.drawing_utils, display with cv2.imshow, quit on 'q'
4. Print landmark array shape (21, 3) to terminal
Deliverable: Live webcam window with 21 green dots tracking your hand

### Day 2 — Temporal collector: capture 30-frame sequences (3h)
Focus: Build data_collector.py that captures (30, 63) numpy arrays on keypress
Tasks:
1. On keypress 's': record 30 consecutive frames, extract 63 landmarks per frame, stack to (30, 63)
2. Wrist normalization per frame: subtract landmark[0] from all 21 landmarks
3. Save to data/raw/{class_label}/{timestamp}.npy with os.makedirs
4. Add 'Recording: frame X/30' overlay text in OpenCV
5. Test: record 3 sequences, load with np.load(), confirm shape (30, 63)
Deliverable: data/raw/test/ with 3 confirmed .npy files of shape (30, 63)

### Day 3 — Data sprint: collect your dataset (4h)
Focus: Record 100 sequences × 20 classes = ~2,000 total sequences
Tasks:
1. Collect 10 ASL letters: A B C D E F G H I L
2. Collect 10 ASL words: HELLO THANKS YES NO PLEASE SORRY HELP MORE GOOD BAD
3. Write scripts/dataset_summary.py to confirm counts and shapes
4. Record 10 varied sequences per class (other hand/angle) → data/raw/varied/
Deliverable: data/raw/ with 20 class folders + data/raw/varied/ OOD test set

### Day 4 — Preprocessing pipeline (3h)
Focus: Build preprocessor.py with stratified 70/15/15 split, no data leakage
Tasks:
1. Load all .npy files → X (N, 30, 63) and y vector
2. LabelEncoder → stratified 70/15/15 split (train_test_split twice with stratify=y)
3. StandardScaler: fit on train only, transform val + test → save scaler.pkl, label_encoder.pkl
4. Save 6 arrays to data/processed/
5. Print class distribution table across all splits
Deliverable: data/processed/ with 6 .npy arrays, models/ with 2 .pkl files

### Day 5 — BiLSTM model + W&B (3h)
Focus: Build Conv1D-BiLSTM, wire up W&B, run first training job
Tasks:
1. pip install wandb, run wandb login
2. Write src/model.py with architecture above
3. Write src/train.py with wandb.init(), WandbCallback(), ModelCheckpoint, EarlyStopping
4. Run training, watch W&B dashboard, save best model to models/gesture_classifier.keras
Deliverable: Trained model + shareable W&B dashboard URL with training curves

### Day 6 — Evaluation: baselines + confusion matrix + F1 report (3h)
Focus: Generate full evaluation report, prove BiLSTM beats baselines
Tasks:
1. Train KNN, SVM, LogReg on flattened (N, 1890) features, test on X_test
2. Run BiLSTM on X_test (3D shape), print 4-model results table
3. Generate seaborn confusion matrix → outputs/confusion_matrix.png
4. sklearn classification_report → outputs/classification_report.txt
5. Run on data/raw/varied/ → report OOD accuracy
Deliverable: Results table, confusion_matrix.png, classification_report.txt, OOD number

### Day 7 — Data augmentation (3h)
Focus: Add 3 augmentations to training via tf.data.Dataset generator
Tasks:
1. Write src/augmentation.py: add_jitter, scale_sequence, time_warp
2. Integrate into train.py as tf.data.Dataset generator (train only)
3. Retrain → log as second W&B run
4. Compare augmented vs non-augmented on in-distribution and OOD test sets
Deliverable: Second W&B run, 2-row comparison table in augmentation.py comments

### Day 8 — FastAPI backend: REST endpoint (3h)
Focus: Build /predict endpoint that accepts (30×63) sequence → returns prediction
Tasks:
1. pip install fastapi uvicorn python-multipart
2. Write backend/main.py: load model/scaler/encoder on startup
3. Pydantic SequenceRequest with landmarks: List[List[float]]
4. POST /predict: transform → reshape → predict → return {label, confidence, all_probs}
5. Test with curl, verify at localhost:8000/docs
Deliverable: /predict endpoint working, confirmed with curl response

### Day 9 — WebSocket: real-time streaming (3h)
Focus: Add /ws endpoint with rolling 30-frame buffer
Tasks:
1. Add WebSocket /ws: deque(maxlen=30) accumulates frames, fires on full buffer
2. Write backend/test_ws_client.py: asyncio client sends dummy frames, prints predictions
3. Add CORSMiddleware (allow_origins=['*'])
4. Test Python client confirms rolling predictions
Deliverable: test_ws_client.py showing real-time predictions in terminal

### Day 10 — React frontend: webcam → WebSocket → predictions (4h)
Focus: Build React UI with MediaPipe WASM + WebSocket + live prediction display
Tasks:
1. npx create-react-app frontend, install @mediapipe/hands @mediapipe/camera_utils
2. Init MediaPipe Hands in useEffect, draw on canvas overlay
3. Open WebSocket to ws://localhost:8000/ws in useEffect
4. On each MediaPipe callback: send 63 floats → receive {label, confidence} → display
5. Add confidence bar (div width = confidence * 100 + '%')
Deliverable: React app at localhost:3000 with live landmark drawing and real-time predictions

### Day 11 — Sentence buffer + TTS (3h)
Focus: Spell words from letters, speak them with Web Speech API
Tasks:
1. sentenceBuffer state: same prediction for 20 frames at >0.85 confidence → append letter
2. Display accumulating word in large text, Clear button resets buffer
3. Speak button: window.speechSynthesis.speak(new SpeechSynthesisUtterance(buffer.join('')))
4. Add prediction history panel (last 10 predictions with timestamps)
Deliverable: Full demo — sign H+I → screen shows HI → Speak button → browser says HI

### Day 12 — Gradio demo + TFLite export (3h)
Focus: Build Gradio app for HuggingFace, export compressed TFLite model
Tasks:
1. pip install gradio, write demo/app.py with webcam input → landmark extraction → prediction
2. Test locally at localhost:7860
3. TFLite export: TFLiteConverter.from_keras_model(model) → save .tflite
4. Test TFLite with tf.lite.Interpreter, confirm predictions match Keras model
Deliverable: Gradio app working locally, gesture_classifier.tflite confirmed working

### Day 13 — Polish: Docker + README + demo video (4h)
Focus: Make the repo look professional, one-command setup, stellar README
Tasks:
1. Dockerfile for FastAPI backend (python:3.11-slim base)
2. docker-compose.yml: backend (FastAPI) + frontend (nginx serving React build)
3. Record 60–90 second demo video showing full loop + W&B dashboard + confusion matrix
4. Write README.md: description, architecture diagram, results table, confusion matrix image, demo GIF, docker compose up instructions
Deliverable: docker compose up starts everything, demo video recorded, README.md complete

### Day 14 — Ship it: HuggingFace Spaces + GitHub (3h)
Focus: Deploy Gradio demo publicly, push clean repo to GitHub
Tasks:
1. Create HuggingFace Space (Gradio SDK), pip install huggingface_hub
2. Push demo/app.py + model files with huggingface-cli upload
3. Test live URL on HuggingFace servers
4. Push to public GitHub repo with .gitignore, HuggingFace badge in README
5. Update portfolio site and LinkedIn with both links
Deliverable: Public GitHub repo + live HuggingFace Spaces URL anyone can visit

---

## Resume Bullets (to add after completion)
1. Built a real-time ASL gesture recognition system using MediaPipe and a Conv1D-BiLSTM model trained on 2,000 temporal hand-pose sequences, achieving XX% test accuracy across 20 sign classes with baseline comparison against KNN, SVM, and logistic regression.
2. Engineered an end-to-end ML pipeline (data collection → augmentation → training → evaluation) with W&B experiment tracking, out-of-distribution generalization testing, and TFLite export for mobile deployment.
3. Deployed a real-time ASL translator as a full-stack application — FastAPI WebSocket backend, React frontend with Web Speech API TTS, and a live Gradio demo on HuggingFace Spaces accessible at [URL].

---

## Current Status
[ ] Day 1 complete
[ ] Day 2 complete
[ ] Day 3 complete
[ ] Day 4 complete
[ ] Day 5 complete
[ ] Day 6 complete
[ ] Day 7 complete
[ ] Day 8 complete
[ ] Day 9 complete
[ ] Day 10 complete
[ ] Day 11 complete
[ ] Day 12 complete
[ ] Day 13 complete
[ ] Day 14 complete
