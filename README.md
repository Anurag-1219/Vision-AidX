# VisionAid X — Spatial AI Voice Assistant for Visually Impaired

VisionAid X is a multimodal assistive system for visually impaired users.
It uses a camera feed, a local YOLOv8 object detector, a local
Vision-Language Model (via Ollama), and text-to-speech to describe
surroundings, read text aloud, warn about nearby hazards, and recall
where objects were last seen.

The project has two interfaces built on the same core AI logic:
1. Web Interface (current, primary) - a React web app + FastAPI
   backend, controlled via on-screen buttons
2. Voice Interface (earlier prototype) - a terminal app controlled
   entirely by spoken commands (English + Hindi)

## Key Features

- Object Detection + Spatial Awareness - YOLOv8 detects objects
  and reports their position (left/right/front) and relative distance
- Safety Alerts - rule-based risk classifier escalates warnings
  for nearby hazards (vehicles, obstacles, furniture)
- Text Reading (OCR) - reads signs, labels, and printed text
  aloud via EasyOCR
- Scene Description (VLM) - natural-language description of
  the surroundings via a local vision-language model (Ollama)
- Session Memory - recall where an object was last seen

## Tech Stack

| Component | Technology |
|---|---|
| Backend Framework | FastAPI + Uvicorn |
| Frontend | React + Vite + TypeScript |
| Object Detection | YOLOv8 (Ultralytics) |
| Scene Description (VLM) | Ollama + moondream |
| OCR (Text Reading) | EasyOCR |
| Text-to-Speech (Web) | Browser Web Speech API |
| Computer Vision | OpenCV |
| Language | Python 3.11 |

## Getting Started - Web Interface (current version)

### 1. Prerequisites
- Python 3.11
- Node.js + npm
- Ollama installed, with a vision model pulled:
  ollama pull moondream

### 2. Set up Python environment
python -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
pip install fastapi uvicorn ollama

### 3. Start the backend (Terminal 1)
uvicorn main_v3:app --host 127.0.0.1 --port 8080 --reload
Wait for "Application startup complete." - keep this terminal running.

### 4. Start the frontend (Terminal 2 - new window)
npm install
npm run dev
Open the printed URL (e.g. http://localhost:5173) in your browser.

### 5. Use it
Click any of the four buttons on the page:
- Detect Objects & Safety - scans surroundings, reports hazards
- Read Text / OCR - reads any visible text aloud
- Describe Scene - full natural-language scene description
- Find My Item - recalls last-seen location of an object

## How it works (architecture)

Browser (React, port 5173)
   sends a GET request when a button is clicked
FastAPI backend (main_v3.py, port 8080)
   opens the webcam directly via OpenCV
   runs YOLOv8 / EasyOCR / Ollama depending on the button
   sends back JSON { "text": "..." }
Browser speaks the result aloud (Web Speech API)

Note: the backend, not the browser, controls the camera directly.
This avoids a Windows limitation where two programs cannot both
hold the webcam open at the same time.

## Known Issues / Limitations

- GPU acceleration is currently unreliable on some NVIDIA laptop GPUs
  (tested: RTX 2050) - driver-related. Runs correctly on CPU, just
  slower for the VLM (Describe Scene) module.
- VLM response time can take 10-30+ seconds on CPU. Avoid clicking
  "Describe Scene" repeatedly back-to-back.
- Port 8000 was unusable on the development machine (WinError 10013 -
  reserved by the OS); the project runs on port 8080 instead.
- Web Speech API voice quality varies by browser/OS.

## Project Structure

- main_v3.py - FastAPI backend for the web interface (current)
- src/App.tsx - React frontend
- main.py - Phase 1 prototype (object detection + speech, standalone)
- test_ocr.py - standalone OCR test/demo script

## Future Scope

- Full SLAM / 3D spatial mapping
- Persistent (vector database) long-term memory
- Edge deployment (Jetson / Raspberry Pi)
- SOS / emergency contact integration
- Restore/merge the voice-command interface alongside the web UI
