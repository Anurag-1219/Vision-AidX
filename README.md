# VisionAid X — Spatial AI Voice Assistant for Visually Impaired 👁️⚡

VisionAid X is a multimodal, voice-driven assistive system for visually
impaired users. It combines real-time camera input, speech recognition,
a local Vision-Language Model, and text-to-speech to describe
surroundings, read text aloud, warn about nearby hazards, and recall
where objects were last seen — running on a laptop with no ongoing
cloud costs.

## 🌟 Key Features

- **🎙️ Voice-Driven Interface** — hands-free operation via live speech
  recognition (English + Hindi trigger words) and spoken responses
- **👁️ Local Scene Description (VLM)** — powered by `minicpm-v` running
  locally via [Ollama](https://ollama.com), no cloud vision API needed
- **📦 Object Detection + Spatial Awareness** — YOLOv8 detects objects
  and reports their position (left/right/front) and relative distance
- **⚠️ Safety Alerts** — rule-based risk classifier escalates warnings
  for nearby hazards (vehicles, obstacles, furniture)
- **📖 Text Reading (OCR)** — reads signs, labels, and printed text
  aloud via EasyOCR
- **🧠 Session Memory** — recall where an object was last seen
  ("where is my phone?")

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Scene Description (VLM) | Ollama + minicpm-v |
| OCR (Text Reading) | EasyOCR |
| Speech Recognition | SpeechRecognition + Google STT |
| Text-to-Speech | gTTS + pygame |
| Computer Vision | OpenCV |
| Language | Python 3.11 |

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11 (newer versions may not yet support all dependencies —
  see Known Issues below)
- [Ollama](https://ollama.com) installed

### 2. Set up environment
```bash
python -m venv venv311
venv311\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3. Pull the local vision model
```bash
ollama pull minicpm-v
```

### 4. Run
```bash
python main_v3.py
```

### Voice commands
| Say | Triggers |
|---|---|
| "detect", "object", "kya hai" | Object + spatial + safety scan |
| "read", "text", "padho" | Reads visible text aloud |
| "describe", "what", "samne" | Scene description via VLM |
| "where is my [item]", "kahan hai" | Recall last-seen location |
| "stop", "exit", "quit" | Shuts down |

## 📋 Project Structure

- `main_v3.py` — main integrated application (current working version)
- `test_ocr.py` — standalone OCR test/demo script
- `main.py` — Phase 1 prototype (object detection + speech only)

## ⚠️ Known Issues / Limitations

- **GPU acceleration**: CUDA support is currently unreliable on some
  NVIDIA laptop GPUs (tested: RTX 2050) — driver-related. Runs
  correctly on CPU, just slower for the VLM module.
- **VLM response time**: scene description can take 10–30+ seconds on
  CPU. Avoid firing repeated "describe" requests back-to-back — let
  each finish before the next.
- **Speech recognition accuracy**: occasional mishearing of similar-
  sounding words (e.g. "person" → "present") — a known limitation of
  cloud speech recognition over a laptop mic.

## 🔭 Future Scope

- Full SLAM / 3D spatial mapping
- Persistent (vector database) long-term memory
- Edge deployment (Jetson / Raspberry Pi)
- SOS / emergency contact integration