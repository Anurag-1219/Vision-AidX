# VisionAid X — Spatial AI Voice Assistant for Visually Impaired 👁️⚡

**VisionAid X** is an end-to-end multimodal, privacy-focused assistive system designed to empower visually impaired individuals. It combines real-time camera feeds, speech recognition, local Vision-Language Models (VLM), and Text-to-Speech (TTS) synthesis to describe surroundings, identify hand-held objects, and assist with spatial awareness—all completely offline.

---

## 🌟 Key Features

* 🎙️ **Voice-Driven Interface:** Hands-free operation using live speech recognition and natural text-to-speech audio feedback.
* 👁️ **Local VLM Reasoning:** Powered by high-accuracy Vision-Language Models (`minicpm-v` / `llava`) running locally via Ollama (100% privacy-focused).
* 📦 **Object & Hand Detection:** Context-aware spatial analysis tailored for objects held in hand or placed in front of the camera.
* ⚡ **Offline First:** Operates locally on CPU/GPU hardware without relying on paid external cloud APIs.

---

## 🛠️ Tech Stack & Architecture

| Component | Technology Used |
| :--- | :--- |
| **Vision Model (VLM)** | Ollama (`minicpm-v` / `llava`) |
| **Computer Vision** | OpenCV (`cv2`), NumPy |
| **Speech Recognition** | PyAudio, SpeechRecognition |
| **Text-to-Speech (TTS)** | PyTTSx3 (Offline Voice Engine) |
| **Language & Tools** | Python 3.11, Base64 Encoding |

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.11 installed along with [Ollama](https://ollama.com/).

### 2. Pull the Local VLM
Start your local Ollama server and pull the vision model:
```bash
ollama pull minicpm-v