import base64
from collections import Counter
import datetime
import os
import re
import time
import cv2
import easyocr
from gtts import gTTS
import pygame
import requests
import speech_recognition as sr
from ultralytics import YOLO


class SpeechEngine:

    def __init__(self):
        pygame.mixer.init()

    def speak(self, text: str, urgent: bool = False):
        if not text or not text.strip():
            return

        prefix = "ALERT! " if urgent else ""
        print(f"Speaking: {prefix}{text}")
        filename = "temp_speech.mp3"

        try:
            tts = gTTS(text=text, lang="en", slow=False)
            tts.save(filename)

            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            pygame.mixer.music.unload()

            if os.path.exists(filename):
                os.remove(filename)

        except Exception as e:
            print(f"Audio Playback Error: {e}")

    def stop(self):
        try:
            pygame.mixer.quit()
        except Exception:
            pass


def capture_frame(filename="captured_scene.jpg"):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not access camera.")
        return False

    time.sleep(0.5)
    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        cv2.imwrite(filename, frame)
        print(f"Captured image: {filename}")
        return True
    else:
        print("Error: Failed to capture image.")
        return False


class VisionMemory:

    def __init__(self):
        self.memory = {}

    def update_memory(self, detections_list):
        for item in detections_list:
            parts = item.split(" ", 1)
            if len(parts) == 2:
                obj_name = parts[0].lower()
                spatial_desc = parts[1]
                self.memory[obj_name] = spatial_desc

    def recall(self, query_object):
        query_object = query_object.lower().strip()
        for obj, desc in self.memory.items():
            if query_object in obj:
                return f"Your {obj} was last seen {desc}."
        return f"I haven't seen any {query_object} recently."


def get_spatial_info(box, img_width, img_height):
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = (x1 + x2) / 2

    if center_x < img_width / 3:
        position = "on your left"
    elif center_x > (2 * img_width) / 3:
        position = "on your right"
    else:
        position = "in front of you"

    box_area_ratio = (box_width * box_height) / (img_width * img_height)
    if box_area_ratio > 0.15:
        distance = "very close"
    elif box_area_ratio > 0.03:
        distance = "nearby"
    else:
        distance = "at a distance"

    return position, distance


def evaluate_safety_risk(label, distance, position):
    high_risk_objects = [
        "car", "truck", "bus", "motorbike", "bicycle",
        "fire hydrant", "stop sign",
    ]
    medium_risk_objects = [
        "chair", "couch", "bed", "dining table", "door",
        "stairs", "person",
    ]

    if label in high_risk_objects and distance in ["very close", "nearby"]:
        return (
            "HIGH",
            f"DANGER! Moving {label} {distance} {position}. Move away immediately!",
        )

    elif (
        label in medium_risk_objects
        and position == "in front of you"
        and distance == "very close"
    ):
        return (
            "MEDIUM",
            f"Caution! {label} very close in front of you. Watch your step.",
        )

    return "LOW", None


def run_safety_and_spatial_detection(
    yolo_model, memory_manager, image_path="captured_scene.jpg"
):
    if not os.path.exists(image_path):
        return "Image file not found.", False

    frame = cv2.imread(image_path)
    img_height, img_width, _ = frame.shape

    results = yolo_model(image_path, conf=0.25, verbose=False)

    descriptions = []
    memory_entries = []
    highest_risk_warning = None
    is_urgent = False

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            label = yolo_model.names[class_id]
            coords = box.xyxy[0].tolist()

            position, distance = get_spatial_info(coords, img_width, img_height)
            desc_str = f"{distance} {position}"
            descriptions.append(f"{label} {desc_str}")
            memory_entries.append(f"{label} {desc_str}")

            risk_level, warning_msg = evaluate_safety_risk(label, distance, position)
            if risk_level == "HIGH":
                highest_risk_warning = warning_msg
                is_urgent = True
                break
            elif risk_level == "MEDIUM" and not highest_risk_warning:
                highest_risk_warning = warning_msg

        if is_urgent:
            break

    if memory_entries:
        memory_manager.update_memory(memory_entries)

    if highest_risk_warning:
        return highest_risk_warning, is_urgent

    if not descriptions:
        return "No clear objects or obstacles detected nearby.", False

    return "I see " + ", and ".join(descriptions) + ".", False


def run_ocr(reader, image_path="captured_scene.jpg"):
    if not os.path.exists(image_path):
        return "Image file not found."

    img = cv2.imread(image_path)
    if img is None:
        return "Could not read the captured image."

    results = reader.readtext(img)
    valid_words = []
    for bbox, text, conf in results:
        text = text.strip()
        if conf > 0.5 and len(text) > 1 and not text.isdigit():
            valid_words.append(text)

    if not valid_words:
        return "No clear text detected in front of the camera."

    raw_sentence = " ".join(valid_words)
    clean_text = re.sub(r"[^\w\s,.?!\'-]", "", raw_sentence)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    return clean_text if clean_text else "Could not read clear text."


def run_vlm(
    image_path="captured_scene.jpg",
    prompt="Describe what you see in short detail for a visually impaired user.",
):
    if not os.path.exists(image_path):
        return "Image not captured properly."

    url = "http://localhost:11434/api/generate"

    try:
        with open(image_path, "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode("utf-8")

        payload = {
            "model": "minicpm-v",
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
        }

        response = requests.post(url, json=payload, timeout=90)
        if response.status_code == 200:
            res_json = response.json()
            return res_json.get("response", "Could not analyze the scene.")
        else:
            return "Ollama server error."
    except Exception as e:
        print("VLM Error:", e)
        return "Unable to connect to Ollama server."


def main():
    print("Initializing Vision-AidX Packaged Demo System...")
    speech = SpeechEngine()

    speech.speak("System initialized with Memory and Safety Layer.")

    print("Loading YOLOv8 model...")
    yolo_model = YOLO("yolov8n.pt")

    print("Loading EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=False)

    memory_manager = VisionMemory()

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    speech.speak("Vision Aid X is fully ready.")

    print("\n=== Vision-AidX Master System Ready ===")
    print("Voice Triggers:")
    print(" - Safety & Spatial Scan: 'detect', 'object', 'obstacle', 'kya hai'")
    print(" - Memory Search: 'where is my [item]', 'find my [item]', 'kahan hai'")
    print(" - OCR/Text Reading: 'read', 'text', 'padho', 'book'")
    print(" - Scene Description: 'describe', 'what', 'samne', 'see'")
    print(" - Quit: 'stop', 'exit', 'quit'\n")

    try:
        while True:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("Listening for command...")
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)
                    command = recognizer.recognize_google(audio).lower()
                    print(f"User said: '{command}'")

                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except sr.RequestError:
                    print("Network error with speech recognition.")
                    continue

            if any(w in command for w in ["exit", "quit", "stop", "quite"]):
                speech.speak("Shutting down Vision Aid X. Goodbye!")
                break

            elif any(w in command for w in ["where is", "find", "kahan hai", "search"]):
                cleaned = command
                for phrase in ["where is my", "where is the", "where is",
                               "find my", "find the", "find",
                               "kahan hai", "search"]:
                    cleaned = cleaned.replace(phrase, "")
                target_obj = cleaned.strip()
                if not target_obj:
                    target_obj = "that item"

                recall_result = memory_manager.recall(target_obj)
                print(f"\n[Memory Recall]: {recall_result}\n")
                speech.speak(recall_result)

            elif any(w in command for w in ["detect", "object", "obstacle", "item", "kya hai", "danger"]):
                speech.speak("Scanning surroundings.")
                if capture_frame("captured_scene.jpg"):
                    message, urgent = run_safety_and_spatial_detection(
                        yolo_model, memory_manager, "captured_scene.jpg"
                    )
                    print(f"\n[Spatial & Safety Result]: {message}\n")
                    speech.speak(message, urgent=urgent)
                else:
                    speech.speak("Failed to capture image from camera.")

            elif any(w in command for w in ["read", "text", "padho", "book", "label"]):
                speech.speak("Capturing image for text reading.")
                if capture_frame("captured_scene.jpg"):
                    speech.speak("Processing text, please wait.")
                    text_out = run_ocr(reader, "captured_scene.jpg")
                    print(f"\n[Clean OCR Result]: {text_out}\n")
                    speech.speak(f"The text says: {text_out}")
                else:
                    speech.speak("Failed to capture image from camera.")

            elif any(w in command for w in ["describe", "what", "samne", "see", "look"]):
                speech.speak("Capturing image to analyze scene.")
                if capture_frame("captured_scene.jpg"):
                    speech.speak("Analyzing scene, please wait.")
                    desc_out = run_vlm("captured_scene.jpg")
                    print(f"\n[VLM Result]: {desc_out}\n")
                    speech.speak(desc_out)
                else:
                    speech.speak("Failed to capture image from camera.")

    finally:
        speech.stop()


if __name__ == "__main__":
    main()
