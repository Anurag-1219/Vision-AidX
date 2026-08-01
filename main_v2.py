import os
import cv2
import time
import torch
import whisper
import easyocr
import pyttsx3
import threading
import numpy as np
from PIL import Image
import sounddevice as sd
import imageio_ffmpeg
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Fix FFmpeg Binary Path for Whisper ---
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_path)
os.environ["PATH"] += os.pathsep + ffmpeg_dir

# --- Direct Non-Blocking Speech Worker ---
class SpeechWorker:
    def __init__(self):
        self.enabled = True

    def say(self, text: str):
        if self.enabled and text:
            print(f"\n[VisionAid Speaking]: {text}")
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 175)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"Speech Exception: {e}")

    def stop(self):
        pass

speaker = SpeechWorker()

# --- Initialize AI Models ---
print("==========================================")
print("     INITIALIZING VISIONAID X MODELS      ")
print("==========================================")

speaker.say("Starting VisionAid X. Loading models, please wait.")

# 1. YOLOv8 Model
print("\n[1/4] Loading YOLOv8 Object Detection...")
yolo_model = YOLO("yolov8s.pt")

# 2. EasyOCR Reader
print("\n[2/4] Loading EasyOCR...")
ocr_reader = easyocr.Reader(['en'], gpu=False)

# 3. BLIP Model
print("\n[3/4] Loading BLIP Scene Description Model...")
blip_id = "Salesforce/blip-image-captioning-large"
blip_processor = BlipProcessor.from_pretrained(blip_id)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_id).to("cpu")

# 4. Whisper Speech-To-Text
print("\n[4/4] Loading Whisper Voice Model...")
whisper_model = whisper.load_model("tiny")

print("\n>>> ALL MODELS LOADED SUCCESSFULLY! <<<\n")
speaker.say("VisionAid X is ready.")

# --- Helper Functions ---
def record_voice(duration=4, fs=16000):
    print("\n[Mic Active] Listening for command... SPEAK NOW!")
    speaker.say("Listening.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("[Mic Closed] Audio captured.")
    
    audio = recording.flatten()
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio

def transcribe_command(audio_data):
    result = whisper_model.transcribe(audio_data, fp16=False)
    return result['text'].strip().lower()

def run_ocr(frame):
    speaker.say("Reading text.")
    results = ocr_reader.readtext(frame)
    extracted_text = " ".join([res[1] for res in results]).strip()
    
    if extracted_text:
        print(f"\n[OCR Text Output]: {extracted_text}")
        speaker.say(f"Text found: {extracted_text}")
    else:
        print("\n[OCR Output]: No text detected.")
        speaker.say("No text detected in front of the camera.")

def run_scene_description(frame):
    speaker.say("Analyzing scene.")
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    inputs = blip_processor(images=pil_image, return_tensors="pt").to("cpu")
    with torch.no_grad():
        generated_ids = blip_model.generate(**inputs, max_new_tokens=50)
        description = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    print(f"\n[BLIP Output]: {description}")
    speaker.say(f"I see {description}")

def run_yolo_detection(frame):
    speaker.say("Detecting objects.")
    results = yolo_model(frame, verbose=False, conf=0.65)[0]
    labels_detected = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id]
        labels_detected.append(label)

    if labels_detected:
        counts = {}
        for l in labels_detected:
            counts[l] = counts.get(l, 0) + 1
        summary = ", ".join([f"{c} {l}" if c == 1 else f"{c} {l}s" for l, c in counts.items()])
        print(f"\n[YOLO Detected]: {summary}")
        speaker.say(f"I detect {summary}")
    else:
        print("\n[YOLO Output]: No objects detected.")
        speaker.say("No clear objects detected.")

# --- Main Engine Loop ---
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        speaker.say("Camera error. Please check your webcam connection.")
        return

    print("\n-----------------------------------------------------")
    print(" Press SPACEBAR on Camera Window to Give Voice Command")
    print(" Press 'q' on Camera Window to Quit")
    print("-----------------------------------------------------\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video feed.")
            break

        cv2.imshow("VisionAid X - Main Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            audio_data = record_voice(duration=4)
            command = transcribe_command(audio_data)
            print(f"\n>>> YOU SAID: \"{command}\"")

            ocr_keywords = ["read", "text", "book", "page", "red", "head", "write"]
            blip_keywords = ["describe", "see", "what", "look", "scene", "show", "color"]
            yolo_keywords = ["detect", "object", "item", "find", "things"]
            exit_keywords = ["exit", "stop", "quit", "bye"]

            if any(word in command for word in ocr_keywords):
                run_ocr(frame)

            elif any(word in command for word in blip_keywords):
                run_scene_description(frame)

            elif any(word in command for word in yolo_keywords):
                run_yolo_detection(frame)

            elif any(word in command for word in exit_keywords):
                speaker.say("Shutting down VisionAid X. Goodbye!")
                break
            else:
                print("\nCommand not recognized clearly.")
                speaker.say("Sorry, I did not catch that. Please try again.")

        elif key == ord('q'):
            speaker.say("Shutting down.")
            break

    cap.release()
    cv2.destroyAllWindows()
    speaker.stop()

if __name__ == "__main__":
    main()