import cv2
import torch
import pyttsx3
import threading
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Reliable Thread-Safe Speech Worker ---
class SpeechWorker:
    def __init__(self):
        self.lock = threading.Lock()

    def speak(self, text):
        def _say():
            with self.lock:
                print(f"\n[Speaking]: {text}")
                # Re-initialize engine per speech thread to prevent event loop freeze
                try:
                    engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    print(f"Speech error: {e}")

        threading.Thread(target=_say, daemon=True).start()

speaker = SpeechWorker()

print("Loading Lightweight BLIP Model on CPU...")

model_id = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id)

device = "cpu"
model.to(device)

print("BLIP Model Loaded Successfully!")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\n--- INSTRUCTIONS ---")
print("Press SPACE to capture frame for Scene Description.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("VisionAid X - Scene Description Test (Press SPACE)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # SPACE key pressed
        print("\nCapturing image and generating description...")
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        inputs = processor(images=pil_image, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        print(f"Generated Description: {generated_text}")
        speaker.speak(generated_text)

    elif key == ord('q'):  # 'q' key to exit
        break

cap.release()
cv2.destroyAllWindows()