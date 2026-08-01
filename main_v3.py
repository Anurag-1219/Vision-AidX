import cv2
import requests
import json
import pyttsx3
import base64
import time
import speech_recognition as sr

# --- 1. Audio Voice Output ---
def speak(text):
    if not text or len(text.strip()) == 0:
        return
    
    clean_text = text.replace('\n', ' ').strip()
    print(f"\n[AI Assistant Speaking]: {clean_text}")
    
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 145)
        engine.say(clean_text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[TTS Error]: {e}")

# --- 2. Voice Input Listener ---
def listen_voice_command():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    
    try:
        with sr.Microphone() as source:
            print("\n[System]: Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=6)
            print("[System]: Processing speech...")
            
            command = recognizer.recognize_google(audio)
            print(f"[User Voice Input]: '{command}'")
            return command.lower()
    except sr.WaitTimeoutError:
        print("[Voice Warning]: No speech detected.")
        return None
    except sr.UnknownValueError:
        print("[Voice Warning]: Speech unclear.")
        return None
    except Exception as e:
        print(f"[Mic Driver Error]: {e}")
        return None

# --- 3. Image Encoder ---
def prepare_image_payload(frame):
    img_resized = cv2.resize(frame, (640, 480))
    _, buffer = cv2.imencode('.jpg', img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buffer).decode("utf-8")

# --- 4. High-Accuracy VLM Reasoning Engine ---
def ask_vision_model(encoded_image, user_prompt=None):
    url = "http://localhost:11434/api/generate"
    
    if user_prompt and len(user_prompt.strip().split()) > 1:
        prompt = (
            f"User asked: '{user_prompt}'. "
            "Examine the image carefully. Identify the main object in front of the camera or held in hand. "
            "Answer clearly and accurately in one concise sentence."
        )
    else:
        prompt = (
            "Identify the main object in front of the camera or held in hand. "
            "Describe what it is and its position in one clear sentence."
        )
    
    # Auto-detect downloaded model (MiniCPM / Qwen / LLaVA)
    payload = {
        "model": "minicpm-v", # If you pulled qwen2-vl or llava, update name here
        "prompt": prompt,
        "images": [encoded_image],
        "stream": False,
        "options": {
            "num_gpu": 0,
            "temperature": 0.2
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=35)
        if response.status_code == 200:
            result = response.json()
            ans = result.get("response", "").strip()
            print(f"[AI Raw Response]: '{ans}'")
            return ans if ans else "I could not identify the object clearly."
        else:
            # Fallback if model name was qwen2-vl or llava
            payload["model"] = "qwen2-vl"
            res2 = requests.post(url, json=payload, timeout=35)
            if res2.status_code == 200:
                return res2.json().get("response", "").strip()
            return f"Error response code {response.status_code}"
    except Exception as e:
        return f"Connection failed: {e}"

# --- 5. Main Autonomous Loop ---
def run_vision_aid():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not detected.")
        return

    print("\n========================================================")
    print("   VisionAid X — Phase 2: High-Accuracy AI Assistant   ")
    print("========================================================")
    print("Press 'v' to Speak Voice Command")
    print("Press 'c' for Direct Capture")
    print("Press 'q' to Exit")

    speak("Vision Aid online with high accuracy vision.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("VisionAid X - Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        # Voice Trigger
        if key == ord('v'):
            speak("Listening")
            cmd = listen_voice_command()
            
            # Clear frame buffer for fresh shot
            for _ in range(3):
                cap.read()
            _, fresh_frame = cap.read()
            
            encoded_img = prepare_image_payload(fresh_frame)
            speak("Analyzing scene")
            
            ans = ask_vision_model(encoded_img, user_prompt=cmd)
            speak(ans)

        # Manual Keyboard Trigger
        elif key == ord('c'):
            for _ in range(3):
                cap.read()
            _, fresh_frame = cap.read()
            
            encoded_img = prepare_image_payload(fresh_frame)
            speak("Analyzing scene")
            
            ans = ask_vision_model(encoded_img)
            speak(ans)

        elif key == ord('q'):
            speak("Shutting down")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision_aid()