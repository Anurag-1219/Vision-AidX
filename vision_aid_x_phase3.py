import cv2
import requests
import json
import base64
import pyttsx3
import threading
import queue
import time

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "moondream"
SCAN_INTERVAL = 6  # Seconds between auto-scans

# Global state & Queue for Non-blocking Voice Output
audio_queue = queue.Queue()
is_processing = False
auto_scan_enabled = False
last_scan_time = 0

# --- TTS WORKER THREAD ---
def tts_worker():
    """Background worker thread so audio doesn't freeze the camera feed"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)  # Speech speed
        engine.setProperty('volume', 1.0)
    except Exception as e:
        print(f"Error initializing pyttsx3: {e}")
        engine = None
    
    while True:
        text = audio_queue.get()
        if text is None:  # Shutdown signal
            break
        print(f"\n🔊 Audio Output: {text}\n")
        if engine:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Audio playback error: {e}")
        audio_queue.task_done()

# Start Audio Thread
audio_thread = threading.Thread(target=tts_worker, daemon=True)
audio_thread.start()

def speak(text):
    """Add text to speech queue safely"""
    # Empty queue if old audio is pending to keep descriptions fresh
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break
    audio_queue.put(text)

# --- OLLAMA INFERENCE ---
def query_moondream(image_bytes, prompt):
    """Sends image & prompt to local Ollama Moondream API"""
    global is_processing
    is_processing = True
    
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "options": {
                "num_predict": 75,
                "temperature": 0.2
            }
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=25)
        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            if result:
                speak(result)
            else:
                speak("No response generated.")
        else:
            speak("Error reaching vision model.")
    except Exception as e:
        print(f"Error during inference: {e}")
        speak("Vision analysis failed.")
    finally:
        is_processing = False

def trigger_async_analysis(frame, prompt):
    """Helper to encode image and launch thread"""
    if is_processing:
        return
    
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    img_bytes = buffer.tobytes()
    
    thread = threading.Thread(target=query_moondream, args=(img_bytes, prompt), daemon=True)
    thread.start()

# --- MAIN CAMERA LOOP ---
def main():
    global auto_scan_enabled, last_scan_time
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ Camera not accessible!")
        speak("Camera not accessible.")
        return

    print("\n" + "="*50)
    print("🌟 VisionAid X - Phase 3 (Advanced Assistant)")
    print("="*50)
    print(" Control Keys:")
    print("  [A] - Toggle Continuous Auto-Scan ON/OFF")
    print("  [C] - Capture & Describe Scene")
    print("  [S] - Spatial & Distance Analysis (Left/Center/Right)")
    print("  [Q] - Quit Application")
    print("="*50 + "\n")
    
    speak("Vision Aid X is ready. Press A for auto scan, or C for manual scan.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame")
            break

        current_time = time.time()
        
        # --- CONTINUOUS AUTO-SCAN LOGIC ---
        if auto_scan_enabled and not is_processing:
            if current_time - last_scan_time >= SCAN_INTERVAL:
                last_scan_time = current_time
                prompt = "Briefly describe key objects in front of the viewer in 1 simple sentence."
                trigger_async_analysis(frame, prompt)

        # UI Overlay Status
        status_text = "AUTO-SCAN: ON" if auto_scan_enabled else "AUTO-SCAN: OFF"
        status_color = (0, 255, 0) if auto_scan_enabled else (0, 0, 255)
        
        if is_processing:
            cv2.putText(frame, "Analyzing...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, status_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, "[A] Auto-Scan | [C] Capture | [S] Spatial | [Q] Quit", 
                    (30, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("VisionAid X - Assistant Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('a'):
            auto_scan_enabled = not auto_scan_enabled
            state = "enabled" if auto_scan_enabled else "disabled"
            print(f"🔄 Auto-scan {state}")
            speak(f"Auto scan {state}")
            last_scan_time = time.time()
            
        elif key == ord('c'):
            print("📸 Manual Capture Triggered")
            prompt = "Describe what is in front of the user clearly in 1 or 2 sentences."
            trigger_async_analysis(frame, prompt)
            
        elif key == ord('s'):
            print("🧭 Spatial / Distance Query Triggered")
            prompt = "Identify main obstacles or objects. Specify if they are on the left, center, or right, and estimate if they are close or far."
            trigger_async_analysis(frame, prompt)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
