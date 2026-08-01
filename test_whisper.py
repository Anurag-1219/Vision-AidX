import time
import whisper
import pyttsx3
import threading
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

# --- Reliable Speech Worker ---
class SpeechWorker:
    def __init__(self):
        self.lock = threading.Lock()

    def speak(self, text):
        def _say():
            with self.lock:
                print(f"\n[Speaking]: {text}")
                try:
                    engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    print(f"Speech error: {e}")

        threading.Thread(target=_say, daemon=True).start()

speaker = SpeechWorker()

print("Loading Whisper 'tiny' Model on CPU...")
model = whisper.load_model("tiny")
print("Whisper Model Loaded Successfully!")

def record_and_transcribe(duration=5, fs=16000):
    input("\n>>> Press ENTER when you are ready to SPEAK... ")
    
    print("\n[Mic Active] Recording NOW... Speak your command!")
    
    # Record audio directly into float32 array in RAM
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait for recording to complete
    print("[Mic Stopped] Recording complete.")

    # Flatten array from 2D (samples, 1) to 1D (samples,)
    audio_data = recording.flatten()

    print("Transcribing your audio directly from memory using Whisper AI...")
    
    # Pass float32 numpy array directly to whisper (No FFmpeg / File load needed!)
    result = model.transcribe(audio_data, fp16=False)
    text = result['text'].strip()
    return text

if __name__ == "__main__":
    print("\n--- VISIONAID X VOICE INPUT TEST (FFmpeg Free) ---")
    
    # Record and Transcribe directly
    user_command = record_and_transcribe(duration=5)
    
    print(f"\n>>> RESULT (What Whisper Heard): \"{user_command}\"")
    
    if user_command:
        speaker.speak(f"You said: {user_command}")
    else:
        speaker.speak("I could not hear anything. Please try again.")