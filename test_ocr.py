import queue
import threading
import time
import cv2
import easyocr
import pyttsx3


class SpeechWorker:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 170)
        self.queue = queue.Queue()
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop:
            try:
                text = self.queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if text:
                self.engine.say(text)
                self.engine.runAndWait()

    def say(self, text: str):
        self.queue.put(text)

    def stop(self):
        self._stop = True
        self.thread.join(timeout=1)


def main():
    print("Loading EasyOCR reader...")
    # Initialize EasyOCR reader
    reader = easyocr.Reader(["en"], gpu=False)
    speech = SpeechWorker()

    print("Opening camera...")
    cap = cv2.VideoCapture(0)

    # Set High Resolution for clearer text focus
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    print("\n--- INSTRUCTIONS ---")
    print("1. Hold book 1 to 1.5 feet away under good lighting.")
    print("2. Press SPACE to capture.")
    print("3. Press Q to quit.\n")

    frame = None
    while True:
        ok, live_frame = cap.read()
        if not ok:
            break

        cv2.imshow("Position text & press SPACE", live_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            frame = live_frame
            break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame is None:
        print("No frame captured.")
        speech.stop()
        return

    cv2.imwrite("captured_frame.jpg", frame)
    print("Saved captured_frame.jpg")

    print("Running OCR on original frame...")
    # Direct original frame usage gives maximum accuracy on books & clear fonts
    results = reader.readtext(frame, paragraph=True)

    extracted_texts = [
        text.strip() for (bbox, text) in results if len(text.strip()) > 1
    ]

    if not extracted_texts:
        print("No readable text found! Bring book closer or improve lighting.")
        speech.say(
            "No readable text detected. Please hold the book steady under good light."
        )
        time.sleep(3)
        speech.stop()
        return

    full_text = " ".join(extracted_texts)

    print("\n--- Clean Extracted Text ---")
    print(full_text)
    print("----------------------------\n")

    print("Speaking extracted text...")
    speech.say(full_text)

    # Calculate speech duration dynamically
    words_count = len(full_text.split())
    time.sleep(max(3, words_count * 0.4))
    speech.stop()


if __name__ == "__main__":
    main()