import queue
import threading

import cv2
import easyocr
import pyttsx3


class SpeechWorker:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 175)
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
    reader = easyocr.Reader(["en"], gpu=False)
    speech = SpeechWorker()

    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    print("A window will open showing your camera feed.")
    print("Position your text clearly in view, then press SPACE to capture.")
    print("Press Q to quit without capturing.")

    frame = None
    while True:
        ok, live_frame = cap.read()
        if not ok:
            break

        cv2.imshow("Position your text, then press SPACE", live_frame)
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

    print("Running OCR...")
    results = reader.readtext(frame)

    good_results = [(text, conf) for (bbox, text, conf) in results if conf > 0.5]

    if not good_results:
        print("No confident text detected. Try again with clearer/larger text in view.")
        speech.stop()
        return

    print("\n--- Text detected ---")
    for text, confidence in good_results:
        print(f"  '{text}'  (confidence: {confidence:.2f})")

    full_sentence = ". ".join(text for text, conf in good_results)
    print(f"\nSpeaking: {full_sentence}")
    speech.say(full_sentence)

    # give speech time to finish before the program exits
    import time
    time.sleep(len(full_sentence.split()) * 0.5 + 2)
    speech.stop()


if __name__ == "__main__":
    main()