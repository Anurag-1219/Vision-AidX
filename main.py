import time
import queue
import threading

import cv2
from ultralytics import YOLO
import pyttsx3


CAMERA_INDEX = 0
MODEL_NAME = "yolov8s.pt"
CONFIDENCE_THRESHOLD = 0.75
SPEAK_COOLDOWN_SECONDS = 3.0
FRAME_SKIP = 2


class SpeechWorker:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 175)
        self.queue = queue.Queue()
        self.enabled = True
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop:
            try:
                text = self.queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if self.enabled and text:
                self.engine.say(text)
                self.engine.runAndWait()

    def say(self, text: str):
        if self.queue.empty():
            self.queue.put(text)

    def stop(self):
        self._stop = True
        self.thread.join(timeout=1)


def build_announcement(labels):
    if not labels:
        return ""
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    parts = []
    for label, count in counts.items():
        if count == 1:
            parts.append(label)
        else:
            parts.append(f"{count} {label}s")
    return ", ".join(parts)


def main():
    print("Loading YOLOv8 model...")
    model = YOLO(MODEL_NAME)

    print(f"Opening camera index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera. Check CAMERA_INDEX or that no other app is using the webcam."
        )

    speech = SpeechWorker()
    last_spoken = {}
    frame_count = 0

    print("Running. Press 'q' to quit, 's' to toggle speech.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            frame_count += 1
            run_detection = (frame_count % FRAME_SKIP == 0)

            if run_detection:
                results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
                labels_this_frame = []

                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])
                    labels_this_frame.append(label)
                    print(f"  detected: {label} ({conf:.2f})")

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                now = time.time()
                to_announce = []
                for label in set(labels_this_frame):
                    if now - last_spoken.get(label, 0) > SPEAK_COOLDOWN_SECONDS:
                        to_announce.append(label)
                        last_spoken[label] = now

                announcement = build_announcement(to_announce)
                if announcement:
                    print(f"Speaking: {announcement}")
                    speech.say(announcement)

            cv2.imshow("VisionAid X - Phase 1", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                speech.enabled = not speech.enabled
                print(f"Speech {'enabled' if speech.enabled else 'muted'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        speech.stop()
        print("Shut down cleanly.")


if __name__ == "__main__":
    main()