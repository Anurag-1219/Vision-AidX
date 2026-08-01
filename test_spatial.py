import os
import cv2
from ultralytics import YOLO

def get_spatial_info(box, img_width, img_height):
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = (x1 + x2) / 2

    # 1. Horizontal Position (Left, Center, Right)
    if center_x < img_width / 3:
        position = "on your left"
    elif center_x > (2 * img_width) / 3:
        position = "on your right"
    else:
        position = "in front of you"

    # 2. Relative Distance (Close, Medium, Far)
    box_area_ratio = (box_width * box_height) / (img_width * img_height)
    if box_area_ratio > 0.15:
        distance = "very close"
    elif box_area_ratio > 0.03:
        distance = "nearby"
    else:
        distance = "at a distance"

    return position, distance

def test_spatial_detection():
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera capture failed.")
        return

    print("Capturing frame from camera...")
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Error: Could not read frame from camera.")
        return

    img_height, img_width, _ = frame.shape
    print(f"Image Resolution: {img_width}x{img_height}")

    # Confidence 25% for better detection
    results = model(frame, conf=0.25, verbose=True)

    detections = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
            position, distance = get_spatial_info(coords, img_width, img_height)
            detections.append(f"{label} -> {position} ({distance})")

    print("\n==========================================")
    print("       SPATIAL DETECTION RESULTS          ")
    print("==========================================")
    if detections:
        for d in detections:
            print(f"📍 {d}")
    else:
        print("❌ No objects detected! Try putting a bottle, chair, or phone directly in front of the camera.")
    print("==========================================\n")

if __name__ == "__main__":
    test_spatial_detection()