from ultralytics import YOLO
import cv2
import time
import torch

# -------------------------
# DEVICE
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
half = device == "cuda"

# -------------------------
# MODELS
# -------------------------
pothole_model = YOLO("../models/best_pothole.pt").to(device)
garbage_model = YOLO("../models/best_garbage.pt").to(device)

# -------------------------
# CAMERA
# -------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera not working")
    exit()

RED = (0, 0, 255)
GREEN = (0, 255, 0)

prev_time = time.time()

# -------------------------
# MAIN LOOP
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    # -------------------------
    # RUN FRESH INFERENCE EVERY FRAME
    # -------------------------
    pothole_res = pothole_model.predict(
        frame,
        imgsz=416,
        conf=0.5,
        iou=0.5,
        device=device,
        half=half,
        verbose=False
    )[0]

    garbage_res = garbage_model.predict(
        frame,
        imgsz=416,
        conf=0.5,
        iou=0.5,
        device=device,
        half=half,
        verbose=False
    )[0]

    # -------------------------
    # DRAW POTHOLES (RED)
    # -------------------------
    for box in pothole_res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
        cv2.putText(frame, f"Pothole {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)

    # -------------------------
    # DRAW GARBAGE (GREEN)
    # -------------------------
    for box in garbage_res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
        cv2.putText(frame, f"Garbage {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

    # -------------------------
    # FPS
    # -------------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("AI Detection (REAL TIME FIXED)", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()