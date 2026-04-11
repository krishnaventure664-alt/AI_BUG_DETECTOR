import os
import shutil
from pathlib import Path

from ultralytics import YOLO


MODEL_PATHS = ("best.pt", "yolov8n.pt")


def _resolve_model_path() -> str:
    base_dir = Path(__file__).resolve().parent

    for filename in MODEL_PATHS:
        candidate = base_dir / filename
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "No YOLO model file found. Place 'best.pt' or 'yolov8n.pt' in the project root."
    )


model = YOLO(_resolve_model_path())


def detect_objects(file):
    temp_file = "temp.jpg"

    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(temp_file)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append(
                {
                    "object": model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                }
            )

    if os.path.exists(temp_file):
        os.remove(temp_file)

    return detections
