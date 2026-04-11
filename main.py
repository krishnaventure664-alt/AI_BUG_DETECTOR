from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

model = YOLO("yolov8n.pt")

@app.get("/")
def home():
    return {"message": "Backend running 🚀"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    temp_file = "temp.jpg"

    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(temp_file)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": int(box.cls[0]),
                "confidence": float(box.conf[0])
            })

    os.remove(temp_file)

    return {"detections": detections}