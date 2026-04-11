from fastapi import FastAPI, File, UploadFile

from database import collection
from yolo_service_new import detect_objects

app = FastAPI()


@app.get("/")
def home():
    return {"message": "AI Bug Detector Backend Running"}


@app.post("/detect")
async def detect(file: UploadFile = File(...), lat: float = 0, lon: float = 0):
    detections = detect_objects(file)

    data = {
        "detections": detections,
        "location": {"latitude": lat, "longitude": lon},
    }
    collection.insert_one(data)

    return {
        "status": "success",
        "count": len(detections),
        "detections": detections,
    }


@app.get("/reports")
def get_reports():
    data = list(collection.find({}, {"_id": 0}))
    return {"total_reports": len(data), "reports": data}
