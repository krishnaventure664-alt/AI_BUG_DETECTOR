from ultralytics import YOLO

model = YOLO("../models/best_garbage.pt")

results = model("../data_garbage/valid/images", save=True)

print("Testing done. Check runs/detect/predict")