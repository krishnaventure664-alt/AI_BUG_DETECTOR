from ultralytics import YOLO

# STEP 1: load model
model = YOLO("../models/best.pt")

# STEP 2: run prediction on images
results = model("../data/test/images", save=True)

print("DONE! Check runs/detect/predict folder")