from ultralytics import YOLO
import os

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")  # path to your model

# Run prediction on Falcon test set (no labels)
model.predict(
    source="data/test/images",  # image folder
    conf=0.25,
    save=True,
    save_txt=False,
    imgsz=640,
    project="runs",
    name="blind_test_output",
    exist_ok=True,
    show=False
)