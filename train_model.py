from ultralytics import YOLO

# Load a small YOLO model
model = YOLO("yolo11n.pt")

# Train the model
model.train(
    data="data.yaml",
    epochs=10,
    imgsz=320,
    batch=4,
    project="plastic_project",
    name="plastic_detector"
)

print("Training completed!")
from ultralytics import YOLO

model = YOLO(
    "runs/detect/plastic_project/plastic_detector-5/weights/last.pt"
)

model.train(
    resume=True
)

print("Training completed!")