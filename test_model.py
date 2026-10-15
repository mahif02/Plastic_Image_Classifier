from ultralytics import YOLO

# Load your trained model
model = YOLO(
    "runs/detect/plastic_project/plastic_detector-4/weights/best.pt"
)

# Image to test
image_path = "test_images/image.png"


# Make prediction
results = model.predict(
    source=image_path,
    conf=0.25,
    save=True
)

print("Prediction completed!")
print("Check the runs/detect folder for the result.")
from ultralytics import YOLO

model = YOLO(
    "runs/detect/plastic_project/plastic_detector-6/weights/best.pt"
)

image_path = "test_images/image.png"

results = model.predict(
    source=image_path,
    conf=0.25,
    save=True
)

print("Prediction completed!")
print("Check the runs/detect folder for the result.")