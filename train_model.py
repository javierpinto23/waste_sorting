from ultralytics import YOLO

# Load the model
model = YOLO("yolov8m.yaml")

# Train the model
results = model.train(data="data2.yaml", epochs=1)
