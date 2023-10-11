from ultralytics import YOLO

device = 1
experiment_name = "n_test"
data_path = "Checkers-Piece-Detection-1/data.yaml"

# Train
model = YOLO("yolov8n.pt")
model.train(data=data_path, epochs=100, imgsz=(640, 400), name=experiment_name)

# Val
model = YOLO(f"runs/detect/{experiment_name}/weights/best.pt")
metrics = model.val(device=device, split="test")
