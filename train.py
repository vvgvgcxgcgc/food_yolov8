from ultralytics import YOLO
model = YOLO("yolov8n.yaml")

# Use the model
results = model.train(data="/workdir/radish/nguyenqh/food_yolov8/config.yaml", epochs=50, pretrained=True, iou=0.5, visualize=True,device=0, patience=0)  # train the model
results = model.val()
     