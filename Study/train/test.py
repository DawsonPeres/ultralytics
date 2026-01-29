from ultralytics import YOLO
# 加载新模型
model = YOLO("yolo26n.yaml")
# # 加载已有模型
# model = YOLO("yolo26n.pt")
# # 从YML构建新模型并转已有的移权重
# model = YOLO("yolo26n.yaml").load("yolo26n.pt")

results = model.train(data="coco8.yaml", epochs=100, imgsz=640)