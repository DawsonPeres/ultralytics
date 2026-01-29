from ultralytics import YOLO

def main():
    model = YOLO('yolov8n-cls.pt')

    result = model.train(
        data=r"D:\Work\LLM\GitHub\ultralytics\Study\datasets\fruits\versions\76\fruits-360_100x100\fruits_dataset",
        epochs=10,
        imgsz=100,
        device=0,
        batch=64,
        workers=4,
        project="study_yolo8cls_fruit",
        name="yolo8n-fruit-classification",
    )

if __name__ == '__main__':
    main()