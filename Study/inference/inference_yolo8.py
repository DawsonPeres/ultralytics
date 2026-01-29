from ultralytics import YOLO

model = YOLO(r"D:\Work\LLM\GitHub\ultralytics\Study\train\study_yolo8cls_fruit\yolo8n-fruit-classification\weights\best.pt")  # build a new model from YAML

# img_path = r'D:\Work\LLM\GitHub\ultralytics\Study\datasets\fruits\versions\76\fruits-360_100x100\fruits_dataset\val\Apple 5\r0_3_100.jpg'
img_path = r'D:\Work\LLM\GitHub\ultralytics\Study\datasets\fruits\versions\76\fruits-360_100x100\fruits_dataset\val\test\test_1.jpg'

results = model(img_path)  # predict on an image file
# print(results)
for result in results:
    print(dir(result))
    print(type(result))
    # result.probs 是分类概率对象
    top1_index = result.probs.top1
    top1_conf = result.probs.top1conf
    class_name = result.names[top1_index]
    print(f"预测结果: {class_name}, 置信度: {top1_conf:.2f}")


    top5_index = result.probs.top5
    top5_conf = result.probs.top5conf
    print(f"Top-5 预测结果索引: {top5_index}, 置信度: {top5_conf}, 类别名称: {[result.names[i] for i in top5_index]}")


    # 显示图片
    # result.show()





