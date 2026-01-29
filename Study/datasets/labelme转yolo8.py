import os
import json
import shutil
import random
from pathlib import Path

# --- 配置区域 ---
# 1. LabelMe json 和图片所在的文件夹路径
INPUT_DIR = r'D:\Work\LLM\GitHub\ultralytics\Study\datasets\labelme_yolo8'
# 2. 输出的 YOLOv8 数据集路径
OUTPUT_DIR = r'D:\Work\LLM\GitHub\ultralytics\Study\datasets\labelme_yolo8\yolo_dataset'
# 3. 你的类别名称（必须与标注时的 label 完全一致，顺序决定 class_id）
CLASSES = ['xilanhua', 'jianguo', 'tudou', 'roubing', 'mianbao', 'huaping', 'hua', 'cao', 'banma', 'changjinglu']  # 示例，请修改为你实际的 label
# 4. 训练集/验证集比例
TRAIN_RATIO = 0.8


def convert(size, box):
    """ 将 box 坐标 (xmin, xmax, ymin, ymax) 转换为 yolo 格式 (x, y, w, h) """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def main():
    # 创建输出目录结构
    for folder in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', folder), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', folder), exist_ok=True)

    # 获取所有 json 文件
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    random.shuffle(json_files)

    split_index = int(len(json_files) * TRAIN_RATIO)
    train_files = json_files[:split_index]
    val_files = json_files[split_index:]

    print(f"Total files: {len(json_files)} | Train: {len(train_files)} | Val: {len(val_files)}")

    # 处理文件
    for files, subset in [(train_files, 'train'), (val_files, 'val')]:
        for json_file in files:
            json_path = os.path.join(INPUT_DIR, json_file)

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 图片处理
            image_name = data['imagePath']
            # 处理路径分隔符差异 (Windows/Linux)
            image_name = os.path.basename(image_name)

            src_image_path = os.path.join(INPUT_DIR, image_name)
            dst_image_path = os.path.join(OUTPUT_DIR, 'images', subset, image_name)

            # 检查图片是否存在（LabelMe json 有时只存路径）
            if not os.path.exists(src_image_path):
                # 尝试把后缀改成 json 同名的 jpg/png 找一下
                base_name = os.path.splitext(json_file)[0]
                for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                    temp_path = os.path.join(INPUT_DIR, base_name + ext)
                    if os.path.exists(temp_path):
                        src_image_path = temp_path
                        dst_image_path = os.path.join(OUTPUT_DIR, 'images', subset, base_name + ext)
                        break

            if not os.path.exists(src_image_path):
                print(f"Warning: Image for {json_file} not found, skipping.")
                continue

            # 复制图片
            shutil.copy(src_image_path, dst_image_path)

            # 标签处理
            img_w = data['imageWidth']
            img_h = data['imageHeight']
            txt_content = []

            for shape in data['shapes']:
                label = shape['label']
                if label not in CLASSES:
                    continue

                class_id = CLASSES.index(label)
                points = shape['points']

                # 将多边形/矩形点转换为外接矩形 (xmin, ymin, xmax, ymax)
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                xmin = min(x_coords)
                xmax = max(x_coords)
                ymin = min(y_coords)
                ymax = max(y_coords)

                # 转为 YOLO 格式
                b = (xmin, xmax, ymin, ymax)
                bb = convert((img_w, img_h), b)

                txt_content.append(f"{class_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")

            # 写入 txt 文件
            txt_filename = os.path.splitext(os.path.basename(dst_image_path))[0] + ".txt"
            with open(os.path.join(OUTPUT_DIR, 'labels', subset, txt_filename), 'w') as f:
                f.write('\n'.join(txt_content))

    print("Conversion complete!")


if __name__ == '__main__':
    main()