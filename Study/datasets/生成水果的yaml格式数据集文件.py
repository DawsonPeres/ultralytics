import os
import shutil
import random
import yaml

# ------------------ 配置 ------------------
SOURCE_DIR = r"C:\Users\Administrator\.cache\kagglehub\datasets\moltean\fruits\versions\76\fruits-360_100x100"
OUTPUT_DIR = r"D:\Work\LLM\GitHub\ultralytics\Study\datasets\fruits"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
TEST_RATIO = 0.0  # 可以设置为非0，如果有测试集

# ------------------ 创建目录 ------------------
for split in ["train", "val", "test"]:
    split_path = os.path.join(OUTPUT_DIR, split)
    os.makedirs(split_path, exist_ok=True)

# ------------------ 分类拆分 ------------------
classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
print(f"Found {len(classes)} classes: {classes}")

for cls in classes:
    cls_src = os.path.join(SOURCE_DIR, cls)
    images = [f for f in os.listdir(cls_src) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val

    # 创建目标类别目录
    for split, n in zip(["train", "val", "test"], [n_train, n_val, n_test]):
        cls_dest = os.path.join(OUTPUT_DIR, split, cls)
        os.makedirs(cls_dest, exist_ok=True)
        for img_name in images[:n]:
            shutil.copy(os.path.join(cls_src, img_name), os.path.join(cls_dest, img_name))
        images = images[n:]  # 剩余部分给下一个 split

# ------------------ 生成 data.yaml ------------------
data_dict = {
    'train': os.path.join(OUTPUT_DIR, 'train'),
    'val': os.path.join(OUTPUT_DIR, 'val'),
    'test': os.path.join(OUTPUT_DIR, 'test'),
    'nc': len(classes),
    'names': classes
}

yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_dict, f)

print("Dataset organized successfully!")
print("data.yaml path:", yaml_path)
