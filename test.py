import cv2
import yaml
import numpy as np
import argparse
from utils.dataset import create_dataset, LoadImages
from utils.general import check_file, xywh2xyxy
from utils.plot import plot_one_box
from models.yolo import Model

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/coco.yaml", help="data.yaml path")
opt = parser.parse_args()
opt.data = check_file(opt.data)
with open(opt.data, "r") as f:
    data_dict = yaml.load(f, Loader=yaml.FullLoader)
train_img_dir = data_dict["train_img_dir"]
train_annotation_file = data_dict["train_annotation_file"]
names = data_dict["names"]
# print(train_path)
dataset, dataloader = create_dataset(train_img_dir, train_annotation_file, batch_size=4)

# model = Model()

# pbar = tqdm(dataloader)
for i, (img0, img, labels) in enumerate(dataloader):
    if i != 0:
        break
    for label in labels:
        img_id = int(label[0])
        name_id = int(label[1])
        plot_one_box(label[2:6], img0[img_id])
    for j, draw_img in enumerate(img0):
        cv2.imwrite(f"check_images/{j}.jpg", draw_img)
