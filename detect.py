import torch
import argparse
import cv2
from models.yolo import Model
from utils.dataset import create_dataset, LoadImages


def detect(opt):
    source, weights, save_img, view_img = (
        opt.source,
        opt.weights,
        opt.save_img,
        opt.view_img,
    )
    model = Model()
    ckpt = torch.load(weights)
    model.load_state_dict(ckpt["model_state_dict"])
    images = LoadImages(source)
    for i, img in enumerate(images):
        pred = model(img[None, ...])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, default="data/images", help="the image to be detected"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="best.pt",
        help="the path of weight to be loaded",
    )
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--save-img", action="store_true", help="save image or not")
    parser.add_argument("--view-img", action="store_true", help="view image or not")
    opt = parser.parse_args()

    print(opt)
    detect(opt)
