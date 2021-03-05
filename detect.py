import torch
import torchvision
import argparse
import cv2
from models.yolo import Model
from utils.dataset import create_dataset, LoadImages
from utils.general import non_max_suppression


def detect(opt):
    source, weights, conf_thres, iou_thres, save_img, view_img = (
        opt.source,
        opt.weights,
        opt.conf_thres,
        opt.iou_thres,
        opt.save_img,
        opt.view_img,
    )
    model = Model()
    ckpt = torch.load(weights)
    model.load_state_dict(ckpt["model_state_dict"])
    images = LoadImages(source)
    for img in images:
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        for i, det in enumerate(pred):
            if save_img:
                # save image
                pass
            if view_img:
                # show img
                pass


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
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="nms iou threshold"
    )
    opt = parser.parse_args()

    print(opt)
    detect(opt)
