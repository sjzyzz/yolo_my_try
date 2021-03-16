import glob
import os
import torch
import torch.tensor as tensor
import math
from pathlib import Path
import re
import random
import logging
import numpy as np

from utils.torch_utils import init_torch_seeds


def check_file(file):
    if os.path.isfile(file) or file == "":
        return file
    else:
        # question here: the original code is "./**/" but this is "../**/", maybe need to check the glob document and file system
        files = glob.glob("../**/" + file, recursive=True)
        # print(glob.glob("../**/", recursive=True))
        assert len(files), f"file {file} not found"
        assert (
            len(files) == 1
        ), f"Multiple files match {file}, specify exact path: {files}"
        return files[0]


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else x.copy()
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]

    return y


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else x.copy()

    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2

    return y


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _box_inter_union(boxes1, boxes2):
    area1 = box_area(boxes1)  # N * 1
    area2 = box_area(boxes2)  # M * 1

    lt = torch.max(boxes1[:, None, :2], boxes[:, :2])  # N * M * 2
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # N * M * 2

    wh = (rb - lt).clamp(min=0)  # N * M * 2
    inter = wh[:, :, 0] * wh[:, :, 1]  # N * M * 1

    union = area1[:, None] + area2 - inter

    return inter, union


def box_iou(boxes1, boxes2):
    inter, union = _box_inter_union(boxes1, boxes2)

    return inter, union


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def check_img_size(img_size, s=32):
    new_size = make_divisible(img_size, s)
    if new_size != img_size:
        print(
            "WARNING: --img-size %g must be multiple of max stride %g, updating to %g"
            % (img_size, s, new_size)
        )
    return new_size


def increment_path(path, exist_ok=True, sep=""):
    path = Path(path)
    if (path.exists and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{sep}{n}"


def get_latest_run(search_dir="."):
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""


def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s", level=logging.INFO if rank in [-1, 0] else logging.WARN
    )


def init_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def non_max_suppression(prediction, args):
    """
    Args:
        prediction: a tensor with shape of num_image * ?? * 85, and the format of single is cx,cy,w,h,obj,cls(80),and the coordinate is in absolute form
    Return:
        output: a tensor with shape of ? * 6, the 6 represent x,y,x,y,conf,cls
    """
    conf_thres = args.conf_thres
    # for every image
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.size(0)
    for xi, x in enumerate(prediction):
        # filter the prediction whose obj value is less than conf_thres
        x = x[conf_thres < x[:, 4:5]]
        x[:, 5:] *= x[:, 4:5]
        # find the bigest value among all 80 classes
        # get all parts and concatenate together and transform the xywh to xyxy
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j), 1)
        x = x[conf_thres < x[:, 4:5]]
        # TODO: perform a NMS

        output[xi] = x

    return output
