import torch
from torch.utils.data import Dataset
import argparse
import yaml
from pathlib import Path
import glob
import cv2
import os
from tqdm import tqdm
import numpy as np


from utils.general import check_file, xyxy2xywh, xywh2xyxy
from utils.plot import plot_one_box


class LoadImages:
    def __init__(self, path, img_size=640):
        p = str(Path(path))
        # p = os.path.abspath(path)
        self.files = glob.glob(os.path.join(p, "*.jpg"))
        self.img_size = img_size
        self.n = len(self.files)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < self.n:
            file = self.files[self.count]
            self.count += 1
        else:
            raise StopIteration

        img = cv2.imread(file)
        img = letterbox(img, self.img_size)[0]
        img = img.transpose(2, 0, 1)
        img = img.astype("float32")
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img)


def create_dataset(path, img_size=640, batch_size=8):
    """
    we need to define the img_size because the we need stack when use batch, but the image may have the different size
    """
    dataset = LoadImagesAndLabels(path, img_size=img_size,)
    loader = torch.utils.data.DataLoader
    dataloader = loader(
        dataset, batch_size=batch_size, collate_fn=LoadImagesAndLabels.collate_fn
    )
    return dataset, dataloader


class LoadImagesAndLabels(Dataset):
    """
    a subclass of torch.utils.data.Dataset with attribution:
    self.img_files(list): all paths of image files in the path dir.
    self.label_files(list): all paths of label files in the path dir.
    """

    def __init__(
        self, path, img_size=640,
    ):
        self.img_size = img_size

        f = []
        p = Path(path)
        if p.is_dir():
            f += glob.glob(str(p / "*.*"), recursive=True)
        else:
            pass
        self.img_files = sorted(f)
        # img = cv2.imread(self.img_files[0], 3)
        # cv2.imshow("test image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        f = []
        self.label_files = img2label_path(self.img_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """
        return:
            img(tensor) in 640 * 640 * 3 shape
            labels(tensor) in xywh normalized form
        """
        img, (h0, w0), (h, w) = self.load_image(index)
        new_shape = self.img_size
        img, ratio, pad = letterbox(img, new_shape)
        # cv2.imshow(f"image {index}, file name {self.img_files[index]}", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # x = self.labels[index]
        # convert pre load the label to the mode that everytime fetch the label, load the label
        x = self.load_label(index)
        nL = x.shape[0]
        labels_out = torch.zeros((nL, 6))
        label = []
        if nL:
            label = x.copy()
            h, w = (
                (img.shape[0] - 2 * pad[1]) / ratio[1],
                (img.shape[1] - 2 * pad[0]) / ratio[0],
            )
            # change the label according to the image transform(where x, y is the top left and bottom right corner absolutely)
            label[:, 1] = ratio[0] * (x[:, 1] - x[:, 3] / 2) * w + pad[0]
            label[:, 2] = ratio[1] * (x[:, 2] - x[:, 4] / 2) * h + pad[1]
            label[:, 3] = label[:, 1] + ratio[0] * label[:, 3] * w
            label[:, 4] = label[:, 2] + ratio[1] * label[:, 4] * h
        if nL:
            label[:, 1:5] = xyxy2xywh(label[:, 1:5])
            label[:, [2, 4]] /= img.shape[0]
            label[:, [1, 3]] /= img.shape[1]
            labels_out[:, 1:] = torch.from_numpy(label)

        # convert image
        img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # because the current dtype of img is int8, but the torch need float in order to calcualte gradient
        img = img.astype("float32")
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out

    def load_image(self, index):
        path = self.img_files[index]
        img = cv2.imread(path)
        assert img is not None, f"Image Not Found {path}"
        h0, w0 = img.shape[:2]
        r = self.img_size / max(h0, w0)
        # if r != 1:
        #     interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        #     img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]

    def load_label(self, index):
        path = self.label_files[index]
        with open(path, "r") as f:
            label = np.array(
                [x.split() for x in f.read().strip().splitlines()], dtype=np.float32
            )
        return label

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0)


def img2label_path(img_paths):
    sa, sb = (os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    return [
        x.replace(sa, sb).replace("." + x.split(".")[-1], ".txt") for x in img_paths
    ]


def letterbox(img, new_shape):
    # resize the img to the new_shape
    """
    return:
        image(np.array): resize image
        ratio(tuple): resized ratio(until now, there is no difference between w and h)
        pad(tuple): padding of w and
    """
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    shape_unpadding = round(shape[1] * r), round(shape[0] * r)
    dw, dh = new_shape[1] - shape_unpadding[0], new_shape[0] - shape_unpadding[1]
    img = cv2.resize(img, shape_unpadding, interpolation=cv2.INTER_LINEAR)
    dh /= 2
    dw /= 2
    # if dont do the round plus and minus, and dh or dw is odd, then the padding will larger the normal by 1
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    color = (144, 144, 144)
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, ratio, (dw, dh)
