import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import argparse
import yaml

from utils.loss import compute_loss
from utils.dataset import create_dataset
from models.yolo import Model
from utils.loss import compute_loss

writter = SummaryWriter("runs/train")


def train(path="data/coco128.yaml", img_size=640, batch_size=16):
    """
    read the data
    for every epoch, iterate the dataset and do the backpropagation
    """
    dataset, dataloader = create_dataset(path, img_size, batch_size)
    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    max_epoch = 300
    for epoch in range(max_epoch):
        for i, (imgs, targets) in enumerate(dataloader):
            # write some image grids via tensorboard
            # img_gird = torchvision.utils.make_grid(imgs)
            # writter.add_image(f"16 images in the batch{i}", img_gird)

            preds = model(imgs)
            loss = compute_loss(preds, targets, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f"In batch{i} of epoch {epoch}, the loss is {loss}")
            writter.add_scalar("training loss", loss, epoch * len(dataloader) + i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/coco128.yaml")
    opt = parser.parse_args()
    with open(opt.data, "r") as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    train_path = data_dict["train"]
    train(train_path)
