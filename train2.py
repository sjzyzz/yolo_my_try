import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import argparse
import yaml

from utils.loss import compute_loss
from utils.dataset import create_dataset
from utils.torch_utils import select_device
from models.yolo import Model

writter = SummaryWriter("runs/train")


def train(opt, device):
    """
    read the data
    for every epoch, iterate the dataset and do the backpropagation
    """
    with open(opt.data, "f") as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    train_path = data_dict["train"]
    img_size = opt.img_size
    batch_size = opt.batch_size
    dataset, dataloader = create_dataset(train_path, img_size, batch_size)
    model = Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    max_epoch = 300
    for epoch in range(max_epoch):
        for i, (imgs, targets) in enumerate(dataloader):
            # write some image grids via tensorboard
            # img_gird = torchvision.utils.make_grid(imgs)
            # writter.add_image(f"16 images in the batch{i}", img_gird)

            preds = model(imgs.to(device))
            loss = compute_loss(preds, targets.to(device), model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f"In batch{i} of epoch {epoch}, the loss is {loss}")
            writter.add_scalar("training loss", loss, epoch * len(dataloader) + i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/coco128.yaml")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    opt = parser.parse_args()

    device = select_device(opt.device, opt.batch_size)
    train(opt, device=device)
