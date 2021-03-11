import time
import torch
from pathlib import Path
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import argparse
import yaml
import torch.nn as nn

from utils.loss import compute_loss
from utils.general import increment_path, get_latest_run
from utils.dataset import create_dataset
from utils.torch_utils import select_device
from models.yolo import Model, MyDataParallel


def train(opt, device, tb_writer=None):
    """
    read the data
    for every epoch, iterate the dataset and do the backpropagation
    """
    with open(opt.data, "r") as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    train_img_dir = data_dict["train_img_dir"]
    train_annotation_file = data_dict["train_annotation_file"]
    save_dir = Path(opt.save_dir)
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / "last.pt"
    best = wdir / "best.pt"

    # save run settings
    with open(save_dir / "opt.yaml", "w") as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    # img_size = opt.img_size
    batch_size = opt.batch_size
    dataset, dataloader = create_dataset(
        train_img_dir, train_annotation_file, batch_size=batch_size
    )
    print("load the image successfully!")
    model = Model()
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = MyDataParallel(model)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    start_epoch = 0
    best_loss = -1.0
    max_epoch = 300
    if opt.resume:
        ckpt = torch.load(opt.weights)
        best_loss = ckpt["best_loss"]
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    running_loss = 0.0
    for epoch in range(start_epoch, max_epoch):
        model.train()
        epoch_tick = time.time()
        epoch_loss = 0.0
        for i, (imgs, targets) in enumerate(dataloader):
            if i == 100:
                break
            preds = model(imgs.to(device))
            loss = compute_loss(preds, targets.to(device), model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            running_loss += loss.item()
            if i % 32 == 31:
                tb_writer.add_scalar(
                    "training loss", running_loss, epoch * len(dataloader) + i
                )
                running_loss = 0
            print(
                f"Now it is training on {i+1}/{len(dataloader)} batch in epoch {epoch + 1}",
                end="\r",
            )
        epoch_time = time.time() - epoch_tick
        remaining_epoch = max_epoch - epoch - 1
        print(
            f"{epoch+1} epoch is done, approximately {epoch_time * remaining_epoch} seconds remaining\n",
            end="\r",
        )
        if best_loss == -1 or epoch_loss < best_loss:
            best_loss = epoch_loss

        # save the checkpoint
        if not opt.nosave:
            save_dict = {
                "epoch": epoch,
                "best_loss": best_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(save_dict, last)
            if epoch_loss == best_loss:
                torch.save(save_dict, best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/coco.yaml")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--nosave", action="store_true", help="decide save the parameter or not"
    )
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="exp")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name is ok, which means there is only one exp directory in the runs/train directory",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="resume the most recent train",
    )  # if the --resume appear in the command but no argument follows, the const value(True) will be assign, if the --resume does not appear in the command, the default value(False) will be assign------the function of "nargs=?"
    opt = parser.parse_args()

    if opt.resume:
        ckpt = get_latest_run()
        with open(Path(ckpt).parent.parent / "opt.yaml", "r") as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
        opt.resume, opt.weights = True, ckpt
    else:
        opt.save_dir = increment_path(
            Path(opt.project) / opt.name, exist_ok=opt.exist_ok
        )
    device = select_device(opt.device, opt.batch_size)
    tb_writer = SummaryWriter(opt.save_dir)
    print(opt)
    train(opt, device=device, tb_writer=tb_writer)
