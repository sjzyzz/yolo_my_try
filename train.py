import torch
from pathlib import Path
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import argparse
import yaml

from utils.loss import compute_loss
from utils.general import increment_path, get_latest_run
from utils.dataset import create_dataset
from utils.torch_utils import select_device
from models.yolo import Model


def train(opt, device, tb_writer=None):
    """
    read the data
    for every epoch, iterate the dataset and do the backpropagation
    """
    with open(opt.data, "r") as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    train_path = data_dict["train"]
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
    dataset, dataloader = create_dataset(train_path, batch_size=batch_size)
    model = Model().to(device)
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
    for epoch in range(start_epoch, max_epoch):
        model.train()
        print(f"Epoch {epoch}:")
        epoch_loss = 0.0
        for i, (imgs, targets) in enumerate(dataloader):
            # write some image grids via tensorboard
            # img_gird = torchvision.utils.make_grid(imgs)
            # writter.add_image(f"16 images in the batch{i}", img_gird)

            preds = model(imgs.to(device))
            loss = compute_loss(preds, targets.to(device), model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"    the loss in batch {i} is {loss}")
            epoch_loss += loss.item()
            # writter.add_scalar("training loss", loss, epoch * len(dataloader) + i)
        # save the checkpoint
        if not opt.nosave:
            if best_loss == -1 or epoch_loss < best_loss:
                best_loss = epoch_loss
            save_dict = {
                "epoch": epoch,
                "best_loss": best_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(save_dict, last)
            if best_loss == epoch_loss:
                torch.save(save_dict, best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/coco128.yaml")
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
    train(opt, device=device, tb_writer=tb_writer)
