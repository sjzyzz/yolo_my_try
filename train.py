import time
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torchvision
import argparse
import yaml
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from utils.loss import compute_loss
from utils.general import increment_path, get_latest_run
from utils.dataset import create_dataset
from utils.torch_utils import select_device
from models.yolo import Model, MyDataParallel


def train(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    print(f"process {gpu} start joining the process group")
    dist.init_process_group(
        "nccl", init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    print(f"process {gpu} join the process group!")
    with open(args.data, "r") as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    train_img_dir = data_dict["train_img_dir"]
    train_annotation_file = data_dict["train_annotation_file"]
    save_dir = Path(args.save_dir)
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / "last.pt"
    best = wdir / "best.pt"

    # save run settings
    if args.rank % ngpus_per_node == 0:
        with open(save_dir / "args.yaml", "w") as f:
            yaml.dump(vars(args), f, sort_keys=False)
    # img_size = args.img_size
    batch_size = args.batch_size
    dataset, dataloader = create_dataset(
        train_img_dir, train_annotation_file, batch_size=batch_size
    )
    model = Model()
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    args.batch_size = int(args.batch_size / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    start_epoch = 0
    best_loss = -1.0
    max_epoch = 300
    if args.resume:
        loc = "cuda:{}".format(args.gpu)
        ckpt = torch.load(args.weights, map_location=loc)
        best_loss = ckpt["best_loss"]
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    running_loss = 0.0
    for epoch in range(start_epoch, max_epoch):
        model.train()
        epoch_loss = 0.0
        for i, (imgs, targets) in enumerate(dataloader):
            if i > 50:
                break
            imgs = imgs.cuda(args.gpu)
            targets = targets.cuda(args.gpu)
            preds = model(imgs)
            loss = compute_loss(preds, targets, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            running_loss += loss.item()
            # if i % 32 == 31:
            #     tb_writer.add_scalar(
            #         "training loss", running_loss, epoch * len(dataloader) + i
            #     )
            #     running_loss = 0
            print(
                f"Now it is training on {i+1}/{len(dataloader)} batch in epoch {epoch + 1}",
                end="\r",
            )
        if best_loss == -1 or epoch_loss < best_loss:
            best_loss = epoch_loss

        # save the checkpoint at the process 0
        if not args.nosave and args.rank % ngpus_per_node == 0:
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dist-url", type=str, default="tcp://127.0.0.1:23456")
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of node for distributed training",
    )
    parser.add_argument("--multiprocessing-distributed", action="store_true")
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
    args = parser.parse_args()

    if args.resume:
        ckpt = get_latest_run()
        with open(Path(ckpt).parent.parent / "args.yaml", "r") as f:
            args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
        args.resume, args.weights = True, ckpt
    else:
        args.save_dir = increment_path(
            Path(args.project) / args.name, exist_ok=args.exist_ok
        )
    print(args)

    ngpus = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = args.world_size * ngpus
        mp.spawn(train, nprocs=ngpus, args=(ngpus, args,))
