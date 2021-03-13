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

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/coco.yaml")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--dist-url", type=str, default="tcp://127.0.0.1:23456")
parser.add_argument(
    "--rank", default=0, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--world-size", default=1, type=int, help="number of node for distributed training",
)
parser.add_argument("--multiprocessing-distributed", action="store_true")
parser.add_argument(
    "--nosave", action="store_true", help="decide save the parameter or not"
)
parser.add_argument("--print-freq", type=int, default=10)
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


def main():
    args = parser.parse_args()
    world_size = args.world_size

    if args.resume:
        ckpt = get_latest_run()
        with open(Path(ckpt).parent.parent / "args.yaml", "r") as f:
            args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
        args.resume, args.weights, args.world_size = True, ckpt, world_size
    else:
        args.save_dir = increment_path(
            Path(args.project) / args.name, exist_ok=args.exist_ok
        )

    ngpus = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = args.world_size * ngpus
        print(args)
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args,))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(
        "nccl", init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
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
    max_epoch = 300
    if args.resume:
        loc = "cuda:{}".format(args.gpu)
        ckpt = torch.load(args.weights, map_location=loc)
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    for epoch in range(start_epoch, max_epoch):
        train(model, dataloader, compute_loss, optimizer, epoch, args)
        # save the checkpoint at the process 0
        if not args.nosave and args.rank % ngpus_per_node == 0:
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(save_dict, last)


def train(model, dataloader, criterion, optimizer, epoch, args):
    data_time = AverageMeter("Data", ":6.3f")
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(dataloader), [data_time, batch_time, losses], "Epoch: [{}]".format(epoch)
    )

    model.train()
    end = time.time()
    for i, (images, target) in enumerate(dataloader):
        data_time.update(time.time() - end)
        images = images.cuda(args.gpu)
        target = target.cuda(args.gpu)
        pred = model(images)
        loss = criterion(pred, target, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


class AverageMeter(object):
    def __init__(self, name, fmt):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_format_str(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        for meter in self.meters:
            entries.append(str(meter))
        print("\t".join(entries))

    def _get_batch_format_str(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
