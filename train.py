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
from utils.general import increment_path, get_latest_run, box_iou, non_max_suppression
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
parser.add_argument("--iou-thres", type=float, default=0.5)
parser.add_argument("--conf-thres", type=float, default=0.001)


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
    val_img_dir = data_dict["val_img_dir"]
    val_annotation_file = data_dict["val_annotation_file"]
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
    _, train_loader = create_dataset(
        train_img_dir, train_annotation_file, batch_size=batch_size
    )
    _, val_loader = create_dataset(
        val_img_dir, val_annotation_file, batch_size=batch_size
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
        train(train_loader, model, compute_loss, optimizer, epoch, args)
        validate(val_loader, model, compute_loss, args)
        # save the checkpoint at the process 0
        if not args.nosave and args.rank % ngpus_per_node == 0:
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(save_dict, last)


def train(train_loader, model, criterion, optimizer, epoch, args):
    data_time = AverageMeter("Data", ":6.3f")
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(train_loader), [data_time, batch_time, losses], "Epoch: [{}]".format(epoch)
    )

    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
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


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    precious = AverageMeter("Precious", ":.4e")
    recall = AverageMeter("Recall", ":.4e")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, precious, recall], "Test: "
    )

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu)
            target = target.cuda(args.gpu)

            inf_out, train_out = model(images)
            loss = criterion(train_out, target, model)
            prec, rec = accuracy(inf_out, target, args)
            batch_time.update(time.time() - end)
            losses.update(loss.item(), images.size(0))
            precious.update(prec)
            recall.update(rec)

            if i % args.print_freq == 0:
                progress.display(i)


def accuracy(output, target, args):
    """
    Get the precision, recall of the total batch, both are in the naive scale and xyxy format
    Args:
        output: a tensor whose shape is num_image * 6, and 6 represent x, y, x, y, obj, cls
        target: a tensor whose shape is num_target * 6, and 5 represent img, cls, x, y, x, y
    """
    output = non_max_suppression(output, args)
    TP, FP = 0, 0
    # for every image
    for ii, pred in enumerate(output):
        labels = target[target[:, 0] == ii, 1:]
        cls_tensor = labels[:, 0]
        # for every class in the target
        for cls in torch.unique(cls_tensor, sorted=True):
            # find the corresponding prediction and target box and get the matrix
            ti = labels[:, 0] == cls
            sub_labels = labels[ti]
            pi = pred[:, -1] == cls
            sub_pred = pred[pi]
            if sub_pred.size(0) == 0:
                continue
            sub_pred = sub_pred[torch.argsort(sub_pred[:, 4], descending=True)]
            bbox_matrix = box_iou(sub_pred[:, :4], sub_labels[:, 1:])
            # for every prediction, find the largest iou corresponding target, if the iou is more than iou_threshold and the target has not been detected, TP plus one, else FP plus one
            seen = torch.zeros(len(sub_labels), dtype=bool)
            max_iou, target_indexs = torch.max(bbox_matrix, dim=1)
            for i, target_index in enumerate(target_indexs):
                if args.iou_thres < max_iou[i]:
                    if not seen[target_index]:
                        TP += 1
                        seen[target_index] = True
                    else:
                        FP += 1
                else:
                    FP += 1

    # calculate the matrics
    npos = target.size(0)
    if not TP + FP == 0:
        prec = TP / (TP + FP)
    else:
        prec = 0
    rec = TP / npos

    return prec, rec


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
