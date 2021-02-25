import torch
import argparse
import yaml
import torch.nn as nn
import torch.optim as optim
import math
import torch.optim.lr_scheduler as lr_scheduler
import logging
import torch.cuda.amp as amp
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from warnings import warn

from utils.dataset import create_dataset
from models.yolo import Model
from utils.general import (
    check_file,
    check_img_size,
    increment_path,
    get_latest_run,
    set_logging,
    init_seed,
)
from utils.loss import compute_loss
from utils.torch_utils import (
    select_device,
    torch_distributed_zero_first,
    intersect_dicts,
)
from utils.google_utils import attemp_download

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None
    logger.info(
        "Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)"
    )


def train(hyp, opt, device, tb_writer=None, wandb=None):
    logger.info(f"Hyperparameters {hyp}")
    save_dir, epochs, batch_size, total_batch_size, weights, rank = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.total_batch_size,
        opt.weights,
        opt.global_rank,
    )

    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / "last.pt"
    best = wdir / "best.pt"
    results_file = save_dir / "result.txt"

    with open(save_dir / "hyp.yaml", "w") as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / "opt.yaml", "w") as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve
    cuda = device != "cpu"
    init_seed(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    # with torch_distributed_zero_first(rank):
    #     check_dataset(data_dict)
    train_path = data_dict["train"]
    test_path = data_dict["val"]
    nc, names = (
        (1, ["item"]) if opt.single_cls else (int(data_dict["nc"]), data_dict["names"])
    )
    assert len(names) == nc, "%g names found for nc=%g dataset in %s" % (
        len(names),
        nc,
        opt.data,
    )

    pretrained = weights.endswith(".pt")
    if pretrained:
        with torch_distributed_zero_first(rank):
            attemp_download(weights)
        ckpt = torch.load(weights, map_location=device)
        if hyp.get("anchors"):
            ckpt["model"].yaml["anchors"] = round(hyp["anchors"])
        model = Model(opt.cfg or ckpt["model"].yaml, ch=3, nc=nc).to(device)
        # exclude = ["anchor"] if opt.cfg or hyp.get("anchors") else []
        state_dict = ckpt["model"].float().state_dict()
        # state_dict = intersect_dicts(state_dict, model.state_dict, exclude=exclude)
        model.load_state_dict(state_dict, strict=False)
        # logger.info(
        #     "transfered  %g/%g items from %s"
        #     % (len(state_dict), len(model.state_dict()), weights)
        # )
    else:
        model = Model(opt.cfg, ch=3, nc=nc).to(device)

    # Freeze
    freeze = []
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            print("freezing %s" % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64
    accumulate = max(round(nbs / total_batch_size), 1)
    hyp["weight_decay"] *= total_batch_size * accumulate / nbs

    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "biases") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999))
    else:
        optimizer = optim.SGD(
            pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True
        )

    optimizer.add_param_group({"params": pg1, "weight_decay": hyp["weight_decay"]})
    optimizer.add_param_group({"params": pg2})
    # logger.info(
    #     "Optimizer group: %g .bias, %g conv.weight, %g other"
    #     % (len(pg2), len(pg1), len(pg0))
    # )
    del pg0, pg1, pg2

    lf = (
        lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2 * (1 - hyp["lrf"]))
        + hyp["lrf"]
    )
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Logging
    if wandb and wandb.run is None:
        opt.hyp = hyp
        wandb.run = wandb.init(
            config=opt,
            resume="allow",
            project="YOLOv3" if opt.project == "runs/train" else Path(opt.project.stem),
            name=save_dir.stem,
            id=ckpt.get("wandb_id") if "ckpt" in locals() else None,
        )
    # logger = {"wandb": wandb}

    # Resume
    start_epoch, best_epoch = 0, 0.0
    if pretrained:
        pass

    # Image size
    # DP mode]
    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    dataset, dataloader = create_dataset(train_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default="yolov3.pt", help="initial weights path"
    )
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument(
        "--data", type=str, default="data/coco128.yaml", help="data.yaml path"
    )
    parser.add_argument(
        "--hyp", type=str, default="data/hyp.scratch.yaml", help="hyperparameters path"
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--batch-size", type=int, default=16, help="total batch size for all GPUs"
    )
    parser.add_argument(
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="[train, test] image sizes",
    )
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="resume most recent training",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="only save final checkpoint"
    )
    parser.add_argument("--notset", action="store_true", help="only test final epoch")
    parser.add_argument(
        "--noautoanchor", action="store_true", help="disable autoanchor check"
    )
    parser.add_argument("--evolve", action="store_true", help="evolve hyperparameters")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument(
        "--cache-images", action="store_true", help="cache images for faster training"
    )
    parser.add_argument(
        "--image-weights",
        action="store_true",
        help="use weighted image selection for training",
    )
    parser.add_argument(
        "--device", default="cpu", help="cuda device, i.e. 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--multi-scale", action="store_true", help="vary img-size +/- 50%%"
    )
    parser.add_argument(
        "--single-cls", action="store_true", help="train as single class dataset"
    )
    parser.add_argument(
        "--adam", action="store_true", help="use torch.optim.Adam() optimizer"
    )
    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="use syncBatchNorm, only available in DDP mode",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="DDP parameter, do not modify"
    )
    parser.add_argument(
        "--log-imgs",
        type=int,
        default=16,
        help="number of images for W&B logging, max 100",
    )
    parser.add_argument("--project", default="runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/run is ok, do not increment",
    )
    opt = parser.parse_args()

    # set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1
    # set_logging(opt.global_rank)
    # if opt.global_rank in [-1, 0]:
    #     check_git_status()

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, yaml.FullLoader)
    # Resume
    if opt.resume:
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        assert os.path.isfile(ckpt), "ERROR: --resume checkpoint does not exist"
        with open(Path(ckpt).parent.parent / "opt.yaml") as f:
            opt = argparse.Namespace(**yaml.load(f, yaml.FullLoader))
        opt.cfg, opt.weights, opt.resume = "", ckpt, True
        logger.info(f"Resume training from {ckpt}")
    else:
        opt.data, opt.cfg, opt.hyp = (
            check_file(opt.data),
            check_file(opt.cfg),
            check_file(opt.hyp),
        )
        assert len(opt.cfg) or len(
            opt.weights
        ), "either --cfg or --weights must be specified"
        # for the situation where there is only one img-size argument is provided
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))
        # do not know why adding these code
        opt.name = "evolve" if opt.evolve else opt.name
        opt.save_dir = increment_path(
            Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve
        )

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        assert (
            opt.batch_size % opt.world_size == 0
        ), "--batch-size must be multiple of CUDA device count"
        opt.batch_size = otp.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, yaml.FullLoader)
        # do not know what's for
        if "box" not in hyp:
            warn(
                'Compatibility: %s missing "box" which was renamed from "giou" in %s'
                % (opt.hyp, "https://github.com/ultralytics/yolov5/pull/1120")
            )
            hyp["box"] = hyp.pop("giou")

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None
        if opt.global_rank in [-1, 0]:
            logger.info(
                f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/'
            )
            tb_writer = SummaryWriter(opt.save_dir)
        train(hyp, opt, device, tb_writer, wandb)

