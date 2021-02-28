import torch
import os
import logging
import torch.backends.cudnn as cudnn
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def select_device(device="", batch_size=None):
    s = f"Using torch {torch.__version__}"
    cpu = device.lower() == "cpu"
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        assert (
            torch.cuda.is_available()
        ), f"CUDA unavailable, invalid device {device} requested"

    cuda = torch.cuda.is_available and not cpu
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:
            assert (
                batch_size % n == 0
            ), f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * len(s)
        for i, d in enumerate(device.split(",") if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"
    else:
        s += "CPU"
    logger.info(f"{s}\n")
    return torch.device("cuda:0" if cuda else "cpu")


def init_torch_seeds(seed=0):
    torch.manual_seed(seed)
    if seed == 0:
        cudnn.benchmark, cudnn.deterministic = False, True
    else:
        cudnn.benchmark, cudnn.deterministic = True, False


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def intersect_dicts(da, db, exclude=()):
    return {
        k: v
        for k, v in da.items()
        if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape
    }
