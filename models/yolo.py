import torch
import logging
import torch.nn as nn
import yaml
from models.common import Conv, Concat, Bottleneck
import logging

# logging.basicConfig(filename="find_error_of_detect_forward.log", level=logging.DEBUG)


class Detect(nn.Module):
    """
    get the input and output the corresponding dectection output which is some bs * anchor number * S * S * output per anchor tensor
    """

    stride = None

    def __init__(self, nc=80, anchors=(), input_channels=()):
        super(Detect, self).__init__()
        self.layer_num = len(input_channels)
        self.anchor_num = len(anchors[0]) // 2
        self.output_num = nc + 5
        a = torch.tensor(anchors).float().view(self.layer_num, -1, 2)
        self.register_buffer("anchors", a)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.layer_num, self.anchor_num, 1, 1, 2)
        )
        self.nc = nc
        self.grid = [torch.zeros(1)] * self.layer_num
        self.m = nn.ModuleList(
            Conv(x, self.output_num * self.anchor_num, 1) for x in input_channels
        )

    def forward(self, x):
        z = []
        for i in range(self.layer_num):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = (
                x[i]
                .view(bs, self.anchor_num, self.output_num, ny, nx)
                .permute(0, 1, 3, 4, 2)
            )
            if not self.training:
                if self.grid[i].shape[0:2] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (
                    y[..., 2:4] ** 2 * self.anchor_grid[i]
                )  # need to multiple the wh of anchor
                z.append(y.contiguous().view(bs, -1, self.output_num))
        return x

    @staticmethod
    def _make_grid(nx, ny):
        yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx))
        return torch.stack((yv, xv), dim=2)


class Model(nn.Module):
    """
    take the input h * w * 3 image and output the detection output
    """

    def __init__(self, cfg="models/yolo.yaml", ch=3, nc=None):
        super(Model, self).__init__()
        with open(cfg, "r") as f:
            self.yaml = yaml.load(f, Loader=yaml.FullLoader)
        self.model, self.save = parse_model(self.yaml, channels=[ch])

        m = self.model[-1]
        if isinstance(m, Detect):
            s = 128
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]
            )
            self.stride = m.stride
            m.anchors = torch.tensor(m.anchors, dtype=torch.float32)
            m.anchors /= m.stride.view(-1, 1)

    def forward(self, x):
        return self.forward_once(x)

    def forward_once(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )
            # logging.debug(f"the module {type(m).__name__}:")
            # shape_before = [y.shape for y in x] if isinstance(x, list) else x.shape
            # logging.debug(f"    input shape: {shape_before}")
            x = m(x)
            # shape_after = [y.shape for y in x] if isinstance(x, list) else x.shape
            # logging.debug(f"    output shape: {shape_after}\n")
            y.append(x if m.i in self.save else None)

        return x


def parse_model(yaml_dict, channels):
    nc = yaml_dict["nc"]
    anchors = yaml_dict["anchors"]
    layers = []
    save = []
    optput_channels_per_layer = [channels[0]]
    for i, (f, num, m, args) in enumerate(yaml_dict["backbone"] + yaml_dict["head"]):
        m = eval(m)
        for j, arg in enumerate(args):
            try:
                args[j] = eval(arg) if isinstance(arg, str) else arg
            except:
                pass
        if m in [Conv, Bottleneck]:
            c1, c2 = optput_channels_per_layer[f], args[0]
            args = [c1, c2, *args[1:]]
        elif m in [nn.Upsample]:
            c1 = c2 = optput_channels_per_layer[f]
        elif m in [Concat]:
            c1 = c2 = sum(optput_channels_per_layer[x if x == -1 else x + 1] for x in f)
            m_ = m(*args)
        elif m in [Detect]:
            c1 = c2 = [optput_channels_per_layer[x + 1] for x in f]
            args.append(c1)
        m_ = nn.Sequential(m(*args)) if num > 1 else m(*args)
        m_.f, m_.i = f, i
        layers.append(m_)
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        optput_channels_per_layer.append(c2)
        param_num = sum(x.numel() for x in m_.parameters())
        t = str(m)[8:-2].replace("__main__.", "")
        # with open(cfg_path, "a") as file:
        #     file.write("%3s%18s%10.0f  %-40s%-30s\n" % (i, f, param_num, t, args))
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    model = Model()
