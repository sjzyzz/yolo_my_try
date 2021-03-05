import torch
import torch.nn as nn
from utils.general import bbox_iou
import cv2
from utils.plot import plot_one_box
import random
import numpy as np


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


def compute_loss(preds, targets, model):
    # targets in normalized xywh form
    lbox, lcls, lobj = (
        torch.tensor(0.0).to(targets.device),
        torch.tensor(0.0).to(targets.device),
        torch.tensor(0.0).to(targets.device),
    )
    tcls, tbox, indices, anchors = build_targets(preds, targets, model)

    cp, cn = smooth_BCE()
    BCEcls = nn.BCEWithLogitsLoss()
    BCEobj = nn.BCEWithLogitsLoss()
    balance = [4.0, 1.0, 0.4] if len(preds) == 3 else [4.0, 1.0, 0.4, 0.1]

    for i, preds_i in enumerate(preds):
        img, a, gj, gi = indices[i]
        n = img.shape[0]

        tobj = torch.zeros_like(preds_i[..., 0]).to(targets.device)
        if n:
            pred_sub = preds_i[img, a, gj, gi]

            # bounding box
            pxy = pred_sub[:, :2].sigmoid()
            pwh = (pred_sub[:, 2:4].sigmoid()) ** 2 * anchors[i]

            pbox = torch.cat((pxy, pwh), 1)
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False)
            lbox += (1 - iou).mean()

            # classification
            t = torch.full_like(pred_sub[:, 5:], cn)
            t[range(n), tcls[i]] = cp
            # don't make sense here
            lcls += BCEcls(pred_sub[:, 5:], t)

            # objectness ???
            tobj[img, a, gj, gi] = 1
        # no question here because the logits only work on pred, doesn't work on targets
        lobj += balance[i] * BCEobj(preds_i[..., 4], tobj)

    loss = lbox + lcls + lobj

    return loss


def build_targets(preds, targets, model):
    """
    build the target for down-stream work
    args:
        targets: in xywh format
    return:
        each output has one part corresponding to the each detect output layer
        tcls(list(np.array)): available targets' class of each layer 
        tbox(list(np.array)): available targets' box of each layer
        indices(list(np.array)): the other useful part of targets(image, anchor, gridy, gridx)
    """
    tcls = []
    tbox = []
    indices = []
    anch = []
    detect = model.model[-1]
    anchor_num = detect.anchor_num
    target_num = targets.shape[0]
    anchor_index = (
        torch.arange(anchor_num)
        .view(anchor_num, 1)
        .repeat(1, target_num)
        .to(targets.device)
    )
    targets = torch.cat(
        (targets.repeat(anchor_num, 1, 1), anchor_index[:, :, None]), -1
    )
    gain = torch.ones(7).to(targets.device)
    for i in range(detect.layer_num):
        anchors = detect.anchors[i].view(-1, 2).to(targets.device)
        gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]  # nx, ny, nx, ny
        t = targets * gain
        if target_num:
            r = t[:, :, 4:6] / anchors[:, None, :]
            # j = torch.max(r, 1.0 / r).max(2)[0] < model.hyp["anchor_t"]
            j = torch.max(r, 1.0 / r).max(2)[0] < 4
            t = t[j]

        else:
            t = targets[0]
        img, c = t[:, :2].T.long()
        gxy = t[:, 2:4]
        gwh = t[:, 4:6]
        a = t[:, 6].long()
        gij = gxy.long()
        offset = gxy - gij
        gi, gj = gij.T
        indices.append((img, a, gj, gi))
        tcls.append(c)
        tbox.append(torch.cat((offset, gwh), 1))
        anch.append(anchors[a])
    return tcls, tbox, indices, anch
