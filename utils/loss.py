# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.general import clip_coords, xywh2xyxy
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


COCO_IMG_W = 640
COCO_IMG_H = 480
COCO_SMALL_T = 32
COCO_MEDIUM_T = 96


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        # self.sort_obj_iou = True  # TPH-YOLOv5
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class ComputeDomainLoss:
    # Compute domain losses
    def __init__(self, model):
        h = model.hyp  # hyperparameters

        # Define criteria
        BCE = nn.BCEWithLogitsLoss()
        self.BCE, self.hyp = BCE, h 

    def __call__(self, sp, tp):  # source predictions, target predictions
        device = sp[0].device

        losses = [torch.zeros(1, device=device) for _ in range(len(sp))]
        accuracies = [torch.zeros(1, device=device) for _ in range(len(sp))]
        targets = self.build_targets(sp, tp)  # targets

        # Losses and accuracies
        for i in range(len(sp)):
            losses[i] += self.BCE(torch.cat((sp[i], tp[i])), targets[i].to(device))
            # losses[i] *= self.hyp['domain']
            accuracies[i] = self.compute_accuracies(torch.cat((sp[i], tp[i])), targets[i].to(device))

        # bs = sp[0].shape[0] * 2  # batch size

        # return sum(losses) * bs, torch.cat(losses).detach(), torch.cat(accuracies).detach()
        return sum(losses)/3., torch.cat(losses).detach(), torch.cat(accuracies).detach()

    def build_targets(self, sp, tp):
        # Build targets for compute_domain_loss()
        t = []
        for i in range(len(sp)):
            t.append(torch.cat((torch.zeros(sp[i].shape), torch.ones(tp[i].shape))))
        return t

    def compute_accuracies(self, scores, ground_truth):
        # Compute accuracies for compute_domain_loss()
        predictions = (scores > 0.) # if > 0 it predicted source
        num_correct = (predictions == ground_truth).sum()
        num_samples = torch.prod(torch.tensor(predictions.shape))
        accuracy = float(num_correct)/float(num_samples)*100
        return torch.tensor([accuracy]).to(scores.device)


class ComputeAttentionLoss:
    # Compute attention losses
    def __init__(self, model):
        h = model.hyp  # hyperparameters

        # Define criteria
        BCE = nn.BCELoss()

        self.BCE, self.hyp = BCE, h 

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        for k in 'na', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, attn_maps, targets):  # objectness maps, targets
        device = targets.device
        lattn = torch.zeros(1, device=device)
        tattn = self.build_targets(attn_maps, targets)  # targets

        # Losses
        for i, attn_map in enumerate(attn_maps):
            lattn += self.BCE(attn_map, tattn[i])

        # lattn *= self.hyp['attn']
        # bs = ...  # batch size

        # return lattn * bs, lattn.detach()

        return lattn

    def build_targets(self, attn_maps, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tattns = []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        for i in range(self.nl):
            anchors = self.anchors[i]
            h, w = attn_maps[i].shape[1:]
            gain[2:6] = torch.tensor([[w, h, w, h]])  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter
                t = torch.unique(t[:, 2:6], dim=0)  # filter

            # Define
            attn_mask = torch.zeros((h, w))
            tbox = xywh2xyxy(t)
            clip_coords(tbox, (h, w))
            for xyxy in tbox:
                left = xyxy[0].round().int()
                top = xyxy[1].round().int()
                right = xyxy[2].round().int()
                bottom = xyxy[3].round().int()
                attn_mask[top:(bottom+1), left:(right+1)] = 1

            # Append
            tattns.append(attn_mask)

        return tattns

    def build_COCO_targets(self, attn_maps, targets):
        # Build binary mask for compute_loss(), input targets(image,class,x,y,w,h)
        small_t = (COCO_SMALL_T / COCO_IMG_W) * (COCO_SMALL_T / COCO_IMG_H)
        medium_t = (COCO_MEDIUM_T / COCO_IMG_W) * (COCO_MEDIUM_T / COCO_IMG_H)
        
        small_mask = torch.zeros(targets.shape[0], device=targets.device, dtype=torch.bool)
        medium_mask = torch.zeros(targets.shape[0], device=targets.device, dtype=torch.bool)
        large_mask = torch.zeros(targets.shape[0], device=targets.device, dtype=torch.bool)

        # Define
        box_area = targets[:, 4] * targets[:, 5]
        small_mask = small_mask.add((box_area < small_t))
        medium_mask = medium_mask.add((box_area > small_t) & (box_area < medium_t))
        large_mask = large_mask.add((box_area > medium_t))
        masks = [small_mask, medium_mask, large_mask]
        
        nt = targets.shape[0]  # number of targets
        tattns = []
        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain

        for i in range(self.nl):
            h, w = attn_maps[i].shape[1:]
            gain[2:6] = torch.tensor([[w, h, w, h]])  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                t = t[masks[i]]
            else:
                t = targets[0]

            # Define
            attn_mask = torch.zeros((h, w))
            tbox = xywh2xyxy(t)
            clip_coords(tbox, (h, w))
            for xyxy in tbox:
                left = xyxy[0].round().int()
                top = xyxy[1].round().int()
                right = xyxy[2].round().int()
                bottom = xyxy[3].round().int()
                attn_mask[top:bottom, left:right] = 1

            # Append
            tattns.append(attn_mask)

        return tattns
