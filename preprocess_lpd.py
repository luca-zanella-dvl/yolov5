# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, clip_coords, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords)
from utils.plots import Annotator
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

VEHICLES = ['car', 'motorcycle', 'truck', 'bus']


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=1280,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        project=ROOT / 'runs/anonymize',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
    ):
    # Directories
    images_dir = Path(project) / 'images' / name
    labels_dir = Path(project) / 'labels' / name
    images_dir.mkdir(parents=True, exist_ok=True)  # make dir
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        h, w = im0s.shape[:2]
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=augment)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy()  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    veh_xyxy = torch.tensor(xyxy).view(1, 4).clone().view(-1).numpy().astype(np.int32)
                    c = int(cls)
                    label = names[c]
                    if label in VEHICLES:
                        with open(str(Path(source.replace('images', 'labels')) / p.name.replace('png', 'txt'))) as f:
                            labels = [l.rstrip('\r\n') for l in f.readlines()]
                            lp_xyxys = []
                            for label in labels:
                                lp_xywh = np.asarray(label.split()[1:], dtype=np.float32)
                                lp_xyxy = xywh2xyxy(lp_xywh * [[w, h, w, h]])  # xyxy pixels
                                lp_xyxy = lp_xyxy.reshape(1, 4).copy().reshape(-1).astype(np.int32)
                                if lp_xyxy[0] >= veh_xyxy[0] and lp_xyxy[1] >= veh_xyxy[1] and lp_xyxy[2] <= veh_xyxy[2] and lp_xyxy[3] <= veh_xyxy[3]:
                                    lp_xyxys.append(lp_xyxy) 
                        if lp_xyxys:
                            save_veh_box_and_labels(veh_xyxy, lp_xyxys, imc, str(names), file=images_dir / f'{p.stem}.png', BGR=True)   


def save_veh_box_and_labels(veh_xyxy, lp_xyxys, im, names, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    veh_xyxy = torch.tensor(veh_xyxy).view(-1, 4).detach().clone()
    veh_b = xyxy2xywh(veh_xyxy)  # boxes
    if square:
        veh_b[:, 2:] = veh_b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    # veh_b[:, 2:] = veh_b[:, 2:] * gain + pad  # box wh * gain + pad
    veh_xyxy = xywh2xyxy(veh_b).long()
    clip_coords(veh_xyxy, im.shape)
    veh_crop = im[int(veh_xyxy[0, 1]):int(veh_xyxy[0, 3]), int(veh_xyxy[0, 0]):int(veh_xyxy[0, 2]), ::(1 if BGR else -1)]
    # annotator = Annotator(np.ascontiguousarray(veh_crop), line_width=3, example=str(names))
    file.parent.mkdir(parents=True, exist_ok=True)  # make directory
    save_path = str(increment_path(file, sep='_').with_suffix('.png'))
    cv2.imwrite(save_path, veh_crop)

    for lp_xyxy in lp_xyxys:
        lp_xyxy = torch.tensor(lp_xyxy).view(-1, 4).detach().clone()
        lp_b = xyxy2xywh(lp_xyxy)  # boxes
        if square:
            lp_b[:, 2:] = lp_b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
        # lp_b[:, 2:] = lp_b[:, 2:] * gain + pad  # box wh * gain + pad
        lp_xyxy = xywh2xyxy(lp_b).long()
        clip_coords(lp_xyxy, im.shape)
        
        if save:
            with open(save_path.replace('images', 'labels').replace('png', 'txt'), 'a') as f:
                lp_xyxy[:, [0, 1]] -= veh_xyxy[:, [0, 1]]
                lp_xyxy[:, [2, 3]] -= veh_xyxy[:, [0, 1]]
                gn = torch.tensor(veh_crop.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                lp_xywh = (xyxy2xywh(lp_xyxy.detach().clone().view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (0, *lp_xywh)  # label format
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # h, w = veh_crop.shape[:2]
            # lp_xywh = np.asarray(lp_xywh, dtype=np.float32)
            # lp_xyxy = xywh2xyxy(lp_xywh * [[w, h, w, h]])  # xyxy pixels
            # annotator.box_label(lp_xyxy.reshape(1, 4).reshape(-1), names[0], color=colors(0, True))

        # im0 = annotator.result()
        # cv2.imwrite(save_path, im0)

    return veh_crop


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default=ROOT / 'runs/anonymize', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)