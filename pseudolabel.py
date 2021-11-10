import os
import sys
import torch
import yaml
import cv2
import numpy as np
import shutil
import argparse
import math
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from models.experimental import attempt_load
from utils.datasets import LoadImages, create_dataloader
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, check_file, check_img_size, check_suffix, check_dataset, colorstr, check_yaml, xyxy2xywh
from utils.torch_utils import select_device


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = ROOT.relative_to(Path.cwd())  # relative


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--pseudo-thres', type=float, default=0.8, help='lower bound of class confidence')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
#     parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
#     parser.add_argument('--batch-size', type=int, default=32, help='batch size')
#     parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
#     parser.add_argument('--task', default='extra', help='val, test, speed or study')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
#     parser.add_argument('--pseudo-thres', type=float, default=0.4, help='lower bound of class confidence')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     opt = parser.parse_args()
#     opt.data = check_yaml(opt.data)  # check YAML
#     return opt


def pseudolabel_generation(pred, img1_shape, img0_shape, pseudo_threshold, boundary_error):
    # Convert model output to target format [class_id, x, y, w, h, conf]
    targets = [] 
    # Process predictions
    for i, det in enumerate(pred):  # per image
        gn = torch.tensor(img0_shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img1_shape, det[:, :4], img0_shape).round()
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if conf.item() > pseudo_threshold:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (int(cls.item()), *xywh, conf.item())
                        if (xywh[0] < 1) and (xywh[1] < 1) and (xywh[2] < 1) and (xywh[3] < 1):
                            targets.append(line)
                        else:
                            boundary_error += 1
                             
    return np.array(targets), boundary_error


def psuedolabel_generation(output, width, height, pseudo_threshold, boundary_error):
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    targets = []
    for i, o in enumerate(output):
        if o is not None:
            for pred in o:
                box = pred[:4]

                conf = pred[4]

                if conf.item() > pseudo_threshold:

                    cls = int(pred[5])
                    w = (box[2] - box[0]) / width
                    h = (box[3] - box[1]) / height
                    x = .5 * (box[2] + box[0]) / width
                    y = .5 * (box[3] + box[1]) / height

                    if (x.item() < 1) and (y.item() < 1) and (w.item() < 1) and (h.item() < 1):

                        targets.append([i, cls, x.item(), y.item(), w.item(), h.item(), conf.item()])

                    else:
                        boundary_error += 1

    return np.array(targets), boundary_error


@torch.no_grad()
def pseudolabel(
    weights=None,  # model.pt path(s)
    source=None,  # file/dir/URL/glob
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold    
    iou_thres=0.6,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    half=False,  # use FP16 half-precision inference
    pseudo_threshold=.8
    ):
    source = str(source)
    
    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)
        
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    count = 1
    uncount = 0
    boundary_error = 0
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
           
        # Inference 
        pred = model(img, augment=augment, visualize=visualize)[0]
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        plabels, boundary_error = pseudolabel_generation(pred, img.shape[2:], im0s.shape, pseudo_threshold, boundary_error)
            
        if len(plabels) > 0:
            labels = plabels[:, :-1]
            
            conf = int(pseudo_threshold * 100)
            file_name = path.replace('images/extra', f'labels/pseudo-conf{conf}').replace('jpg','txt').replace('JPG','txt').replace('png','txt').replace('PNG','txt')
            
            if (np.sum(np.isnan(labels)) == 0) and (np.sum(np.isinf(labels)) == 0):
                np.savetxt(file_name, labels, delimiter=' ',fmt=['%d','%4f','%4f','%4f','%4f'])
                image = Image.open(path)
                image.save(path.replace('extra', f'pseudo-conf{conf}'))
                count += 1
                image.close()
            else:
                uncount += 1
            

    print(f'Completed generating {count} pseudo labels.')
    print(f'Eliminated {uncount} images.')
    print(f'Boundary Error: {boundary_error} objects')

    
# @torch.no_grad()
# def pseudolabel(
#     data,
#     weights=None,
#     batch_size=16,
#     imgsz=640,
#     conf_thres=0.001,
#     iou_thres=0.6,  # for NMS
#     task='extra',
#     single_cls=False,
#     augment=False,
#     save_hybrid=True,  # save label+prediction hybrid results to *.txt
#     pseudo_threshold=.4,
#     half=False,
#     model=None,
#     dataloader=None
#     ):
#     # Initialize/load model and set device
#     training = model is not None
#     if training:  # called by train.py
#         device = next(model.parameters()).device  # get model device
#     else:  # called directly
#         device = select_device(opt.device, batch_size=batch_size)

#         # Load model
#         check_suffix(weights, '.pt')
#         model = attempt_load(weights, map_location=device)  # load FP32 model
#         gs = max(int(model.stride.max()), 32)  # grid size (max stride)
#         imgsz = check_img_size(imgsz, s=gs)  # check image size

#         # Data
#         data = check_dataset(data)  # check
#         # # Parse yaml
#         # path = Path(data.get('path') or '')  # optional 'path' default to '.'
#         # for k in 'extra', 'pseudo':
#         #     if data.get(k):  # prepend path
#         #         data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]

#     # Half
#     half &= device.type != 'cpu'  # half precision only supported on CUDA
#     model.half() if half else model.float()

#     # Configure
#     model.eval()
#     # with open(data) as f:
#     #     data = yaml.load(f, Loader=yaml.FullLoader)  # model dict

#     # Dataloader
#     if not training:
#         if device.type != 'cpu':
#             model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
#         task = 'extra' if task == 'extra' else 'val'  # path to val/test images
#        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=0.5, rect=True,
#                                        prefix=colorstr(f'{task}: '))[0]

#     count = 1
#     uncount = 0
#     boundary_error = 0

#     for batch_i, (img, targets, paths, shapes) in tqdm(enumerate(dataloader), desc="PseudoLabel", mininterval=0.01):
#         img = img.to(device, non_blocking=True)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         targets = targets.to(device)
#         nb, _, height, width = img.shape  # batch size, channels, height, width

#         # Run model
#         out, _ = model(img, augment=augment)  # inference and training outputs

#         # Run NMS
#         targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
#         lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
#         out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
#         # output = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres)
            
#         plabels, boundary_error = psuedolabel_generation(out, width, height, pseudo_threshold, boundary_error)

#         if len(plabels) > 0:
#             for i in plabels[:,0].astype('int'):

#                 idx = np.where(plabels[:,0] == i)[0]
#                 save_labels = plabels[idx]
#                 labels = save_labels[:,1:-1]

#                 file_name = paths[i].replace('images/extra', 'labels/pseudo').replace('jpg','txt').replace('JPG','txt').replace('png','txt').replace('PNG','txt')

#                 if (np.sum(np.isnan(labels)) == 0) and (np.sum(np.isinf(labels)) == 0):
#                     # np.savetxt(file_name, labels, delimiter=' ',fmt=['%d','%4f','%4f','%4f','%4f'])
#                     image = Image.open(paths[i])
#                     # image.save(paths[i].replace('extra', 'pseudo'))
#                     count += 1
#                     image.close()
#                 else:
#                     print(file_name)
#                     uncount += 1

#     print(f'Completed generating {count} pseudo labels.')
#     print(f'Eliminated {uncount} images.')
#     print(f'Boundary Error: {boundary_error} objects')

def main(opt):
    print(opt)
    print("Creating pseudo labels...")
    pseudolabel(
        opt.weights,
        opt.source,
        opt.imgsz,
        opt.conf_thres,
        opt.iou_thres,
        opt.max_det,
        opt.device,
        opt.classes,
        opt.agnostic_nms,
        opt.augment,
        opt.visualize,
        opt.half,
        opt.pseudo_thres,
    )
    
    
# def main(opt):
#     print(opt)
#     print("Creating pseudo labels...")
#     pseudolabel(
#         opt.data,
#         opt.weights,
#         opt.batch_size,
#         opt.imgsz,
#         opt.conf_thres,
#         opt.iou_thres,
#         opt.task,
#         opt.single_cls,
#         opt.augment,
#         opt.save_hybrid,
#         opt.pseudo_thres,
#         opt.half
#     )

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
