"""
The code is based on YOLOv5 https://github.com/ultralytics/yolov5
&
DirectMHP https://github.com/hnuzhy/DirectMHP/tree/main
"""

import argparse
import json
import os, os.path as osp
import sys
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add kapao/ to path

import numpy as np
import torch
from tqdm import tqdm
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import check_dataset, check_file, check_img_size, \
    non_max_suppression, scale_coords, set_logging, colorstr, xyxy2xywh
from utils.torch_utils import select_device, time_sync
import tempfile

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.pose_metrics import calculate_errors

@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=16,  # batch size
        imgsz=1280,  # inference size (pixels)
        task="val",  # train, val, test, speed or study
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        conf_thres_nms=0.001,  # NMS confidence threshold
        conf_thres=0.7, # Validation confidence threshold
        iou_thres_nms=0.65,  # NMS IoU threshold
        iou_thres=0.65,  # Validation IoU threshold
        scales=[1],
        flips=[None],
        rect=False,
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        compute_loss=None,
        pad=0,
        json_name="",
        limit=0
        ):

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Data
        data = check_dataset(data)  # check

    # Half
    half &= device.type != "cpu"  # half precision only supported on CUDA
    if half:
        model.half()
        
    # Configure
    model.eval()
    nc = int(data["nc"])  # number of classes

    # Dataloader
    if not training:
        if device.type != "cpu":
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        dataloader = create_dataloader(data[task], data["labels"], imgsz, batch_size, gs, 
            pad=pad, rect=rect, prefix=colorstr(f"{task}: "), limit=limit)[0]

    seen = 0
    names = {k: v for k, v in enumerate(model.names if hasattr(model, "names") else model.module.names)}
    mp, mr, map50, mAP, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(7, device=device)
    json_dump = []
    pbar = tqdm(dataloader, desc="Processing {} images".format(task))
    for batch_i, (imgs, targets, paths, shapes) in enumerate(pbar):
        t_ = time_sync()
        imgs = imgs.to(device, non_blocking=True)
        imgs = imgs.half() if half else imgs.float()  # uint8 to fp16/32
        imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        t = time_sync()
        t0 += t - t_

        # Run model
        out, train_out = model(imgs, augment=True, scales=scales, flips=flips)
        t1 += time_sync() - t

        # Compute loss
        if train_out:  # only computed if no scale / flipping
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls, geo, l2, n_bodies, n_heads

        t = time_sync()
        
        # Run NMS
        out = non_max_suppression(out, conf_thres_nms, iou_thres_nms, 
            multi_label=True, agnostic=single_cls, num_rot=data["num_rot"])
        
        t2 += time_sync() - t
        
        # Statistics per image
        for si, pred in enumerate(out):
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1
            
            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            
            if len(pred) == 0 or pred.sum() == 0.:  # this image has NULL detections
                continue
            
            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(imgs[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            box = xyxy2xywh(predn[:, :4])  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for p, b in zip(predn.tolist(), box.tolist()):
                # predictions (Array[N, 12]), x1, y1, x2, y2, conf, class, 6D rep
                sixD = p[-6:] 
                c = p[5]
                json_dump.append({
                    "image_id": image_id,
                    "category_id": int(c + 1), # 1 for head, 2 for body
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),  # person score,
                    "6D": [round((x - 0.5) * 2, 3) for x in sixD] # [0, 1] to [-1, 1] range
                })
                
    if not training:  # save json
        save_dir, weights_name = osp.split(weights)
        if not json_name:
            json_name = "{}_{}_c{}_i{}.json".format(
                task, osp.splitext(weights_name)[0],
                conf_thres, iou_thres)
        else:
            if not json_name.endswith(".json"):
                json_name += ".json"
        json_path = osp.join(save_dir, json_name)
    else:
        tmp = tempfile.NamedTemporaryFile(mode="w+b")
        json_path = tmp.name

    with open(json_path, "w") as f:
        json.dump(json_dump, f)

    error_list = []
    if len(json_dump) == 0:
        # error_list = [MAE, pitch_error, yaw_error, roll_error, geo, ae]
        error_list = [-999, -999, -999, -999, -999, -999]
        return (mp, mr, map50, mAP, *(loss.cpu() / len(dataloader)).tolist()), np.zeros(nc), t, error_list

    if task in ("train", "val"):
        annot = osp.join(data["path"], data["{}_annotations".format(task)])
        coco = COCO(annot)
        result = coco.loadRes(json_path)
        eval = COCOeval(coco, result, iouType="bbox")
        eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        mAP, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        
        # Whether to limit pitch and roll range to [-90, 90) or [-180, 180)
        if "AGORA" in data["path"] or "CMU" in data["path"]:
            limit_range = True
        else:
            limit_range = False

        limit_range=False

        # Calculate metrics for head
        total_num_head, left_num_head, geo_loss_head, pose_matrix_head, ae_head = \
            calculate_errors(annot, json_path, category_id=1, iou_thres=iou_thres, conf_thres=conf_thres, limit=limit_range)

        # Print stats for head
        [pitch_error, yaw_error, roll_error] = pose_matrix_head.tolist()    
        MAE = pose_matrix_head.mean().item()
        print("\nHead results:")
        print("\t Accuracy: {}/{};\n\t Mean Absolute Error: {:.4f};\n\t Errors [Pitch, Yaw, Roll]: {:.4f}, {:.4f}, {:.4f};\n\t Geodesic Distance: {:.4f};\n\t Mean Angular Error: {:.4f}".format(
            left_num_head, total_num_head, MAE, pitch_error, yaw_error, roll_error, geo_loss_head.item(), ae_head.item()))
        error_list = [MAE, pitch_error, yaw_error, roll_error, geo_loss_head, ae_head]

        if nc > 1:
            # Calculate metrics for body
            total_num_body, left_num_body, geo_loss_body, pose_matrix_body, ae_body = \
                calculate_errors(annot, json_path, category_id=2, iou_thres=iou_thres, conf_thres=conf_thres, limit=limit_range)
            # Print stats for body
            [pitch_error, yaw_error, roll_error] = pose_matrix_body.tolist()    
            MAE = pose_matrix_body.mean().item()
            print("\nBody results:")
            print("\t Accuracy: {}/{}\n\t MAE: {:.4f}; Errors [Pitch, Yaw, Roll]: {:.4f}, {:.4f}, {:.4f};\n\t Geodesic Distance: {:.4f};\n\t Mean Angular Error: {:.4f}\n".format(
                left_num_body, total_num_body, MAE, pitch_error, yaw_error, roll_error, geo_loss_body.item(), ae_body.item()))

    if training:
        tmp.close()

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image
    if not training and task != "test":
        os.rename(json_path, osp.splitext(json_path)[0] + "_ap{:.4f}.json".format(mAP))
        shape = (batch_size, 3, imgsz, imgsz)
        print(f"Speed: %.3fms pre-process, %.3fms inference, %.3fms NMS per image at shape {shape}" % t)

    model.float()  # for training
    # return (mp, mr, map50, mAP, *(loss.cpu() / len(dataloader)).tolist()), np.zeros(nc), t  # for compatibility with train
    return (mp, mr, map50, mAP, *(loss.cpu() / len(dataloader)).tolist()), np.zeros(nc), t, error_list


def parse_opt():
    parser = argparse.ArgumentParser(prog="val.py")
    parser.add_argument("--data", type=str, default="data/coco.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", default="yolov5s6.pt")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=1280, help="inference size (pixels)")
    parser.add_argument("--task", default="val", help="train, val, test")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--conf-thres-nms", type=float, default=0.001, help="NMS confidence threshold")
    parser.add_argument("--iou-thres-nms", type=float, default=0.65, help="NMS IoU threshold")
    parser.add_argument("--conf-thres", type=float, default=0.7, help="Validation confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.65, help="Validation IoU threshold")
    parser.add_argument("--scales", type=float, nargs="+", default=[1])
    parser.add_argument("--flips", type=int, nargs="+", default=[-1])
    parser.add_argument("--rect", action="store_true", help="rectangular input image")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--pad", type=int, default=0, help="padding for two-stage inference")
    parser.add_argument("--json-name", type=str, default="", help="optional name for saved json file")
    parser.add_argument("--limit", type=int, default=0, help="How many samples to use (for subset validation).")

    opt = parser.parse_args()
    opt.flips = [None if f == -1 else f for f in opt.flips]
    opt.data = check_file(opt.data)  # check file
    return opt


def main(opt):
    set_logging()
    print(colorstr("val: ") + ", ".join(f"{k}={v}" for k, v in vars(opt).items()))
    if opt.task in ("train", "val", "test"):  # run normally
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
