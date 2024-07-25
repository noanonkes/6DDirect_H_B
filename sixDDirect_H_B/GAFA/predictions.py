import sys
sys.path.append('../')

from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

import torch
import argparse
import os, pickle

from tqdm import tqdm

from torch.utils.data import DataLoader

from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, check_file, check_dataset
from utils.datasets import GazeDataset
from models.experimental import attempt_load
from pytorch3d.transforms import rotation_6d_to_matrix


def main(root_dir="../GazeNet/data/preprocessed/", target_dir="../GazeNet/data/preprocessed/", filename="6DDirect_H_B_preds.pkl"):
    
    opt = parse_opt()

    train_exp_names = [
        "library/1026_3",
        "library/1028_2",
        "library/1028_5",
        "lab/1013_1",
        "lab/1014_1",
        "kitchen/1022_4",
        "kitchen/1015_4",
        "living_room/004",
        "living_room/005",
        "courtyard/004",
        "courtyard/005",
    ]

    test_exp_names = [
        "library/1029_2",
        "lab/1013_2",
        "kitchen/1022_2",
        "living_room/006",
        "courtyard/002",
        "courtyard/003",
        ]
    
    get_predictions(opt, train_exp_names, root_dir, target_dir, filename)
    get_predictions(opt, test_exp_names, root_dir, target_dir, filename)

def parse_opt():
    parser = argparse.ArgumentParser(prog="predictions.py")
    parser.add_argument("--data", type=str, default="data/gafa_coco_H_B.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", default="weights/gafa_ptAGORA_720_e50_t40_b128_b04_h06.pt")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=720, help="inference size (pixels)")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--scales", type=float, nargs="+", default=[1])
    parser.add_argument("--flips", type=int, nargs="+", default=[-1])
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")

    opt = parser.parse_args()
    opt.flips = [None if f == -1 else f for f in opt.flips]
    opt.data = check_file(opt.data)  # check file
    return opt
  
@torch.no_grad()
def run(data,
        dataset,
        model,
        device,
        weights=None,  # model.pt path(s)
        batch_size=16,  # batch size
        imgsz=720,  # inference size (pixels)
        single_cls=False,  # treat as single-class dataset
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.65,  # NMS IoU threshold
        scales=[1],
        flips=[None],
        half=True,  # use FP16 half-precision inference
        dataloader=None,
        workers=8,
        ):

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    dataloader = DataLoader(dataset,
                    batch_size=batch_size,
                    num_workers=nw,
                    pin_memory=True)
    
    num_rot = data["num_rot"]

    data_dict = []
    pbar = tqdm(dataloader, desc="Processing {} images".format(len(dataset)))
    for batch_i, item in enumerate(pbar):
        
        imgs = item["image"]
        indices = item["index"]
        head_dirs = item["head_dir"]
        body_dirs = item["body_dir"]
        gaze_dirs = item["gaze_dir"]

        imgs = imgs.to(device, non_blocking=True)
        imgs = imgs.half() if half else imgs.float()  # uint8 to fp16/32
        
        # Run model
        out, _ = model(imgs, augment=True, scales=scales, flips=flips)

        # Run NMS
        out = non_max_suppression(out, conf_thres, iou_thres, 
            multi_label=True, agnostic=single_cls, num_rot=num_rot)
        
        index = 0
        # Statistics per image
        for si, pred in enumerate(out):
            
            if len(pred) == 0 or pred.sum() == 0.:  # this image has NULL detections
                index += 1
                continue
            
            for p in pred.tolist():
                sixD = p[-num_rot:] #[0, 1] range
                c = p[5]
                score = p[4]

                data_dict.append({
                    "index": indices[si].tolist()[0],
                    "category_id": int(c + 1), # 1 for head, 2 for body
                    "score": round(score, 5),  # person score
                    "6D": [(x - 0.5) * 2 for x in sixD],
                    "gt_body_dir": body_dirs[si].tolist(),
                    "gt_head_dir": head_dirs[si].tolist(),
                    "gt_gaze_dir": gaze_dirs[si].tolist()
                })
                index += 1
    return data_dict

def get_predictions(opt, exp_names, root_dir, target_dir, filename="6DDirectHB_preds_6D_2.pkl"):

    device = select_device(opt.device, batch_size=opt.batch_size)

    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, s=gs)  # check image size

    data = check_dataset(opt.data)

    # Half
    opt.half &= device.type != "cpu"  # half precision only supported on CUDA
    if opt.half:
        model.half()
        
    # Configure
    model.eval()

    # Dataloader
    if device.type != "cpu":
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    opt.model = model
    opt.device = device
    opt.data = data

    for i, scene in enumerate(exp_names):
        print("Scene:", scene)

        for vd in os.listdir(os.path.join(root_dir, scene)):
            if os.path.exists(os.path.join(target_dir, scene, vd, filename)):
                print("Video already processed:", scene, vd)
                continue
            elif vd.endswith(".pkl"):
                continue
            
            video_path = os.path.join(root_dir, scene, vd)
            dataset = GazeDataset(video_path, imgsz)
            if len(dataset) == 0:
                print(f"{video_path} had invalid annotations.")
                continue
                
            opt.dataset = dataset
            v_dict = run(**vars(opt))

            with open(os.path.join(target_dir, video_path, filename), "wb") as f:
                pickle.dump(v_dict, f)


if __name__ == "__main__":
    main()