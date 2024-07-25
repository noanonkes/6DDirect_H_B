import sys
sys.path.append('../')

from pathlib import Path
FILE = Path(__file__).absolute()
print(FILE)
print(FILE.parents[0], FILE.parents[1])
sys.path.append(FILE.parents[1].as_posix())

import os, os.path as osp
import argparse
import cv2
import os

import torch
import numpy as np
from tqdm import tqdm
from pytorch3d.transforms import rotation_6d_to_matrix

from utils.torch_utils import select_device
from utils.renderer import Renderer

from visualize.plotting import convert_rotation_bbox_to_6dof, plot_3axis_Zaxis
from visualize.error_methods import mae_calculate, geodesic_calculate, projection2D_calculate

from PIL import Image, ImageDraw, ImageFont

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def get_image(img, sixD, euler, x1, y1, x2, y2):
    imz = img.copy()
    
    rot = rotation_6d_to_matrix(torch.tensor(sixD)) # 3 * 3
    pitch, yaw, roll = euler
    
    (h, w, _) = imz.shape
        
    global_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])

    global_pose = convert_rotation_bbox_to_6dof(rot, [x1, y1, x2, y2], global_intrinsics)
    global_poses = [global_pose]

    trans_vertices = renderer.transform_vertices(imz, global_poses)
    imz = renderer.render(imz, trans_vertices, alpha=1.0)

    imz = plot_3axis_Zaxis(imz, yaw, pitch, roll, tdx=(x1+x2)/2, tdy=(y1+y2)/2, 
    size=max(y2-y1, x2-x1)*0.8, thickness=2)

    return imz

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--img-path", default="../AGORA/H_B/images/", help="path to image dir")
    parser.add_argument("--save-dir", type=str, help="path to save images")
    parser.add_argument("--pd-json", type=str)
    parser.add_argument("--gt-path", type=str, default="../AGORA/H_B/annotations/full_head_coco_style_validation.json")
    parser.add_argument("--task", type=str, default="validation")

    parser.add_argument("--method", type=str, default="mae", choices=["mae", "geo", "2d"])
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--error-thres", type=float, default=30.)
    parser.add_argument("--top-k", type=int, default=None)

    parser.add_argument("--weights", default="weights/agora_m_1280_e300_t40_lw010_bs32_0.4_0.6.pt")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or cpu")

    args = parser.parse_args()

    """ Create the renderer for 3D face/head visualization """   
    renderer = Renderer(
        vertices_path="pose_references/vertices_trans.npy", 
        triangles_path="pose_references/triangles.npy"
    )

    device = select_device(0, batch_size=1)
    print("Using device: {}".format(device))

    if "AGORA" in args.img_path:
        dset = "agora"
    else:
        dset = "cmu"

    save_dir, weights_name = osp.split(args.weights)
    pd_json = osp.join(save_dir, args.pd_json)
    gt_json = args.gt_path
    
    kwargs = {"gt_json_path": gt_json, "pd_json_path": pd_json}
    if args.method == "mae":
        method = mae_calculate
        kwargs["index"] = args.index
    elif args.method == "geo":
        method = geodesic_calculate
    elif args.method == "2d":
        method = projection2D_calculate
    else:
        raise ValueError("Unknown method.")

    output_dict = method(**kwargs)
    
    if output_dict is None:
        raise ValueError("No heads detected.")

    if args.top_k is not None:
        if args.top_k < 0:
            # the worst k
            mask = np.argsort(output_dict["error_list"])[args.top_k:]
        else:
            # the best k
            mask = np.argsort(output_dict["error_list"])[:args.top_k]
    else:
        mask = output_dict["error_list"] > args.error_thres
    
    img_data = output_dict["img_data"][mask]
    id_data = output_dict["id_data"][mask]
    errors = output_dict["error_list"][mask]
    pd_euler_data = output_dict["pd_euler_data"][mask]
    gt_euler_data = output_dict["gt_euler_data"][mask]
    pd_6D_data = output_dict["pd_6D_data"][mask]
    gt_6D_data = output_dict["gt_6D_data"][mask]
    pd_data_bbox = output_dict["pd_bbox_data"][mask]
    gt_data_bbox = output_dict["gt_bbox_data"][mask]

    pbar = tqdm(zip(img_data, errors,
                    pd_euler_data, gt_euler_data,
                    pd_6D_data, gt_6D_data,
                    pd_data_bbox,
                    gt_data_bbox))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for i, (img, error, pd_euler, gt_euler, pd_6D, gt_6D, pd_bbox, gt_bbox) in enumerate(pbar):

        im0 = cv2.imread(str(osp.join(args.img_path, args.task, str(img) + ".jpg")))
        
        h_img, w_img, c_img = im0.shape

        # Ground truth bbox in x and y
        x1, y1, x2, y2 = gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]
        w, h = x2 - x1, y2 - y1

        # Bigger bbox for saving to show little bit of surroundings
        EXTRA = 100
        _x1 = max(0, int(x1-EXTRA))
        _y1 = max(0, int(y1-EXTRA))
        _x2 = min(w_img, int(x2+EXTRA))
        _y2 = min(h_img, int(y2+EXTRA))

        box_w = _x2 - _x1
        box_h = _y2 - _y1

        # Empty image for original, ground truth, prediction, and text
        concat_img = np.zeros((2 * box_h, 2 * box_w, c_img))
    
        # Original image, left up
        concat_img[0:box_h, 0:box_w, :] = im0[_y1:_y2, _x1:_x2, :]

        # Prediction image, left down
        x1, y1, x2, y2 = pd_bbox[0], pd_bbox[1], pd_bbox[0] + pd_bbox[2], pd_bbox[1] + pd_bbox[3]
        img_pd = get_image(im0, pd_6D, pd_euler, x1, y1, x2, y2)
        concat_img[box_h:, :box_w, :] = img_pd[_y1:_y2, _x1:_x2, :]
        concat_img = cv2.putText(concat_img, "PD", (20, int(box_h+25)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Ground truth image, right up
        img_gt = get_image(im0, gt_6D, gt_euler, x1, y1, x2, y2)
        concat_img[0:box_h, box_w:, :] = img_gt[_y1:_y2, _x1:_x2, :]
        concat_img = cv2.putText(concat_img, "GT", (int(box_w+20), 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
       
        # Text, right down        
        im = Image.new("RGB", (box_w, box_h), "#fff")
        box = ((5, 5, box_w-5, box_h-5))
        draw = ImageDraw.Draw(im)
        draw.rectangle(box)

        text = f"     Pitch,  yaw,  roll\nGT: {[round(x, 2) for x in gt_euler]}\nPD: {[round(x, 2) for x in pd_euler]}\n\n     ERROR {error:.2f}"
        print("Text:\n", text, "\n")
        font_size = 100
        size = None
        while (size is None or size[0] > box[2] - box[0] or size[1] > box[3] - box[1]) and font_size > 0:
            font = ImageFont.truetype("Arial.ttf", font_size)
            size = font.getsize_multiline(text)
            font_size -= 1
        draw.multiline_text((box[0], box[1]), text, "#000", font)

        concat_img[box_h:, box_w:] = np.array(im) 

        cv2.imwrite(f"{args.save_dir}/{i}_{img}" + ".jpg", concat_img)
