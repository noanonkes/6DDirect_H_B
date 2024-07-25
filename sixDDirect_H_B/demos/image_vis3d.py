import sys
sys.path.append('../')

from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())


import torch
import argparse
import yaml
import cv2
import numpy as np
from pytorch3d.transforms import rotation_6d_to_matrix

from demos.plotting import convert_rotation_bbox_to_6dof, plot_cuboid_Zaxis_by_euler_angles

from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords, non_max_suppression, select_euler, inverse_rotate_zyx
from utils.datasets import LoadImages
from models.experimental import attempt_load
from utils.renderer import Renderer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--img-path", default="test_imgs/", help="path to image or dir")
    parser.add_argument("--data", type=str, default="data/agora_coco.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--weights", default="weights/agora_m_1280_e300_t40_bs32_b04_h06.pt")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or cpu")
    parser.add_argument("--conf-thres", type=float, default=0.7, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.65, help="NMS IoU threshold")
    parser.add_argument("--scales", type=float, nargs="+", default=[1])
    parser.add_argument("--thickness", type=int, default=2, help="thickness of Euler angle lines")

    args = parser.parse_args()
    
    """ Create the renderer for 3D face/head visualization """   
    renderer = Renderer(
        vertices_path="pose_references/vertices_trans.npy", 
        triangles_path="pose_references/triangles.npy"
    )

    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict

    device = select_device(args.device, batch_size=1)
    print("Using device: {}".format(device))

    model = attempt_load(args.weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    dataset = LoadImages(args.img_path, img_size=imgsz, stride=stride, auto=True)
    dataset_iter = iter(dataset)
    
    for index in range(len(dataset)):
        
        (single_path, img, im0, _) = next(dataset_iter)
        if "_res" in single_path: continue
        print(index, single_path, "\n")
        
        img = torch.from_numpy(img).to(device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        out_ori = model(img, augment=True, scales=args.scales)[0]
        out = non_max_suppression(out_ori, args.conf_thres, args.iou_thres, num_rot=data["num_rot"])

        (h, w, c) = im0.shape
        global_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
        
        # predictions (Array[N, 12]), x1, y1, x2, y2, conf, class, a1-a6
        bboxes = scale_coords(img.shape[2:], out[0][:, :4], im0.shape[:2]).cpu().numpy()  # native-space pred
        scores = out[0][:, 4].cpu().numpy() 
        sixDs = out[0][:, 6:].cpu()   # N*6
        c_ids = out[0][:, 5].long() + 1

        print(f"Found {len(c_ids[c_ids==1])} heads.")
        print(f"Found {len(c_ids[c_ids==2])} bodies.")

        global_poses = []
        euler_angles = []
        classes = []
        looking_aways = []

        im = im0.copy()
        for i, [x1, y1, x2, y2] in enumerate(bboxes):
            c_name = "head" if c_ids[i] == 1 else "body"
            classes.append(c_name)

            # Get rotation R
            sixDs_i = (sixDs[i] - 0.5) * 2 # [0, 1] to [-1, 1]
            rot = rotation_6d_to_matrix(sixDs_i) # 3 * 3

            # Get Euler angles
            status, [pitch, yaw, roll] = select_euler(
            np.rad2deg(inverse_rotate_zyx(rot)), # inverse rotation in order of ZYX
            pred=True
            )

            yaw = -yaw
            roll = -roll

            euler_angles.append([pitch, yaw, roll])

            # Calculate if person is facing forward or not; for correct cube drawing
            looking_vec = rot @ torch.tensor([0., 0., -1.])
            looking_away = True if looking_vec[-1] > 0 else False
            looking_aways.append(looking_away)

            # Get global pose for renderer
            global_pose = convert_rotation_bbox_to_6dof(rot, [x1, y1, x2, y1], global_intrinsics)
            global_poses.append(global_pose)

            # Draw bounding boxes
            start = (int(x1), int(y1))
            end = (int(x2), int(y2))

            color = (255, 255, 255)
            im = cv2.rectangle(im, start, end, color=color, thickness=2)  

        for c_name, euler_angle, global_pose, looking_away, bbox in zip(classes, euler_angles, global_poses, looking_aways, bboxes):
            if c_name == "head":
                trans_vertices = renderer.transform_vertices(im, global_poses)
                im = renderer.render(im, trans_vertices, alpha=1.0)
            else:
                [x1, y1, x2, y2] = bbox
                im = plot_cuboid_Zaxis_by_euler_angles(im, yaw, pitch, roll, tdx=x1+(x2-x1)/2, tdy=y1+(y2-y1)/4,  
                    size=max((x2-x1), (y2-y1))*0.2, looking_away=looking_away)

        cv2.imwrite(f"{single_path[:-4]}_vis3d_res.jpg", im)
  