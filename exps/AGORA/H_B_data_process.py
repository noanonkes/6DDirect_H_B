import torch
import numpy as np

import os
import copy
import json
import pickle
import shutil
import argparse
from tqdm import tqdm

from H_B_utils import (
    projectPoints,
    align_3d,
    reference_head,
    reference_body,
    get_sphere,
    select_euler,
    inverse_rotate_zyx,
)

from pytorch3d.transforms import matrix_to_rotation_6d

############################################################################################


coco_style_h_b_dict = {
    "info": {
        "description": "Head and body bounding boxes and their 6D rotations of AGORA Dataset",
        "url": "https://agora.is.tue.mpg.de/",
        "version": "1.0",
        "year": 2024,
        "contributor": "Noa Nonkes",
        "date_created": "2024/04/18",
    },
    "licences": [
        {
            "url": "http://creativecommons.org/licenses/by-nc/2.0",
            "name": "Attribution-NonCommercial License",
        }
    ],
    "images": [],
    "annotations": [],
    "categories": [
        {"supercategory": "person", "id": 1, "name": "head"},
        {"supercategory": "person", "id": 2, "name": "body"},
    ],
}

coco_style_dict_val = copy.deepcopy(coco_style_h_b_dict)
coco_style_dict_train = copy.deepcopy(coco_style_h_b_dict)

############################################################################################

# init transform params

Cref = np.mat([[1, 0, 0, 0.0], [0, -1, 0, 0], [0, 0, -1, 50], [0, 0, 0, 1]])

# Reference body from SMPL-X; front view T-posing body
Bref = reference_body()
Bref_h = np.ones((4, 127), dtype=np.float32)
Bref_h[0:3] = Bref

# Reference head from WHENet; front view face
Href, _ = reference_head(scale=1, pyr=(0.0, 0.0, 0.0))
Href_h = np.ones((4, 58), dtype=np.float32)
Href_h[0:3] = Href

# right shoulder (17), left shoulder (16), pelvis (0), (OPTIONAL:) neck (12), right hip (2), left hip (1)
kp_idx_body = np.asarray([17, 16, 0])

# keypoints for left eye corner, right lip corner, etc.
kp_idx_head = np.asarray(
    [0, 4, 9, 5, 28, 25, 22, 19, 18, 14, 37, 31, 40]
)  # 13 indexs of refered points in AGORA
kp_idx_model = np.asarray(
    [38, 34, 33, 29, 13, 17, 25, 21, 54, 50, 43, 39, 45]
)  # 13 indexs of refered points in FaceModel

# Sphere for head bounding box
sphere = []
for theta in range(0, 360, 10):
    for phi in range(0, 180, 10):
        sphere.append(get_sphere(theta, phi, 18))
sphere = np.asarray(sphere)
sphere = sphere + [0, 5, -5]
sphere = sphere.T

img_w, img_h = 1280, 720

############################################################################################


def get_rotation_6d(R, t, camR, camT):
    """
    Computes the 6D rotation representation and compound transformation matrix.

    Parameters:
    R (ndarray): Rotation matrix of shape (3, 3).
    t (ndarray): Translation vector of shape (3, 1).
    camR (ndarray): Camera rotation matrix of shape (3, 3).
    camT (ndarray): Camera translation vector of shape (3, 1).

    Returns:
    tuple: A tuple containing the 6D rotation representation and the compound transformation matrix.
    """
    # Create homogeneous transformation matrix for object
    Mc = np.zeros((4, 4))
    Mc[0:3, 0:3] = R
    Mc[0:3, 3:4] = t
    Mc[3, 3] = 1

    # Create homogeneous transformation matrix for camera
    Creal = np.zeros((4, 4))
    Creal[0:3, 0:3] = camR
    Creal[0:3, 3:4] = camT
    Creal[3, 3] = 1

    # Compute compound transformation matrix
    compound = Creal @ (Mc @ np.linalg.inv(Cref))

    # Get 6D representation
    sixD = matrix_to_rotation_6d(torch.tensor(compound[:3, :3]))

    return sixD, compound


def get_bbox(points2d, ratio=0.7, verbose=False):
    """
    Computes the bounding box for a set of 2D points and checks if it is within the frame.

    Parameters:
    points2d (ndarray): Array of 2D points of shape (N, 2).
    ratio (float): Ratio to determine if the bounding box is significantly out of frame.

    Returns:
    tuple: A tuple containing a boolean indicating if the bounding box is valid and the bounding box coordinates [x_min, y_min, w, h].
    """
    x_min, y_min = points2d.min(0)
    x_max, y_max = points2d.max(0)

    w, h = abs(x_max - x_min), abs(y_max - y_min)

    # Check if totally out of frame
    if (x_min < 0 and x_max < 0) or (y_min < 0 and y_max < 0) or (x_min > img_w and x_max > img_w) or (y_min > img_h and y_max > img_h):
        if verbose:
            print("Totally out of frame:", [x_min, x_max, y_min, y_max])
        return False, [0, 0, 0, 0]
    # Check if more than allowed ratio out of frame
    elif (x_min + w * ratio < 0 or x_max - w * ratio > img_w) or (y_min + h * ratio < 0 or y_max - h * ratio > img_h):
        if verbose:
            print(f"More than {ratio*100}% out of frame:", [x_min, x_max, y_min, y_max])
        return False, [0, 0, 0, 0]
    else:
        # Ensure points are within image dimensions
        valid_mask = ((0 <= points2d[:, 0]) & (points2d[:, 0] < img_w)) & (
            (0 <= points2d[:, 1]) & (points2d[:, 1] < img_h))

        x_min, y_min = points2d[valid_mask].min(0)
        x_max, y_max = points2d[valid_mask].max(0)

        x_min = int(max(0, x_min))
        x_max = int(min(x_max, img_w))
        y_min = int(max(0, y_min))
        y_max = int(min(y_max, img_h))

        w, h = x_max - x_min, y_max - y_min

    return True, [x_min, y_min, w, h]


def get_body_annotation(points3d, points2d, camR, camT, verbose=False):
    """
    Generates body annotations including bounding box, 6D rotation, and Euler angles.

    Parameters:
    points3d (ndarray): Array of 3D body points of shape (3, N).
    points2d (ndarray): Array of 2D body points of shape (N, 2).
    camR (ndarray): Camera rotation matrix of shape (3, 3).
    camT (ndarray): Camera translation vector of shape (3, 1).

    Returns:
    tuple: A tuple containing a boolean indicating if the annotation is valid and a dictionary with annotation details.
    """
    valid, body_bbox = get_bbox(points2d, ratio=0.7, verbose=verbose)

    # Check if body is in frame
    if valid:
        rotation_body, translation_body, _, _ = align_3d(
            np.mat(Bref_h[0:3, kp_idx_body]), np.mat(points3d[:, kp_idx_body]))

        sixD_body, compound = get_rotation_6d(
            rotation_body, translation_body, camR, camT)
        _, [pitch, yaw, roll] = select_euler(
            # inverse rotation in order of ZYX
            np.rad2deg(inverse_rotate_zyx(compound)),
            pred=True
        )

        # Adjust yaw and roll
        yaw = -yaw
        roll = -roll

        return True, {
            "bbox": body_bbox,
            "6D": sixD_body.tolist(),
            "euler_angles": [pitch, yaw, roll],
            "category": "body"}
    else:
        return False, {}


def get_head_annotation(points3d, camR, camT, camK, verbose=False):
    """
    Generates head annotations including bounding box, 6D rotation, and Euler angles.

    Parameters:
    points3d (ndarray): Array of 3D head points of shape (3, N).
    camR (ndarray): Camera rotation matrix of shape (3, 3).
    camT (ndarray): Camera translation vector of shape (3, 1).
    camK (ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
    tuple: A tuple containing a boolean indicating if the annotation is valid and a dictionary with annotation details.
    """
    rotation_head, translation_head, error, scale = align_3d(
        np.mat(Href_h[0:3, kp_idx_model]), np.mat(points3d[:, kp_idx_head]))

    sixD_head, compound = get_rotation_6d(
        rotation_head, translation_head, camR, camT)

    status, [pitch, yaw, roll] = select_euler(
        # inverse rotation in order of ZYX
        np.rad2deg(inverse_rotate_zyx(compound))
    )

    # Adjust yaw and roll
    yaw = -yaw
    roll = -roll

    # Check if head is within the correct yaw, pitch, roll range
    if status:
        # Aligning the generic head sphere with the real head model
        sphere_new = scale * rotation_head @ (sphere) + translation_head

        # Project the 3D points of the head model (sphere_new) into 2D image coordinates
        pt_helmet = projectPoints(
            sphere_new, camK, camR, camT, [0, 0, 0, 0, 0])

        # Get bounding box for the 2D projection
        valid, head_bbox = get_bbox(pt_helmet.T[:, :2], ratio=0.5, verbose=verbose)

        # Check if head is in frame
        if valid:
            return True, {
                "bbox": head_bbox,
                "6D": sixD_head.tolist(),
                "euler_angles": [pitch, yaw, roll],
                "category": "head"}

    return False, {}


def auto_labels_generating(filter_joints_list, verbose=False):
    """
    Generates annotations for body and head from a list of filtered joints from a single image.

    Parameters:
    filter_joints_list (list): List of filtered joints data, each entry containing [verts2d, occlusion, body3d, camR, camT, camK].

    Returns:
    tuple: A tuple containing the list of valid annotations and counts of lost bodies and heads.
    """
    valid_bbox_euler_list = []
    lost_bodies, lost_heads = 0, 0

    for [verts2d, occlusion, body3d, camR, camT, camK] in filter_joints_list:
        pair = []

        body3d = np.array(body3d).reshape((-1, 3)).transpose()
        face3d = body3d[:, 56 + 4 + 10 + 6:]  # Keypoints for the face

        valid, body_annotation = get_body_annotation(
            body3d, verts2d, camR, camT, verbose=verbose)

        # Check if body is in frame
        if valid:
            pair.append(body_annotation)
        else:
            lost_bodies += 1

        valid, head_annotation = get_head_annotation(face3d, camR, camT, camK, verbose=verbose)

        # Check if head is in frame and in correct angle range
        if valid:
            pair.append(head_annotation)
        else:
            lost_heads += 1

        valid_bbox_euler_list.append(pair)

    return valid_bbox_euler_list, lost_bodies, lost_heads


def parse_pkl_file(
    data,
    type,
    index,
    images_folder,
    images_save_folder,
    mapping_dict,
    debug,
):
    total_lost_bodies, total_lost_heads = 0, 0

    for idx in tqdm(range(len(data))):
        cur_image_path = data.iloc[idx].at["imgPath"]
        cur_valid_flag = data.iloc[idx].at["isValid"] # bool list, True or False
        cur_occlusion = data.iloc[idx].at["occlusion"]  # float list, value in range [0,100]
        cur_kid = data.iloc[idx].at["kid"]  # bool list, True or False
        cur_age = data.iloc[idx].at["age"]  # string list, e.g., "31-50", "50+"
        cur_verts_list = data.iloc[idx].at["gt_verts_2d"]  # numpy array list [10475 x 2]
        cur_joints_3d_list = data.iloc[idx].at["cam_j3d"]  # numpy array list, length is [127 x 3]
        cur_camR_list = data.iloc[idx].at["camR"]  # numpy array list, camera extrinsics R (3x3)
        cur_camT_list = data.iloc[idx].at["camT"]  # numpy array list, camera extrinsics T (3x1)
        cur_camK_list = data.iloc[idx].at["camK"]  # numpy array list, camera intrinsics K (3x3)

        if type == "train":
            image_path = os.path.join(
                images_folder + "_" + str(index), cur_image_path)
        else:
            image_path = os.path.join(images_folder, cur_image_path)

        enum = enumerate(
            zip(
                cur_valid_flag,
                cur_occlusion,
                cur_kid,
                cur_age,
                cur_verts_list,
                cur_joints_3d_list,
                cur_camR_list,
                cur_camT_list,
                cur_camK_list,
            )
        )

        filter_joints_list, remove_joints_list = [], []
        for ind, (isValid, occlusion, kid, age, verts, joints_3d, camR, camT, camK) in enum:

            body_3d = joints_3d

            if debug:
                print(index, cur_image_path, ind, isValid, occlusion, kid, age)

            if occlusion < 90:
                filter_joints_list.append(
                    [verts, occlusion, body_3d, camR, camT, camK])
            else:
                remove_joints_list.append(
                    [verts, occlusion, body_3d, camR, camT, camK])

        valid_bbox_euler_list, lost_bodies, lost_heads = auto_labels_generating(
            filter_joints_list, verbose=debug
        )
        total_lost_bodies += lost_bodies
        total_lost_heads += lost_heads

        """begin to process original labels of AGORA"""
        seq_name = cur_image_path.replace(
            "_1280x720.png", ""
        )  # seq_key_5_15_xxxxx_1280x720.png
        cur_frame = int(seq_name[-5:])

        seq_key = seq_name[:-6]
        if seq_key in mapping_dict[type]:
            mapping_dict[type][seq_key] += 1
        else:
            mapping_dict[type][seq_key] = 1
        seq_key_list = list(mapping_dict[type].keys())
        seq_ind = seq_key_list.index(seq_key) + 1

        # 1yyyyxxxxx for train, 2yyyyxxxxx for validation
        if type == "train":
            image_id = 1000000000 + seq_ind * 100000 + cur_frame
        if type == "validation":
            image_id = 2000000000 + seq_ind * 100000 + cur_frame

        """coco_style_sample"""
        image_dict = {
            "file_name": str(image_id) + ".jpg",
            "height": img_h,
            "width": img_w,
            "id": image_id,
            "seq_key": seq_key,
        }

        count = 0 # for unique id per annotations
        temp_annotations_list = []
        for ind_i, pair in enumerate(valid_bbox_euler_list):
            for labels in pair:
                temp_annotation = {
                    "bbox": labels["bbox"],  # please use the default "bbox" as key in cocoapi
                    "6D": labels["6D"],
                    "euler_angles": labels["euler_angles"],
                    "image_id": image_id,
                    "person_id": image_id * 100 + ind_i,  # we support that no image has more than 100 persons/poses
                    "id": image_id * 100 + count, # we support that no image has more max 50 persons with head+body anno
                    "category_id": 1 if labels["category"] == "head" else 2,
                    "iscrowd": 0,
                    "segmentation": [],  # This script is not for segmentation
                    "area": round(labels["bbox"][-1] * labels["bbox"][-2], 4),
                }
                temp_annotations_list.append(temp_annotation)
                count += 1

        if len(temp_annotations_list) != 0:
            if type == "train":
                coco_style_dict_train["images"].append(image_dict)
                coco_style_dict_train["annotations"] += temp_annotations_list
            if type == "validation":
                coco_style_dict_val["images"].append(image_dict)
                coco_style_dict_val["annotations"] += temp_annotations_list

            dst_img_path = os.path.join(
                images_save_folder, str(image_id) + ".jpg")
            shutil.copy(image_path, dst_img_path)
        else:
            continue  # after processing, this cur_frame with Null json annotation, skip it

        # finish one image/frame

    print("Total lost bodies:", total_lost_bodies)
    print("Total lost heads:", total_lost_heads)
    return mapping_dict


def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--load_pkl_flag", action="store_true", help="True will always load pkl files (taking much time)")
    parser.add_argument("--debug", action="store_true", help="True will print statistics of annotations.")
    parser.add_argument("--has_parsed", action="store_true", help="True will not operate parse_pkl_file() for saving time")
 
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":

    opt = parse_opt()

    # Please set debug = False and has_parsed = False when you first running this script.
    debug = opt.debug  # True or False, set as True will print statistics of annotations
    # True or False, set as True will not operate parse_pkl_file() for saving time
    has_parsed = opt.has_parsed
    # True or False, set as True will always load pkl files (taking much time)
    load_pkl_flag = opt.load_pkl_flag

    mapping_dict = {"train": {}, "validation": {}}

    # for type in ["validation", "train"]:
    for type in ["validation"]:
        # for type in ["train"]:
        images_folder = "./demo/images/%s" % (type)

        images_save_folder = "./H_B/images/%s" % (type)
        anno_save_folder = "./H_B/annotations/full_body_head_coco_style_%s.json" % (
            type
        )

        print("\nGenerating %s set ..." % (type))

        total_images, labeled_images = 0, 0
        if type == "validation":
            total_images = len(os.listdir(images_folder))

        index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for index in index_list:
            pkl_withjv_path = "./demo/Cam/%s_Cam/%s_%d_withjv.pkl" % (
                type, type, index)
            print(pkl_withjv_path)

            if not load_pkl_flag:
                continue

            with open(pkl_withjv_path, "rb") as f:
                data = pickle.load(f, encoding="iso-8859-1")
            
            if index == 0:
                print(list(data.keys()))
            labeled_images += len(data)

            if type == "train": total_images += len(os.listdir(images_folder+"_"+str(index)))

            if not has_parsed:
                mapping_dict = parse_pkl_file(
                    data,
                    type,
                    index,
                    images_folder,
                    images_save_folder,
                    mapping_dict,
                    debug,
                )

        if type == "train":
            if not has_parsed and not debug:
                print(len(mapping_dict["train"]), "\n", mapping_dict["train"])
                with open(anno_save_folder, "w") as dst_ann_file:
                    json.dump(coco_style_dict_train, dst_ann_file)
            else:
                with open(anno_save_folder, "r") as dst_ann_file:
                    coco_style_dict_train = json.load(dst_ann_file)
            print(
                "\ntrain: original images-->%d, lfabeled images-->%d, left images-->%d, left instances-->%d"
                % (
                    total_images,
                    labeled_images,
                    len(coco_style_dict_train["images"]),
                    len(coco_style_dict_train["annotations"]),
                )
            )


        if type == "validation":
            if not has_parsed and not debug:
                print(len(mapping_dict["validation"]),
                      "\n", mapping_dict["validation"])
                with open(anno_save_folder, "w") as dst_ann_file:
                    json.dump(coco_style_dict_val, dst_ann_file)
            else:
                with open(anno_save_folder, "r") as dst_ann_file:
                    coco_style_dict_val = json.load(dst_ann_file)
            print(
                "\nvalidation: original images-->%d, labeled images-->%d, left images-->%d, left instances-->%d"
                % (
                    total_images,
                    labeled_images,
                    len(coco_style_dict_val["images"]),
                    len(coco_style_dict_val["annotations"]),
                )
            )
