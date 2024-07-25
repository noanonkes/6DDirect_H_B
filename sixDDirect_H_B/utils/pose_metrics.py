import json
import numpy as np
from tqdm import tqdm
import torch
from pytorch3d.transforms import rotation_6d_to_matrix

from utils.loss import GeodesicLoss
from utils.general import select_euler, inverse_rotate_zyx


def sort_labels_by_image_id(labels_list):
    images_labels_dict = {}
    for i, labels_dict in enumerate(labels_list):
        image_id = str(labels_dict["image_id"])
        if image_id in images_labels_dict.keys():
            images_labels_dict[image_id].append(labels_dict)
        else:
            images_labels_dict[image_id] = [labels_dict]
    return images_labels_dict


def sort_images_by_image_id(images_list):
    images_images_dict = {}
    for i, images_dict in enumerate(images_list):
        image_id = str(images_dict["id"])
        images_images_dict[image_id] = images_dict
    return images_images_dict


def calculate_bbox_iou(bboxA, bboxB, format="xyxy"):
    if format == "xywh":  # xy is in top-left, wh is size
        [Ax, Ay, Aw, Ah] = bboxA[0:4]
        [Ax0, Ay0, Ax1, Ay1] = [Ax, Ay, Ax+Aw, Ay+Ah]
        [Bx, By, Bw, Bh] = bboxB[0:4]
        [Bx0, By0, Bx1, By1] = [Bx, By, Bx+Bw, By+Bh]
    if format == "xyxy":
        [Ax0, Ay0, Ax1, Ay1] = bboxA[0:4]
        [Bx0, By0, Bx1, By1] = bboxB[0:4]
        
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    else:
        areaA = (Ax1 - Ax0)*(Ay1 - Ay0)
        areaB = (Bx1 - Bx0)*(By1 - By0)
        crossArea = W * H
        return crossArea/(areaA + areaB - crossArea)


def calculate_angular_error_from_matrix(R1, R2):
    """
    Calculate the mean angular error between two sets of rotation matrices.

    This function computes the angular error by comparing vectors transformed by two sets of rotation matrices (R1 and R2).
    It uses reference vectors pointing towards the camera to determine the angular error in degrees between the transformations.

    Arguments:
        R1 (torch.Tensor): A tensor of shape (N, 3, 3) representing the first set of N rotation matrices.
        R2 (torch.Tensor): A tensor of shape (N, 3, 3) representing the second set of N rotation matrices.

    Returns:
        float: The mean angular error in degrees.
    """

    # vectors looking at the camera
    ref_vecs = torch.zeros(len(R1), 3).unsqueeze(2)
    ref_vecs[:, -1, :] = -1.

    # R1 directions wrt camera
    R1_vecs = torch.bmm(R1, ref_vecs).squeeze()

    # R2 directions wrt camera
    R2_vecs = torch.bmm(R2, ref_vecs).squeeze()

    # mean angular error
    cos = torch.sum(R1_vecs * R2_vecs, dim=-1)
    cos[cos > 1.] = 1.
    cos[cos < -1.] = -1.
    rad = torch.acos(cos)
    mae = torch.rad2deg(torch.mean(rad))

    return mae


def calculate_angular_error_from_matrix_vector(R, vecs):
    """
    Calculate the mean angular error between reference vectors transformed by rotation matrices and given vectors.

    This function computes the angular error by comparing vectors transformed by rotation matrices (R) to given vectors (vecs). 
    It first ensures the input lengths match, then computes the angular error in degrees.

    Arguments:
        R (torch.Tensor): A tensor of shape (N, 3, 3) representing N rotation matrices.
        vecs (torch.Tensor): A tensor of shape (N, 3) representing N vectors.

    Returns:
        float: The mean angular error in degrees.

    Raises:
        ValueError: If the lengths of R and vecs do not match.
    """

    if len(R) != len(vecs):
        raise ValueError("Inputs are not same length", len(R), len(vecs))
       
    # vectors looking at the camera
    ref_vecs = torch.zeros(len(R), 3).unsqueeze(2)
    ref_vecs[:, -1, :] = -1.

    R_vecs = torch.bmm(R, ref_vecs).squeeze()

    # mean angular error
    cos = torch.sum(R_vecs * vecs, dim=-1)
    cos[cos > 1.] = 1.
    cos[cos < -1.] = -1.
    rad = torch.acos(cos)
    mae = torch.rad2deg(torch.mean(rad))

    return mae


def calculate_euler_error_from_matrix(R1, R2, limit_R2=False):
    """
    Calculate the mean absolute error between Euler angles from rotation matrices.

    Arguments:
        R1 (torch.Tensor): A tensor of shape (N, 3, 3) representing N rotation matrices.
        R2 (torch.Tensor): A tensor of shape (N, 3, 3) representing N rotation matrices.

    Returns:
        pose_matrix (torch.Tensor): The mean absolute error in degrees for pitch, yaw, and roll.
        false_status (list): Indices for which ground truth was not in [-90, 90) for pitch and roll.
    """
    r1_euler_data, r2_euler_data = [], []
    false_status = []
    for i, (r1, r2) in enumerate(zip(R1, R2)):
        _, [pitch, yaw, roll] = select_euler(
            np.rad2deg(inverse_rotate_zyx(r1)), # inverse rotation in order of ZYX
            pred=True
        )
        r1_euler_data.append([pitch, -yaw, -roll])
        

        status, [pitch, yaw, roll] = select_euler(
            np.rad2deg(inverse_rotate_zyx(r2)), # inverse rotation in order of ZYX
            pred=True if not limit_R2 else False
        )
        if not status:
            false_status.append(i)
            continue
        r2_euler_data.append([pitch, -yaw, -roll])


    # euler angles in [pitch, yaw, roll]
    r1_pyr = torch.tensor(r1_euler_data)
    r2_pyr = torch.tensor(r2_euler_data)

    # mean absolute error between euler angles
    error_list = torch.abs(r1_pyr - r2_pyr)
    if limit_R2:
        error_list[:, 1] = torch.min(error_list[:, 1], 360 - error_list[:, 1])  # yaw range is [-90,90]
    else:
        error_list = torch.min(error_list, 360 - error_list)  # yaw range is [-180,180]

    pose_matrix = error_list.mean(dim=0)

    return pose_matrix, false_status


def find_matches(gt_json_path, pd_json_path, category_id=1, iou_thres=0.65, conf_thres=0.7):
    """
    Find matches between predicted heads/bodies and ground truth.
    
    Arguments:
        gt_json_path (str): Path to the JSON file containing ground truth predictions.
        pd_json_path (str): Path to the JSON file containing model predictions (generated by calling val.py).
        category_id (int): Category ID for which errors are calculated (1 for head, 2 for body).
        iou_thres (float): Intersection over Union (IoU) threshold to determine a match between predicted and ground truth bounding boxes.
        conf_thres (float): Confidence threshold below which predictions are disregarded.

    Returns:
        total_num (int): Total number of heads/bodies.
        left_num (int): Number of heads/bodies were found.
        pd_data (list): List of predicted 6D rotations.
        gt_data (list): List of ground truth 6D rotations.
        gt_vecs (list): If vectors are in dataset, save these in list.
    """

    matched_iou_threshold = iou_thres  # our default nms_iou_thre is 0.65
    score_threshold = conf_thres

    gt_data, pd_data = [], []  # shapes of both should be N*6
    gt_vecs = [] # shape should be N*3

    gt_json = json.load(open(gt_json_path, "r"))
    pd_json = json.load(open(pd_json_path, "r"))
    
    gt_labels_list = gt_json["annotations"]
    pd_images_labels_dict = sort_labels_by_image_id(pd_json)
    
    total_num = len(gt_labels_list)
    other = 0
    vec = True if "vec" in gt_labels_list[0] else False

    for gt_label_dict in tqdm(gt_labels_list):  # matching for each GT label
        
        gt_c_id = gt_label_dict["category_id"] # 1 = head, 2 = body
        image_id = str(gt_label_dict["image_id"])

        if gt_c_id != category_id: # skip
            other += 1
            continue
        elif image_id not in pd_images_labels_dict:  # this image has no bboxes been detected
            continue
        
        gt_bbox = gt_label_dict["bbox"]
        gt_6D = gt_label_dict["6D"]

        pd_results = pd_images_labels_dict[image_id]
        max_iou, matched_index = 0, -1
        for i, pd_result in enumerate(pd_results):  # match predicted bboxes in target image

            score = pd_result["score"]
            pd_c_id = pd_result["category_id"]

            if score < score_threshold:  # remove head bbox with low confidence
                continue
            elif pd_c_id != category_id: # skip
                continue
                
            pd_bbox = pd_result["bbox"]
            temp_iou = calculate_bbox_iou(pd_bbox, gt_bbox, format="xywh")
            if temp_iou > max_iou:
                max_iou = temp_iou
                matched_index = i
                
        if max_iou > matched_iou_threshold:
            pd_6D = pd_results[matched_index]["6D"]

            pd_data.append(pd_6D)
            gt_data.append(gt_6D)

            if vec:
                gt_vecs.append(gt_label_dict["vec"])

    total_num -= other
    left_num = len(gt_data)
    return total_num, left_num, pd_data, gt_data, gt_vecs
            

def calculate_errors(gt_json_path, pd_json_path, category_id=1, iou_thres=0.65, conf_thres=0.7, limit=False):
    """
    Calculate four different metrics for evaluating predictions:

    1. Accuracy: Measures the proportion of heads/bodies correctly found.
    2. Geodesic Distance: Assesses the similarity of rotation matrices between ground truth (GT) and predictions (PD) for detected heads/bodies.
    3. Absolute Error: Evaluates the difference between predicted (PD) and ground truth (GT) Euler angles.
    4. Angular Error: Compares the predicted (PD) vectors to the ground truth (GT) vectors for alignment accuracy.

    Arguments:
        gt_json_path (str): Path to the JSON file containing ground truth predictions.
        pd_json_path (str): Path to the JSON file containing model predictions (generated by calling val.py).
        category_id (int): Category ID for which errors are calculated (1 for head, 2 for body).
        iou_thres (float): Intersection over Union (IoU) threshold to determine a match between predicted and ground truth bounding boxes.
        conf_thres (float): Confidence threshold below which predictions are disregarded.
        limit (bool): Specifies if the dataset has limited pitch, yaw, and roll ranges. 
                    - If True: pitch and roll are within [-90, 90) and yaw within [-180, 180).
                    - If False: all angles are within [-180, 180).
    """
    total_num, left_num, pd_data, gt_data, gt_vecs = find_matches(gt_json_path, pd_json_path, category_id, iou_thres, conf_thres)

    if left_num == 0:
        return total_num, left_num, torch.tensor(-999.), torch.tensor([-999., -999., -999.]), torch.tensor(-999.)
    
    gt_rots = rotation_6d_to_matrix(torch.tensor(gt_data))
    pd_rots = rotation_6d_to_matrix(torch.tensor(pd_data))
    gt_vecs = torch.tensor(gt_vecs)

    # mean absolute error
    pose_matrix, false_status = calculate_euler_error_from_matrix(pd_rots, gt_rots, limit) # order of these matters!
    
    # Some had invalid euler angles, we want to skip these for fair comparisons
    if len(false_status) > 0:
        # Indices to "throw away"
        indices = np.array(false_status)

        # Create a boolean mask where all indices are initially True
        mask = np.ones(len(gt_rots), dtype=bool)

        # Set the mask to False at the positions given in indices
        mask[indices] = False

        # Use the mask to filter the array
        gt_rots = gt_rots[mask]
        pd_rots = pd_rots[mask]
        gt_vecs = gt_vecs[mask]

        total_num -= len(false_status)

    # geodesic distance between rotation matrices in degrees
    geo_loss = torch.rad2deg(GeodesicLoss()(pd_rots, gt_rots))
    
    # mean angular error between vectors
    if len(gt_vecs) != 0:
        # vector are in dataset so take those
        mae = calculate_angular_error_from_matrix_vector(pd_rots, torch.tensor(gt_vecs))
    else:
        mae = calculate_angular_error_from_matrix(pd_rots, gt_rots)

    return total_num, left_num, geo_loss, pose_matrix, mae