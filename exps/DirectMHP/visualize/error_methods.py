from utils.mae import sort_labels_by_image_id, calculate_bbox_iou

from tqdm import tqdm
import json
import numpy as np
import torch
from numpy import cos, sin, pi

from pytorch3d.transforms import matrix_to_euler_angles, rotation_6d_to_matrix, euler_angles_to_matrix
import torch.nn as nn

class GeodesicLoss(nn.Module):
    """ 
    Input: Rotations matrices of shape batch x 3 x 3
           Both matrices are orthogonal rotation matrices

    Output: Average theta between 0 to pi radians / pi.
    """
    def __init__(self, eps=1e-7, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    def forward(self, m1, m2):
        if len(m1.shape) < 3:
            m1 = m1[None]
        if len(m2.shape) < 3:
            m2 = m2[None]

        m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
        
        cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2        
        theta = torch.acos(torch.clamp(cos, -1+self.eps, 1-self.eps))
        
        if self.reduction == "mean":
            return torch.mean(theta) / np.pi
        return theta / np.pi
    
def get_predictions(gt_json_path, pd_json_path, matched_iou_threshold=0.65, score_threshold=0.7):
    gt_rot_data, pd_rot_data = [], []  # shapes of both should be N*6
    gt_euler_data, pd_euler_data = [], []  # shapes of both should be N*3
    pd_data_bbox, gt_data_bbox = [], []
    img_data, id_data = [], []

    gt_json = json.load(open(gt_json_path, "r"))
    pd_json = json.load(open(pd_json_path, "r"))
    
    gt_labels_list = gt_json["annotations"]
    pd_images_labels_dict = sort_labels_by_image_id(pd_json)
            
    for gt_label_dict in tqdm(gt_labels_list):  # matching for each GT label
        image_id = str(gt_label_dict["image_id"])
        gt_bbox = gt_label_dict["bbox"]
        [gp, gy, gr] = gt_label_dict["euler_angles"]
        if image_id not in pd_images_labels_dict:  # this image has no bboxes been detected
            continue

        pd_results = pd_images_labels_dict[image_id]
        max_iou, matched_index = 0, -1
        for i, pd_result in enumerate(pd_results):  # match predicted bboxes in target image
            score = pd_result["score"]
            if score < score_threshold:  # remove head bbox with low confidence
                continue
                
            pd_bbox = pd_result["bbox"]
            temp_iou = calculate_bbox_iou(pd_bbox, gt_bbox, format="xywh")
            if temp_iou > max_iou:
                max_iou = temp_iou
                matched_index = i
                
        if max_iou > matched_iou_threshold:
            pp = pd_results[matched_index]["pitch"]
            py = pd_results[matched_index]["yaw"]
            pr = pd_results[matched_index]["roll"]

            gt_R = euler_angles_to_matrix(torch.deg2rad(torch.tensor([gr, gy, gp])), "ZYX").tolist()
            pd_R = euler_angles_to_matrix(torch.deg2rad(torch.tensor([pr, py, pp])), "ZYX").tolist()

            gt_euler_data.append([gp, gy, gr])
            pd_euler_data.append([pp, py, pr])

            gt_rot_data.append(gt_R)
            pd_rot_data.append(pd_R)

            img_data.append(image_id)
            id_data.append(gt_label_dict["id"])
            pd_data_bbox.append(pd_results[matched_index]["bbox"])
            gt_data_bbox.append(gt_bbox)

      
    total_num = len(gt_labels_list)
    found = len(gt_euler_data)
    
    img_data = np.array(img_data)
    id_data = np.array(id_data)
    gt_euler_data = np.array(gt_euler_data)
    pd_euler_data = np.array(pd_euler_data)
    gt_rot_data = np.array(gt_rot_data)
    pd_rot_data = np.array(pd_rot_data)
    pd_data_bbox = np.array(pd_data_bbox)
    gt_data_bbox = np.array(gt_data_bbox)
    
    return total_num, found, img_data, id_data, pd_euler_data, gt_euler_data, pd_rot_data, gt_rot_data, pd_data_bbox, gt_data_bbox

def mae_calculate(gt_json_path, pd_json_path, index=None):
    """
    Input:
        gt_json_path: path to coco style annotations
        pd_json_path: path to json with predictions, written during validating
        index: return only pitch (0), yaw (1), roll(1)
    Output:
        total_num: total of heads in dataset
        found: number of heads found
        img_data: the IDs of the images
        geo_losses: the geodesic losses for each prediction
        pd_data: the predicted yaw, pitch and roll
        gt_data: the ground truth yaw, pitch and roll
        pd_data_bbox: the predicted bounding boxes
        gt_data_bbox: the ground truth bounding boxes
    """

    total_num, found, img_data, id_data, pd_euler_data, gt_euler_data, pd_rot_data, gt_rot_data, pd_bbox_data, gt_bbox_data = get_predictions(gt_json_path, pd_json_path)
    
    if found == 0:
        return None

    # [pitch, yaw, roll]
    error_list = np.abs(np.array(pd_euler_data) - np.array(gt_euler_data))
    error_list[:, 1] = np.min((error_list[:, 1], 360 - error_list[:, 1]), axis=0)  # yaw range is [-180,180]

    if index is not None:
        error_list = error_list[:, index]
    else:
        # MAE per data point
        error_list = np.mean(error_list, axis=1)

    output_dict = {"total_num": total_num,
                   "found": found,
                   "img_data": img_data,
                   "id_data": id_data,
                   "error_list": error_list,
                   "pd_euler_data": pd_euler_data,
                   "gt_euler_data": gt_euler_data,
                   "pd_rot_data": pd_rot_data,
                   "gt_rot_data": gt_rot_data,
                   "pd_bbox_data": pd_bbox_data,
                   "gt_bbox_data": gt_bbox_data
                   }
    
    return output_dict

def geodesic_calculate(gt_json_path, pd_json_path):
    """
    Input:
        gt_json_path: path to coco style annotations
        pd_json_path: path to json with predictions, written during validating
    Output:
        total_num: total of heads in dataset
        found: number of heads found
        img_data: the IDs of the images
        error_list: the absolute error for each prediction in yaw pitch and roll
        pd_data: the predicted 6D representation
        gt_data: the ground truth 6D representation
        pd_data_bbox: the predicted bounding boxes
        gt_data_bbox: the ground truth bounding boxes
    """

    total_num, found, img_data, id_data, pd_euler_data, gt_euler_data, pd_rot_data, gt_rot_data, pd_bbox_data, gt_bbox_data = get_predictions(gt_json_path, pd_json_path)

    if found == 0:
        return None

    geo_losses = np.array(GeodesicLoss(reduction=None)(torch.tensor(pd_rot_data), torch.tensor(gt_rot_data)))
    
    output_dict = {"total_num": total_num,
                   "found": found,
                   "img_data": img_data,
                   "id_data": id_data,
                   "error_list": geo_losses,
                   "pd_euler_data": pd_euler_data,
                   "gt_euler_data": gt_euler_data,
                   "pd_rot_data": pd_rot_data,
                   "gt_rot_data": gt_rot_data,
                   "pd_bbox_data": pd_bbox_data,
                   "gt_bbox_data": gt_bbox_data
                   }
    
    return output_dict
    
def projection2D_calculate(gt_json_path, pd_json_path):
    """
    Input:
        gt_json_path: path to coco style annotations
        pd_json_path: path to json with predictions, written during validating
    Output:
        total_num: total of heads in dataset
        found: number of heads found
        img_data: the IDs of the images
        manhattan_list: the absolute error for each prediction in manhattan distance
        pd_data: the predicted yaw, pitch and roll
        gt_data: the ground truth yaw, pitch and roll
        pd_data_bbox: the predicted bounding boxes
        gt_data_bbox: the ground truth bounding boxes
    """
    
    total_num, found, img_data, id_data, pd_euler_data, gt_euler_data, pd_rot_data, gt_rot_data, pd_bbox_data, gt_bbox_data = get_predictions(gt_json_path, pd_json_path)
    if found == 0:
        return None
        
    # PREDICTION
    pd_p = pd_euler_data[:, 0] * pi / 180
    pd_y = -(pd_euler_data[:, 1] * pi / 180)
    pd_r = pd_euler_data[:, 2] * pi / 180
    
    # X-Axis (pointing to right) drawn in red
    pd_x1 = (cos(pd_y) * cos(pd_r))
    pd_y1 = (cos(pd_p) * sin(pd_r) + cos(pd_r) * sin(pd_p) * sin(pd_y))
    
    # Y-Axis (pointing to down) drawn in green
    pd_x2 = (-cos(pd_y) * sin(pd_r))
    pd_y2 = (cos(pd_p) * cos(pd_r) - sin(pd_p) * sin(pd_y) * sin(pd_r))
    
    # Z-Axis (out of the screen) drawn in blue
    pd_x3 = (sin(pd_y))
    pd_y3 = (-cos(pd_y) * sin(pd_p))
    
    # GROUND TRUTH
    gt_p = gt_euler_data[:, 0] * pi / 180
    gt_y = -(gt_euler_data[:, 1] * pi / 180)
    gt_r = gt_euler_data[:, 2] * pi / 180
    
    # X-Axis (pointing to right) drawn in red
    gt_x1 = (cos(gt_y) * cos(gt_r))
    gt_y1 = (cos(gt_p) * sin(gt_r) + cos(gt_r) * sin(gt_p) * sin(gt_y))
    
    # Y-Axis (pointing to down) drawn in green
    gt_x2 = (-cos(gt_y) * sin(gt_r))
    gt_y2 = (cos(gt_p) * cos(gt_r) - sin(gt_p) * sin(gt_y) * sin(gt_r))
    
    # Z-Axis (out of the screen) drawn in blue
    gt_x3 = (sin(gt_y))
    gt_y3 = (-cos(gt_y) * sin(gt_p))
    
    p1 = abs(pd_x1 - gt_x1) + abs(pd_y1 - gt_y1)
    p2 = abs(pd_x2 - gt_x2) + abs(pd_y2 - gt_y2)
    p3 = abs(pd_x3 - gt_x3) + abs(pd_y3 - gt_y3)
    
    manhattan_list = (p1 + p2 + p3) / 3
    
    output_dict = {"total_num": total_num,
                   "found": found,
                   "img_data": img_data,
                   "id_data": id_data,
                   "error_list": manhattan_list,
                   "pd_euler_data": pd_euler_data,
                   "gt_euler_data": gt_euler_data,
                   "pd_rot_data": pd_rot_data,
                   "gt_rot_data": gt_rot_data,
                   "pd_bbox_data": pd_bbox_data,
                   "gt_bbox_data": gt_bbox_data
                   }
    
    return output_dict
