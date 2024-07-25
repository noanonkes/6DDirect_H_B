import os
import json
import cv2
import shutil
import subprocess
import numpy as np
import torch
from pytorch3d.transforms import matrix_to_rotation_6d
from tqdm import tqdm
import tempfile

from H_B_utils import projectPoints
from H_B_utils import align_3d
from H_B_utils import reference_head, reference_body
from H_B_utils import get_sphere
from H_B_utils import select_euler
from H_B_utils import inverse_rotate_zyx


############################################################################################

coco_style_hpe_dict = {
    "info": {
        "description": "6D rotations, euler angles and bounding boxes of head and body of CMU Panoptic Studio Dataset",
        "url": "http://domedb.perception.cs.cmu.edu/",
        "version": "1.0",
        "year": 2024,
        "contributor": "Noa Nonkes",
        "date_created": "2024/04/03",
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

# body center, lshoulder, rshoulder
kp_idx_body = np.asarray([2, 3, 9])
kp_idx_body_model = np.asarray([0, 16, 17])

# keypoints for left eye corner, right lip corner, etc.
kp_idx_head = np.asarray(
    [17, 21, 26, 22, 45, 42, 39, 36, 35, 31, 54, 48, 57, 8]
)  # 14 indexs of refered points in CMUPanoptic
kp_idx_head_model = np.asarray(
    [38, 34, 33, 29, 13, 17, 25, 21, 54, 50, 43, 39, 45, 6]
)  # 14 indexs of refered points in FaceModel

# Sphere for head bounding box
sphere = []
for theta in range(0, 360, 10):
    for phi in range(0, 180, 10):
        sphere.append(get_sphere(theta, phi, 18))
sphere = np.asarray(sphere)
sphere = sphere + [0, 5, -5]
sphere = sphere.T

############################################################################################


def parsing_camera_calibration_params(hd_camera_path):
    with open(hd_camera_path) as cfile:
        calib = json.load(cfile)

    # Cameras are identified by a tuple of (panel#,node#)
    cameras = {(cam["panel"], cam["node"]): cam for cam in calib["cameras"]}

    # Convert data into numpy arrays for convenience
    for k, cam in cameras.items():
        cam["K"] = np.matrix(cam["K"])
        cam["distCoef"] = np.array(cam["distCoef"])
        cam["R"] = np.matrix(cam["R"])
        cam["t"] = np.array(cam["t"]).reshape((3, 1))

    return cameras


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


def get_bbox(points2d, img_h, img_w, ratio=0.7, verbose=False):
    """
    Computes the bounding box for a set of 2D points and checks if it is within the frame.

    Parameters:
    points2d (ndarray): Array of 2D points of shape (N, 2).
    img_h, img_w (int): Sizes of the image.
    ratio (float): Ratio to determine if the bounding box is significantly out of frame.
    verbose (bool): Print statistics.

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
            print(f"More than {ratio*100}% out of frame:",
                  [x_min, x_max, y_min, y_max])
        return False, [0, 0, 0, 0]
    else:
        # Ensure points are within image dimensions
        valid_mask = ((0 <= points2d[:, 0]) & (points2d[:, 0] < img_w)) & (
            (0 <= points2d[:, 1]) & (points2d[:, 1] < img_h))
        # To be sure we don't get an error
        try:
            x_min, y_min = points2d[valid_mask].min(0)
            x_max, y_max = points2d[valid_mask].max(0)
        except:
            return False, [0, 0, 0, 0]

        x_min = int(max(0, x_min))
        x_max = int(min(x_max, img_w))
        y_min = int(max(0, y_min))
        y_max = int(min(y_max, img_h))

        w, h = x_max - x_min, y_max - y_min

    return True, [x_min, y_min, w, h]


def get_head_annotation(face3d, kp_head_clean, kp_model_clean, cam, img_h, img_w, verbose=False):
    """
    Generates head annotations including bounding box, 6D rotation, and Euler angles.

    Parameters:
    face3d (ndarray): Array of 3D head points of shape (3, N).
    cam (dict): Dictionary with camera matrices.

    Returns:
    tuple: A tuple containing a boolean indicating if the annotation is valid and a dictionary with annotation details.
    """
    rotation_head, translation_head, error, scale = align_3d(
        np.mat(Href_h[0:3, kp_model_clean]),
        np.mat(face3d[:, kp_head_clean]),
    )

    sixD_head, compound = get_rotation_6d(
        rotation_head, translation_head, cam["R"], cam["t"]
    )

    status, [pitch, yaw, roll] = select_euler(
        np.rad2deg(
            inverse_rotate_zyx(compound)
        )  # inverse rotation in order of ZYX
    )
    yaw = -yaw
    roll = -roll

    if status:
        # aligning the generic head sphere with the real head model
        sphere_new = scale * rotation_head @ (sphere) + translation_head

        # project the 3D points of the head model (sphere_new) into 2D image coordinates
        pt_helmet = projectPoints(
            sphere_new, cam["K"], cam["R"], cam["t"], [0, 0, 0, 0, 0]
        )[:2, :]
        # 2D projection with real camera paramters Creal

        valid, head_bbox = get_bbox(
            pt_helmet.T, img_h, img_w, ratio=0.5, verbose=verbose)
        if valid:
            return True, {
                "bbox": head_bbox,
                "6D": sixD_head.tolist(),
                "euler_angles": [pitch, yaw, roll],
                "category": "head",
            }

    return False, {}


def get_body_annotations(body3d, body2d, cam, img_h, img_w, verbose):

    rotation_body, translation_body, _, _ = align_3d(
        np.mat(Bref_h[0:3, kp_idx_body_model]), np.mat(body3d[:, kp_idx_body])
    )

    sixD_body, compound = get_rotation_6d(
        rotation_body, translation_body, cam["R"], cam["t"]
    )

    valid, body_bbox = get_bbox(
        body2d, img_h, img_w, ratio=0.7, verbose=verbose)

    if valid:
        _, [pitch, yaw, roll] = select_euler(
            np.rad2deg(inverse_rotate_zyx(compound)),
            pred=True,  # inverse rotation in order of ZYX
        )
        yaw = -yaw
        roll = -roll

        return True, {
            "bbox": body_bbox,
            "6D": sixD_body.tolist(),
            "euler_angles": [pitch, yaw, roll],
            "category": "body",
        }
    return False, {}


def auto_labels_generating_head(cam, fframe_dict, frame, verbose=False):

    valid_bbox_6D_list = []

    img_h, img_w = frame.shape[0], frame.shape[1]

    cam["K"] = np.mat(cam["K"])
    cam["distCoef"] = np.array(cam["distCoef"])
    cam["R"] = np.mat(cam["R"])
    cam["t"] = np.array(cam["t"]).reshape((3, 1))

    lost_faces = 0

    for face in fframe_dict["people"]:
        # 3D Face has 70 3D joints, stored as an array [x1,y1,z1,x2,y2,z2,...]
        face3d = np.array(face["face70"]["landmarks"]
                          ).reshape((-1, 3)).transpose()
        face_conf = np.asarray(face["face70"]["averageScore"])
        clean_match = (
            face_conf[kp_idx_head] > 0.1
        )  # only pick points confidence higher than 0.1
        kp_head_clean = kp_idx_head[clean_match]
        kp_model_clean = kp_idx_head_model[clean_match]

        if len(kp_head_clean) > 6:
            # This head was in frame and valid angle range
            valid, head_annotation = get_head_annotation(
                face3d, kp_head_clean, kp_model_clean, cam, img_h, img_w, verbose=verbose)
            if valid:
                valid_bbox_6D_list.append(head_annotation)
            else:
                lost_faces += 1
        else:
            if verbose:
                print("Not enough keypoints.")
            lost_faces += 1

    return valid_bbox_6D_list, lost_faces


def auto_labels_generating_body(cam, fframe_dict, frame, verbose=False):

    valid_bbox_6D_list = []

    img_h, img_w = frame.shape[0], frame.shape[1]

    cam["K"] = np.mat(cam["K"])
    cam["distCoef"] = np.array(cam["distCoef"])
    cam["R"] = np.mat(cam["R"])
    cam["t"] = np.array(cam["t"]).reshape((3, 1))

    lost_bodies = 0

    for body in fframe_dict["bodies"]:
        # 3D body has 19 3D joints, stored as an array [x1,y1,z1,c1, x2,y2,z2,c2,...]
        # c is confidence score
        body3d = np.array(body["joints19"]).reshape((-1, 4))[:, :3].T
        body2d = projectPoints(body3d, cam["K"], cam["R"], cam["t"], [
                               0, 0, 0, 0, 0]).T[:, :2]

        valid, body_annotation = get_body_annotations(
            body3d, body2d, cam, img_h, img_w, verbose)

        # This body was in frame
        if valid:
            valid_bbox_6D_list.append(body_annotation)
        else:
            lost_bodies += 1

    return valid_bbox_6D_list, lost_bodies


"""Sampling frames by opencv2, frame_ids and labels could be aligned successfully"""


def create_directory(path):
    if os.path.exists(path):
        raise ValueError(f"Path already exists: {path}")
    os.mkdir(path)


def prepare_directories(sampled_result_path):
    create_directory(os.path.join(sampled_result_path, "images_sampled"))
    create_directory(os.path.join(sampled_result_path, "annotations"))


def download_data(seq_name, hdVideoNum=31):
    subprocess.call(["./scripts/getData_hdVideo.sh",
                    str(seq_name), str(hdVideoNum)])


def parse_video_metadata(capture):
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_seconds = int(frame_count / video_fps)
    video_duration = f"{video_seconds // 60:02}:{video_seconds % 60:02}"
    return video_fps, frame_count, video_duration


def read_frame(capture, frame_id, interval):
    ret, frame = capture.read()
    if frame_id % interval == 0:
        return frame, frame_id
    return None, None


def process_frame_data(seq_name, seq_id, vi, real_frame_id, cam, frame, sampled_result_path, raw_data_path, debug=False):

    hd_face3d_path = os.path.join(raw_data_path, seq_name, "hdFace3d")
    hd_body3d_path = os.path.join(
        raw_data_path, seq_name, "hdPose3d_stage1_coco19")

    face_json_fname = os.path.join(
        hd_face3d_path, f"faceRecon3D_hd{real_frame_id:08d}.json")
    body_json_fname = os.path.join(
        hd_body3d_path, f"body3DScene_{real_frame_id:08d}.json")

    valid_bbox_6D_list_head, lost_faces = [], 0
    if os.path.exists(face_json_fname):
        with open(face_json_fname) as dfile:
            fframe = json.load(dfile)
        valid_bbox_6D_list_head, lost_faces = auto_labels_generating_head(
            cam, fframe, frame, verbose=debug)
    else:
        print("face json fname", face_json_fname)

    valid_bbox_6D_list_body, lost_bodies = [], 0
    if os.path.exists(body_json_fname):
        with open(body_json_fname) as dfile:
            fframe = json.load(dfile)
        valid_bbox_6D_list_body, lost_bodies = auto_labels_generating_body(
            cam, fframe, frame, verbose=debug)
    else:
        print("body json fname", body_json_fname)

    if debug:
        print(f"Lost {lost_bodies} bodies and {lost_faces} faces.")

    # No valid heads and bodies were found
    if len(valid_bbox_6D_list_head) == 0 and len(valid_bbox_6D_list_body) == 0:
        return None, None, None

    image_id = 10000000000 + (seq_id + 1) * 100000000 + \
        real_frame_id * 100 + vi
    dst_img_path = os.path.join(
        sampled_result_path, "images_sampled", f"{image_id}.jpg")
    cv2.imwrite(dst_img_path, frame)
    if os.path.getsize(dst_img_path) < 100 * 1024:  # <100KB
        os.remove(dst_img_path)
        return None, None, None  # this frame may be damaged, skip it

    return image_id, valid_bbox_6D_list_head, valid_bbox_6D_list_body


def generate_coco_annotations(image_id, valid_bbox_6D_list_head, valid_bbox_6D_list_body, img_h, img_w):
    temp_image = {"file_name": f"{image_id}.jpg",
                  "height": img_h, "width": img_w, "id": image_id}
    temp_annotations_list = []

    for index, labels in enumerate(valid_bbox_6D_list_body + valid_bbox_6D_list_head):
        temp_annotation = {
            "bbox": labels["bbox"],
            "6D": labels["6D"],
            "euler_angles": labels["euler_angles"],
            "image_id": image_id,
            "id": image_id * 100 + index,
            "category_id": 1 if labels["category"] == "head" else 2,
            "iscrowd": 0,
            "segmentation": [],
            "area": round(labels["bbox"][-1] * labels["bbox"][-2], 4),
        }
        temp_annotations_list.append(temp_annotation)

    return temp_image, temp_annotations_list


def save_annotations(sampled_result_path, seq_name, coco_style_hpe_dict):
    annotations_path = os.path.join(
        sampled_result_path, "annotations", seq_name)
    if not os.path.exists(annotations_path):
        os.mkdir(annotations_path)

    dst_ann_path = os.path.join(annotations_path, "coco_style_sample.json")
    with open(dst_ann_path, "w") as dst_ann_file:
        json.dump(coco_style_hpe_dict, dst_ann_file)


def process_and_extract_frames_face3d_cv2read(raw_data_path, sampled_result_path, seq_names, skip_frames, debug=True):
    prepare_directories(sampled_result_path)
    img_w, img_h = 1920, 1080  # Default resolution of HD videos

    if debug:
        seq_names = ["171204_pose3", "171026_pose3"]  # For testing

    print("Raw data path:", raw_data_path)
    # Initialize the annotations dictionary
    coco_style_hpe_dict = {"images": [], "annotations": []}

    for seq_id, seq_name in enumerate(tqdm(seq_names)):
        print(f"\n\n{seq_name}\n")

        # Download 31 HD videos and face/body keypoints
        download_data(seq_name)

        hd_camera_path = os.path.join(
            raw_data_path, seq_name, f"calibration_{seq_name}.json")
        cameras = parsing_camera_calibration_params(hd_camera_path)

        for vi in range(31):  # Index 0~30, total 31 HD videos
            cam = cameras[(0, vi)]
            hd_video_path = os.path.join(
                raw_data_path, seq_name, "hdVideos", f"hd_00_{vi:02d}.mp4")
            if not os.path.exists(hd_video_path):
                continue

            capture = cv2.VideoCapture(hd_video_path)
            video_fps, frame_count, video_duration = parse_video_metadata(
                capture)
            print(
                f"{seq_id} {vi} {seq_name} fps: {video_fps} frame_cnt: {frame_count} duration: {video_duration}")

            interval = skip_frames * \
                3 if int(video_duration.split(':')[0]) > 5 else skip_frames

            for frame_id in range(frame_count):
                if debug and frame_id > 900:
                    break

                frame, real_frame_id = read_frame(capture, frame_id, interval)
                if frame is None:
                    continue
                
                image_id, valid_bbox_6D_list_head, valid_bbox_6D_list_body = process_frame_data(
                    seq_name, seq_id, vi, real_frame_id, cam, frame, sampled_result_path, raw_data_path, debug)
                if image_id is None:
                    continue

                temp_image, temp_annotations_list = generate_coco_annotations(
                    image_id, valid_bbox_6D_list_head, valid_bbox_6D_list_body, img_h, img_w)
                coco_style_hpe_dict["images"].append(temp_image)
                coco_style_hpe_dict["annotations"] += temp_annotations_list

            capture.release()
        print(f"{seq_id}\t finished one seq named: {seq_name}")
        shutil.rmtree(os.path.join(raw_data_path, seq_name))
        save_annotations(sampled_result_path, seq_name, coco_style_hpe_dict)


if __name__ == "__main__":

    # If this is changed it also needs to be changed in script `getData_hd_video.sh`
    raw_data_path = "./"

    sampled_result_path = "H_B/"
    if not os.path.exists(sampled_result_path):
        os.mkdir(sampled_result_path)

    seq_names = [
        "171204_pose3",
        "171026_pose3",
        "170221_haggling_b3",
        "170221_haggling_m3",
        "170224_haggling_a3",
        "170228_haggling_b1",
        "170404_haggling_a1",
        "170407_haggling_a2",
        "170407_haggling_b2",
        "171026_cello3",
        "161029_piano4",
        "160422_ultimatum1",
        "160224_haggling1",
        "170307_dance5",
        "160906_ian1",
        "170915_office1",
        "160906_pizza1",
    ]  # 17 names

    """
    Sampled HD video frames every skip_frames (e.g. FPS=30) frames.
    We totally have about 7020 seconds and 31 hd views. 
    Set 60 (2 seconds), will get 109K~ images; Set 180 (6 seconds), will get 36K~ images.
    We set basic skip_frames as 60 for videos with duration <=5 mins, and skip_frames as 60*3 for duration >5 mins
    """
    skip_frames = 60

    print("process_and_extract_frames_face3d_cv2read ...")
    # process_and_extract_frames_face3d_cv2read(raw_data_path, sampled_result_path, seq_names, skip_frames, debug=True)
    process_and_extract_frames_face3d_cv2read(
        raw_data_path, sampled_result_path, seq_names, skip_frames, debug=True
    )
