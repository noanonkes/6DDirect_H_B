"""
Program for processing Gaze from Afar dataset (GAFA) to COCO style annotations.
`preprocess.py` and `reprocess.py` should be called first!

For each annotations(_i).pkl and corresponding images(_i).pkl file it:
    1. Samples every 7th frame of valid frames;
    2. Gets all valid frames for samples and its annotations (bbox, gaze/head/body dir + 6D + euler angles);
    3. Save image information and annotations to COCO style dictionary.

Do not forget to change `root_dir` to correct path!
"""

import os
import cv2
import joblib, json
from pathlib import Path
import pickle

from utils import get_rot_reps


coco_style_hpe_dict = {
    "info": {
        "description": "Gaze, head and chest 3D directions from the GAFA dataset.",
        "url": "https://vision.ist.i.kyoto-u.ac.jp/",
        "version": "1.0",
        "year": 2024,
        "contributor": "Noa Nonkes",
        "date_created": "2024/06/20",
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
        {"supercategory": "person", "id": 3, "name": "gaze"},
    ],
}

cat2id = {"head": 1,
          "body": 2,
          "gaze": 3}


def parse_annotations(indices, annotations):
    """
    Indices is which datapoints in the sequence we sample (len(indices) == len(annotations["index"]) if sample everything)
    """
    data = []

    for i in indices:
        try:
            sixD, euler = get_rot_reps(annotations["heads_dir"][i])
        except ValueError as e:
            print("Head value error:", e)
            continue

        head = {
            "bbox": annotations["head_bb"][i],
            "6D": sixD,
            "euler_angles": euler,
            "vec": list(annotations["heads_dir"][i]),
            "category": "head",
        }

        try:
            sixD, euler = get_rot_reps(annotations["gazes_dir"][i])
        except ValueError as e:
            print("Gaze value error:", e)
            continue

        gaze = {
            "bbox": annotations["head_bb"][i],
            "6D": sixD,
            "euler_angles": euler,
            "vec": list(annotations["gazes_dir"][i]),
            "category": "gaze",
        }

        try:
            sixD, euler = get_rot_reps(annotations["bodies_dir"][i])
        except ValueError as e:
            print("Body value error:", e)
            continue
    
        body = {
            "bbox": annotations["body_bb"][i],
            "6D": sixD,
            "euler_angles": euler,
            "vec": list(annotations["bodies_dir"][i]),
            "category": "body",
        }

        data.append(([head, gaze, body], annotations["index"][i]))

    return data


if __name__ == "__main__":
    DEBUG = False

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

    if DEBUG:
        train_exp_names = train_exp_names[:1]
    
    root_dir = Path("data/preprocessed/")

    anno_save_folder = "H_B/annotations/"
    anno_save_file = "GAFA_coco_style.json"

    if os.path.exists(root_dir / anno_save_folder / anno_save_file):
        raise ValueError("Annotations already generated.")

    os.makedirs(root_dir / anno_save_folder, exist_ok=True)

    anno_save_path = root_dir / anno_save_folder / anno_save_file
    
    images_save_folder = "H_B/images/"
    os.makedirs(root_dir / images_save_folder, exist_ok=True)
    if len(os.listdir(root_dir / images_save_folder)) != 0:
        raise ValueError("Images already generated.")

    total_images = 0

    # from sequence name to amount
    mapping_dict = {}
    
    # for sequence in train_exp_names + test_exp_names:
    for sequence in train_exp_names:
        print("Current sequence:", str(root_dir / sequence))

        cameras = os.listdir(root_dir / sequence)

        for camera in cameras:
            if camera.endswith(".pkl"):
                continue
            
            annotation_file_path = root_dir / sequence / camera

            annotations_files = sorted([file for file in os.listdir(annotation_file_path) if file.startswith("clean_annotations")])
            images_files = sorted([file for file in os.listdir(annotation_file_path) if file.startswith("images")])

            for i in range(len(annotations_files)):

                annotation_file_path = root_dir / sequence / camera / annotations_files[i]
                image_file_path = root_dir / sequence / camera / images_files[i]
                try:
                    print("Annotations file:", str(annotation_file_path))
                    print("Images file:", str(image_file_path))

                    annotations = joblib.load(annotation_file_path)
                    for k in ["bodies_dir", "heads_dir", "head_bb", "body_bb"]:
                        if len(annotations["index"]) != len(annotations[k]):
                            raise ValueError("Annotations not all same length", len(annotations["index"]), len(annotations[k]))
                    
                    with open(image_file_path, "rb") as f:
                        images = pickle.load(f)
                    
                    total_images += len(images)
        
                except Exception as e:
                    print(f"Annotations file {str(annotation_file_path)} or images file {str(image_file_path)} not correct.")
                    print("ERROR:", e)
                    continue
                
                total_samples = len(annotations["index"])
                if total_samples != len(images):
                    print("Not enough data for the amount of images:", len(total_samples), len(images))
                    continue

                # uniformly sample 1/7 of sequence
                num_samples = total_samples // 7
                step = total_samples // num_samples
                valid_indices = list(range(total_samples))[::step]
                print(f"Sampled {len(valid_indices)} data points.")

                valid_data = parse_annotations(valid_indices, annotations)
                n_valid = len(valid_data)

                if n_valid == 0:
                    print("No valid data")
                    continue

                # get a unique number representing sequence name (e.g. ./raw_data/living_room/006/Camera_13_4 -> 3)
                seq_name = "/".join([sequence, camera])
                if seq_name in mapping_dict:
                    mapping_dict[seq_name] += 1
                else:
                    mapping_dict[seq_name] = 1
                seq_key_list = list(mapping_dict.keys())
                seq_ind = seq_key_list.index(seq_name) + 1

                # all videos in a video have same height and width
                img_h, img_w, c = list(images.values())[0].shape

                for i, (dir_dicts, idx) in enumerate(valid_data):
                    # idx is frame index
                    image_key = f"{idx:06}.jpg"

                    image_id = (
                        1000000000 + seq_ind * 100000 + idx
                    )  # idx is 0-20000 -> frame index

                    image_dict = {
                        "file_name": "/".join([sequence, camera, str(image_key)]),
                        "height": int(img_h),
                        "width": int(img_w),
                        "id": int(image_id),
                        "seq_key": seq_name,
                    }

                    cur_img = images.get(image_key, None)
                    try:
                        dst_img_path = os.path.join(root_dir / images_save_folder, str(image_id) + ".jpg")
                        cv2.imwrite(str(dst_img_path), cur_img[...,::-1])
                    except Exception as e:
                        print("Cant write image", image_key)
                        print("Error:", e)
                        continue

                    temp_annotations = []
                    for j, dir_dict in enumerate(dir_dicts):
                        category_id = cat2id[dir_dict["category"]]

                        temp_annotation = {
                            "bbox": [round(float(x)) for x in dir_dict["bbox"]],  # please use the default "bbox" as key in cocoapi
                            "6D": [round(float(x), 4) for x in dir_dict["6D"]],
                            "euler_angles": [round(float(x), 4) for x in dir_dict["euler_angles"]],
                            "vec": [round(float(x), 4) for x in dir_dict["vec"]],
                            "image_id": int(image_id),
                            "id": int(image_id * 100 + j),  # we support that no image has more than 100 persons/poses
                            "category_id": int(category_id),
                            "iscrowd": 0,
                            "segmentation": [],  # This script is not for segmentation
                            "area": round(float(dir_dict["bbox"][-1] * dir_dict["bbox"][-2]), 4)
                        }

                        temp_annotations.append(temp_annotation)

                    coco_style_hpe_dict["images"].append(image_dict)
                    coco_style_hpe_dict["annotations"] += temp_annotations

    with open(root_dir / anno_save_folder / anno_save_file, "w") as dst_ann_file:
        json.dump(coco_style_hpe_dict, dst_ann_file)
    
    print(
        "\ntrain: original images-->%d, left images-->%d, left instances-->%d"
        % (
            total_images,
            len(coco_style_hpe_dict["images"]),
            len(coco_style_hpe_dict["annotations"]),
        )
    )