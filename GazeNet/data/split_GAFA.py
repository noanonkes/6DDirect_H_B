"""
Program for splitting Gaze from Afar dataset (GAFA) COCO style annotations to train and test.
`preprocess.py`, `reprocess.py`, and `coco_GAFA.py` should be called first!

Randomly splits GAFA_coco_style.json into train and validation splits (75, 25 respectively).

Do not forget to change `root_dir` to correct path!
"""

import os
import json
import copy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil

############################################################################################

coco_dict_template = {
    "info": {
        "description": "6D rotations, euler angles and bounding boxes of head and body of CMU Panoptic Studio Dataset",
        "url": "http://domedb.perception.cs.cmu.edu/",
        "version": "1.0",
        "year": 2024,
        "contributor": "Noa Nonkes",
        "date_created": "2024/06/20",
    },
    "licences": [{
        "url": "http://creativecommons.org/licenses/by-nc/2.0",
        "name": "Attribution-NonCommercial License"
    }],
    "images": [],
    "annotations": [],
    "categories": [{
        "supercategory": "person",
        "id": 1,
        "name": "head"
    }, {
        "supercategory": "person",
        "id": 2,
        "name": "body"
    }, {
        "supercategory": "person",
        "id": 3,
        "name": "gaze"
    }]
}

############################################################################################

def sort_labels_by_image_id(labels_list):
    images_labels_dict = {}
    for i, labels_dict in enumerate(labels_list):
        image_id = str(labels_dict["image_id"])
        if "head_bbox" in labels_dict:
            labels_dict["bbox"] = labels_dict["head_bbox"]  # please use the default "bbox" as key in cocoapi
            del labels_dict["head_bbox"]
        if "area" not in labels_dict:  # generate standard COCO style json file
            labels_dict["segmentation"] = []  # This script is not for segmentation
            labels_dict["area"] = round(labels_dict["bbox"][-1] * labels_dict["bbox"][-2], 4)
        if image_id in images_labels_dict.keys():
            images_labels_dict[image_id].append(labels_dict)
        else:
            images_labels_dict[image_id] = [labels_dict]
    return images_labels_dict


if __name__ == "__main__":
    train_split, val_split = 0.75, 0.25

    root_dir = Path("data/preprocessed/")

    anno_save_folder = "H_B/annotations/"
    image_save_folder = "H_B/images/"

    sampled_anno_path = root_dir / anno_save_folder / "GAFA_coco_style.json"
    sampled_train_path = root_dir / anno_save_folder / "GAFA_coco_style_train.json"
    sampled_val_path = root_dir / anno_save_folder / "GAFA_coco_style_validation.json"
    
    image_root_path = root_dir / image_save_folder
    image_dst_path = root_dir / image_save_folder

    if os.path.exists(os.path.join(image_dst_path, "train")) or os.path.exists(os.path.join(image_dst_path, "validation")):
        raise ValueError("Images have already been split, see:", image_dst_path)
   
    os.mkdir(os.path.join(image_dst_path, "train"))
    os.mkdir(os.path.join(image_dst_path, "validation"))
    
    with open(sampled_anno_path, "r") as json_file:
        annos_dict = json.load(json_file)

    labels_list = annos_dict["annotations"]
    images_labels_dict = sort_labels_by_image_id(labels_list)

    coco_dict_train = copy.deepcopy(coco_dict_template)
    coco_dict_val = copy.deepcopy(coco_dict_template)

    images_list_train, images_list_val = train_test_split(annos_dict["images"], train_size=train_split, test_size=val_split, random_state=42)
    
    for target_type, images_list in zip(["train", "validation"], [images_list_train, images_list_val]):
        for image_dict in tqdm(images_list):
            image_id = image_dict["id"]
            
            labels_list = images_labels_dict.get(str(image_id), None)
            if labels_list is None:
                continue
            
            anno_nums = len(labels_list)
            
            src_image_path = os.path.join(image_root_path, str(image_id)+".jpg")
            dst_image_path = os.path.join(image_dst_path, target_type, str(image_id)+".jpg")
            
            if os.path.exists(src_image_path):
                shutil.move(src_image_path, dst_image_path)
            else:
                continue

            if target_type == "train":
                coco_dict_train["images"].append(image_dict)
                coco_dict_train["annotations"] += labels_list
                
            if target_type == "validation":
                coco_dict_val["images"].append(image_dict)
                coco_dict_val["annotations"] += labels_list
                    

    print("\ntrain: images --> %d, head instances --> %d"%(len(coco_dict_train["images"]), len(coco_dict_train["annotations"])))  
    with open(sampled_train_path, "w") as json_file:
        json.dump(coco_dict_train, json_file)
    print("val: images --> %d, head instances --> %d"%(len(coco_dict_val["images"]), len(coco_dict_val["annotations"])))
    with open(sampled_val_path, "w") as json_file:
        json.dump(coco_dict_val, json_file)