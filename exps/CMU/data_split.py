
import os
import json
import copy
from tqdm import tqdm

import shutil

############################################################################################

coco_dict_template = {
    "info": {
        "description": "6D rotations, euler angles and bounding boxes of head and body of CMU Panoptic Studio Dataset",
        "url": "http://domedb.perception.cs.cmu.edu/",
        "version": "1.0",
        "year": 2024,
        "contributor": "Noa Nonkes",
        "date_created": "2024/05/17",
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
    }]
}

############################################################################################

def combine_json(seq_names, sampled_result_path="H_B/annotations/"):
    coco_style_hpe_dict = coco_dict_template.copy()

    for seq_name in seq_names:
        if not os.path.exists(os.path.join(sampled_result_path, seq_name)):
            continue
        
        with open(os.path.join(sampled_result_path, seq_name, "coco_style_sample.json"), "r") as f:
            file_content = json.load(f)

        coco_style_hpe_dict["images"].extend(file_content["images"])
        coco_style_hpe_dict["annotations"].extend(file_content["annotations"])

    print("Saving at all combined annotations at:", "CMU/HPE/annotations/coco_style_sample.json")
    with open(os.path.join(sampled_result_path, "coco_style_sample.json"), "w") as f:
        json.dump(coco_style_hpe_dict, f)


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
    
    sampled_anno_dir = "./H_B/annotations/"
    sampled_anno_path = os.path.join(sampled_anno_dir, "coco_style_sample.json")
    sampled_train_path = os.path.join(sampled_anno_dir, "coco_style_sampled_train.json")
    sampled_val_path = os.path.join(sampled_anno_dir, "coco_style_sampled_validation.json")
    
    image_root_path = "./H_B/images_sampled"
    
    image_dst_path = "./H_B/images"
    if os.path.exists(image_dst_path):
        raise ValueError("Images have already been split, see:", image_dst_path)

    os.mkdir(image_dst_path)
    os.mkdir(os.path.join(image_dst_path, "train"))
    os.mkdir(os.path.join(image_dst_path, "validation"))
 
    
    """[start] do not change"""
    # ORDER MATTERS!
    seq_names = ["171204_pose3", "171026_pose3", "170221_haggling_b3", "170221_haggling_m3", "170224_haggling_a3", "170228_haggling_b1", "170404_haggling_a1", "170407_haggling_a2", "170407_haggling_b2", "171026_cello3", "161029_piano4", "160422_ultimatum1", "160224_haggling1", "170307_dance5", "160906_ian1", "170915_office1", "160906_pizza1"]  # 17 names
    
    seq_names_train = ["171204_pose3", "161029_piano4", "160422_ultimatum1", "170307_dance5", "160906_pizza1", "170221_haggling_b3", "170224_haggling_a3", "170404_haggling_a1", "170407_haggling_b2"]  # 9 names, person: 1+1+7+1+5+3+3+3+3
    seq_names_val = ["171026_pose3", "171026_cello3", "160224_haggling1", "160906_ian1", "170915_office1", "170221_haggling_m3", "170228_haggling_b1", "170407_haggling_a2"]  # 8 names, person: 1+1+3+2+1+3+3+3
    
    if not os.path.exists(sampled_anno_path):
        print("First combining all the sequences in a single json.")
        combine_json(seq_names, sampled_anno_dir)

    train_seq_num_list, val_seq_num_list = [], []
    for seq_num, seq_name in enumerate(seq_names):
        if seq_name in seq_names_train: train_seq_num_list.append(seq_num)
        if seq_name in seq_names_val: val_seq_num_list.append(seq_num)

    with open(sampled_anno_path, "r") as json_file:
        annos_dict = json.load(json_file)
    images_list = annos_dict["images"]
    labels_list = annos_dict["annotations"]
    images_labels_dict = sort_labels_by_image_id(labels_list)

    coco_dict_train = copy.deepcopy(coco_dict_template)
    coco_dict_val = copy.deepcopy(coco_dict_template)
    
    person_instances_stat = {}

    for image_dict in tqdm(images_list):
        image_id = image_dict["id"]
        seq_num = (image_id - 10000000000) // 100000000 - 1
        if seq_num in train_seq_num_list: target_type = "train"
        if seq_num in val_seq_num_list: target_type = "validation"
        
        labels_list = images_labels_dict.get(str(image_id), None)
        if labels_list is None:
            continue
        
        anno_nums = len(labels_list) // 2

        image_dict["seq"] = seq_names[seq_num]
        
        src_image_path = os.path.join(image_root_path, image_dict["file_name"])
        dst_image_path = os.path.join(image_dst_path, target_type, image_dict["file_name"])
        if os.path.exists(src_image_path):
            shutil.copy(src_image_path, dst_image_path)

        if target_type == "train":
            coco_dict_train["images"].append(image_dict)
            coco_dict_train["annotations"] += labels_list
            if str(anno_nums) not in person_instances_stat:
                person_instances_stat[str(anno_nums)] = [1, 0]  # [1, 0] for [train, val]
            else:
                person_instances_stat[str(anno_nums)][0] += 1
        if target_type == "validation":
            coco_dict_val["images"].append(image_dict)
            coco_dict_val["annotations"] += labels_list
            if str(anno_nums) not in person_instances_stat:
                person_instances_stat[str(anno_nums)] = [0, 1]  # [0, 1] for [train, val]
            else:
                person_instances_stat[str(anno_nums)][1] += 1            
    """[end] do not change"""
    
    print("\nperson_instances_stat:", person_instances_stat)
    image_cnt, person_cnt = [0,0], [0,0]
    for key, value in person_instances_stat.items():
        image_cnt[0], image_cnt[1] = image_cnt[0] + value[0], image_cnt[1] + value[1]
        person_cnt[0], person_cnt[1] = person_cnt[0] + int(key)*value[0], person_cnt[1] + int(key)*value[1]
        print("Images number containing [%s] persons: %d, \ttrain/val = %d/%d"%(key, sum(value), value[0], value[1]))
    print("Person instances per image: %.4f, \ttrain/val = %.4f/%.4f"%(
        sum(person_cnt)/sum(image_cnt), person_cnt[0]/image_cnt[0], person_cnt[1]/image_cnt[1]))

    print("\ntrain: images --> %d, head instances --> %d"%(len(coco_dict_train["images"]), len(coco_dict_train["annotations"])))  
    with open(sampled_train_path, "w") as json_file:
        json.dump(coco_dict_train, json_file)
    print("val: images --> %d, head instances --> %d"%(len(coco_dict_val["images"]), len(coco_dict_val["annotations"])))
    with open(sampled_val_path, "w") as json_file:
        json.dump(coco_dict_val, json_file)