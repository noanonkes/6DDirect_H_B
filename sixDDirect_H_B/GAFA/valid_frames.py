from collections import defaultdict
import argparse
import pickle
import os


def main():

    train_exp_names = [
        "library/1026_3/",
        "library/1028_2/",
        "library/1028_5/",
        "lab/1013_1/",
        "lab/1014_1/",
        "kitchen/1022_4/",
        "kitchen/1015_4/",
        "living_room/004/",
        "living_room/005/",
        "courtyard/004/",
        "courtyard/005/",
    ]

    test_exp_names = [
        "library/1029_2/",
        "lab/1013_2/",
        "kitchen/1022_2/",
        "living_room/006/",
        "courtyard/002/",
        "courtyard/003/",
        ]
    
    opt = parse_opt()

    n_frames = opt.n_frames
    root_dir = opt.root_dir
    target_dir = opt.target_dir
    filename = opt.filename

    for i, scene in enumerate(train_exp_names + test_exp_names):
        print("Scene:", scene)
        scene_dict = {"input": [],
                      "target": []}

        for vd in os.listdir(os.path.join(root_dir, scene)):
            data_path = os.path.join(root_dir, scene, vd, filename)
            if not os.path.exists(data_path):
                continue
            gaze_input, gaze_target = get_frames(data_path, scene + vd, n_frames)
            scene_dict["input"].append(gaze_input)
            scene_dict["target"].append(gaze_target)

        with open(os.path.join(target_dir, scene, "6D_gaze_data.pkl"), "wb") as f:
            pickle.dump(scene_dict, f)


def parse_opt():
    parser = argparse.ArgumentParser(prog="valid_frames.py")
    parser.add_argument("--n-frames", type=int, default=7, help="amount of frames to use as gaze input")
    parser.add_argument("--root-dir", type=str, default="../GazeNet/data/preprocessed/")
    parser.add_argument("--target-dir", type=str, default="../GazeNet/data/preprocessed/")
    parser.add_argument("--filename", type=str, default="6DDirect_H_B_preds.pkl", help="name of file containing predictions")

    opt = parser.parse_args()
    return opt


def get_frames(file, video_name, n_frames=7):
    with open(file, "rb") as f:
        preds = pickle.load(f)

    # Get predictions per index frame
    index2preds = defaultdict(list)
    for p in preds:
        index2preds[p["index"]].append(p)
        
    # Get the most certain prediction from predictions
    valid_data = {
        "video": video_name,
        "indices": [], 
        "pd_head_dirs": [], 
        "pd_body_dirs": [], 
        "head_scores": [],
        "body_scores": [],
        "gt_head_dirs": [], 
        "gt_body_dirs": [],
        "gt_gaze_dirs": []
        }

    for idx in index2preds:
        head_max, body_max = 0, 0
        head_best, body_best = None, None
        for p in index2preds[idx]:
            if p["category_id"] == 1 and p["score"] > head_max:
                head_best = p
                head_max = p["score"]
            elif p["category_id"] == 2 and p["score"] > body_max:
                body_best = p
                body_max = p["score"]
        # This frame had 2 predictions!
        if head_best is not None and body_best is not None:
            valid_data["indices"].append(idx)
            valid_data["pd_head_dirs"].append(head_best["6D"])
            valid_data["head_scores"].append(head_best["score"])
            valid_data["pd_body_dirs"].append(body_best["6D"])
            valid_data["body_scores"].append(body_best["score"])
            # GT are the same for head and body
            valid_data["gt_head_dirs"].append(body_best["gt_head_dir"])
            valid_data["gt_body_dirs"].append(body_best["gt_body_dir"])
            valid_data["gt_gaze_dirs"].append(body_best["gt_gaze_dir"])
        else:
            print(f"Did not find head and body prediction for {video_name} at index {idx}")
            
    # Get n_frames consecutive frames
    gaze_input = []
    gaze_target = []
    for i in range(len(valid_data["indices"])-n_frames):
        valid = True
        for j in range(n_frames):
            # These are not consecutive frames
            if valid_data["indices"][i+j] != valid_data["indices"][i+j+1]-1:
                print(f"Index {valid_data['indices'][i+j]} does not equal {valid_data['indices'][i+j+1]-1}")
                valid = False
                break
        if valid:
            gaze_input.append({"head_dirs": valid_data["pd_head_dirs"][i:i+n_frames],
                               "head_scores": valid_data["head_scores"][i:i+n_frames],
                               "body_dirs": valid_data["pd_body_dirs"][i:i+n_frames],
                               "body_scores": valid_data["body_scores"][i:i+n_frames],
                               "indices": valid_data["indices"][i:i+n_frames],
                               "video": valid_data["video"]})
            gaze_target.append(valid_data["gt_gaze_dirs"][i:i+n_frames])

    return gaze_input, gaze_target
    
            
if __name__ == "__main__":
    main()