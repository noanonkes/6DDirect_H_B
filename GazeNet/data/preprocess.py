"""
This code is based on Dynamic 3D Gaze from Afar:
    https://github.com/kyotovision-public/dynamic-3d-gaze-from-afar


Program for processing Gaze from Afar dataset (GAFA) to annotations and pickle files.

For each video:
    1. For all directions (gaze, body, head) get the corresponding rotation matrices
    2. Resize image of frame so image width = 720;
    3. Get head and body bounding box from OpenPose 2D annotations;
    4. Save all frames to `image.pkl` file;
    5. Save all annotations to `annotations.pkl` file.
    
NOTE: Long videos are saved in three parts to save RAM.

Do not forget to change `root_dir` and `target_dir` to correct path!
"""

import os, pickle
import numpy as np

from PIL import Image
from tqdm import tqdm

from glob import glob

from torch.utils.data import Dataset
from torchvision.transforms import Compose as ComposeTransform

from utils import get_rotation

from preprocessed.transforms import (
    ExpandBB,
    ExpandBBRect,
    CropBB,
    ReshapeBBRect,
    KeypointsToBB,
)


class GazeSeqDataset(Dataset):
    def __init__(self, video_path, annotation_path, target_width=720):
        print("Video path:", video_path)
        self.video_path = video_path
        self.target_width = target_width

        # load annotation
        with open(annotation_path, "rb") as f:
            anno_data = pickle.load(f)
        
        self.keypoints = anno_data["keypoints2d"]
        self.bodies_dir = anno_data["chest"]["direction"].astype(np.float32)
        self.heads_dir = anno_data["head"]["direction"].astype(np.float32)
        self.gazes_dir = anno_data["gaze"]["direction"].astype(np.float32)
        self.body_pos = anno_data["chest"]["position"].astype(np.float32)
        self.head_pos = anno_data["head"]["position"].astype(np.float32)
        self.R_cam = anno_data["R_cam2w"].astype(np.float32)
        self.t_cam = anno_data["t_cam2w"].astype(np.float32)

        if not all(len(l) == len(self.heads_dir) for l in [self.heads_dir, self.bodies_dir, self.gazes_dir]):
            print("Lengths do not match up", len(self.heads_dir), len(self.bodies_dir), len(self.gazes_dir))
        
        ref_vec = np.zeros_like(self.heads_dir)
        ref_vec[:, -1] = -1.
        self.heads_rot = get_rotation(ref_vec, self.heads_dir / np.linalg.norm(self.heads_dir, axis=-1, keepdims=True))
        ref_vec = np.zeros_like(self.bodies_dir)
        ref_vec[:, -1] = -1.
        self.bodies_rot = get_rotation(ref_vec, self.bodies_dir / np.linalg.norm(self.bodies_dir, axis=-1, keepdims=True))
        ref_vec = np.zeros_like(self.gazes_dir)
        ref_vec[:, -1] = -1.
        self.gazes_rot = get_rotation(ref_vec, self.gazes_dir / np.linalg.norm(self.gazes_dir, axis=-1, keepdims=True))

        # extract valid frames
        self.valid_index = np.where(~np.all(
            self.keypoints[:, [0, 1, 15, 16, 17, 18], :2] == 0, axis=(1,2)))[0]

        # head bounding box transform
        self.head_transform = ComposeTransform(
            [
                KeypointsToBB((0, 1, 15, 16, 17, 18)),
                ExpandBB(0.85, -0.2, 0.1, 0.1, "bb"),
                ExpandBBRect("bb"),
            ]
        )

        # define transform for body
        self.body_transform = ComposeTransform(
            [
                KeypointsToBB(slice(None)),
                ExpandBB(0.15, 0.05, 0.1, 0.1, "bb"),
                ExpandBBRect("bb"),
                ReshapeBBRect((256, 192)),
                CropBB(bb_key="bb"),
            ]
        )

    def __len__(self):
        return len(self.valid_index)

    def run_preprocessing(self, target_dir):
        head_bb_list = []
        body_bb_list = []

        image_dict = dict()
        for idx in tqdm(self.valid_index):

            # read image
            img_path = os.path.join(self.video_path, f"{idx:06}.jpg")
            img = Image.open(img_path)
            
            width, height = img.size
            target_height = int(height * (self.target_width / width))
            img = img.resize((self.target_width, target_height))

            item = {
                "image": img,
                "keypoints": self.keypoints[idx, :, :2],
            }

            w_ratio = self.target_width / width
            h_ratio = target_height / height

            # apply image transform
            head_trans = self.head_transform(item)
            head_bb = head_trans["bb"]
            head_bb = [int(w_ratio * head_bb["u"]), int(h_ratio * head_bb["v"]), int(w_ratio * head_bb["w"]), int(h_ratio *head_bb["h"])]
            head_bb_list.append(head_bb)

            body_trans = self.body_transform(item) 
            body_bb = body_trans["bb"]
            body_bb = [int(w_ratio * body_bb["u"]), int(h_ratio * body_bb["v"]), int(w_ratio * body_bb["w"]), int(h_ratio * body_bb["h"])]
            body_bb_list.append(body_bb)

            # save images
            image_dict[f"{idx:06}.jpg"] = np.asarray(img)            

        # save images
        with open(os.path.join(target_dir, "images.pkl"), "wb") as f:
            print("Saving image data to: images.pkl")
            pickle.dump(image_dict, f)

        # save annotations
        data = {
            # "keypoints": self.keypoints[self.valid_index],
            "bodies_dir": self.bodies_dir[self.valid_index],
            "heads_dir": self.heads_dir[self.valid_index],
            "gazes_dir": self.gazes_dir[self.valid_index],
            "bodies_rot": self.bodies_rot[self.valid_index],
            "heads_rot": self.heads_rot[self.valid_index],
            "gazes_rot": self.gazes_rot[self.valid_index],
            # "body_pos": self.body_pos[self.valid_index],
            # "head_pos": self.head_pos[self.valid_index],
            "head_bb": np.array(head_bb_list),
            "body_bb": np.array(body_bb_list),
            # "R_cam": self.R_cam,
            # "t_cam": self.t_cam,
            "index": self.valid_index
        }

        with open(os.path.join(target_dir, "annotations.pkl"), "wb") as f:
            print("Saving annotations to: annotations.pkl")
            pickle.dump(data, f)
        
    def run_preprocessing_thrice(self, target_dir):
        head_bb_list = []
        body_bb_list = []
        part_valid_index = []

        image_dict = dict()
        n = 0

        total_iterations = len(self.valid_index)
        for i, idx in tqdm(enumerate(self.valid_index), total=total_iterations):

            # read image
            img_path = os.path.join(self.video_path, f"{idx:06}.jpg")
            img = Image.open(img_path)
            
            width, height = img.size
            target_height = int(height * (self.target_width / width))
            img = img.resize((self.target_width, target_height))

            item = {
                "image": img,
                "keypoints": self.keypoints[idx, :, :2],
            }

            w_ratio = self.target_width / width
            h_ratio = target_height / height

            # apply image transform
            head_trans = self.head_transform(item)
            head_bb = head_trans["bb"]
            head_bb = [int(w_ratio * head_bb["u"]), int(h_ratio * head_bb["v"]), int(w_ratio * head_bb["w"]), int(h_ratio *head_bb["h"])]
            head_bb_list.append(head_bb)

            body_trans = self.body_transform(item) 
            body_bb = body_trans["bb"]
            body_bb = [int(w_ratio * body_bb["u"]), int(h_ratio * body_bb["v"]), int(w_ratio * body_bb["w"]), int(h_ratio * body_bb["h"])]
            body_bb_list.append(body_bb)

            # save images
            image_dict[f"{idx:06}.jpg"] = np.asarray(img) 

            part_valid_index.append(idx)

            # save if 1/3 mark, 2/3 mark, or last iteration
            if i == len(self.valid_index) // 3 or i == 2 * (len(self.valid_index) // 3) or i == len(self.valid_index) - 1:   
                print("Ready to save", i)     

                # save images
                with open(os.path.join(target_dir, f"images_{n}.pkl"), "wb") as f:
                    print("Saving image data to: images.pkl")
                    pickle.dump(image_dict, f)

                # save annotations
            
                part_valid_index = np.array(part_valid_index)

                data = {
                    # "keypoints": self.keypoints[part_valid_index],
                    "bodies_dir": self.bodies_dir[part_valid_index],
                    "heads_dir": self.heads_dir[part_valid_index],
                    "gazes_dir": self.gazes_dir[part_valid_index],
                    "bodies_rot": self.bodies_rot[part_valid_index],
                    "heads_rot": self.heads_rot[part_valid_index],
                    "gazes_rot": self.gazes_rot[part_valid_index],
                    # "body_pos": self.body_pos[part_valid_index],
                    # "head_pos": self.head_pos[part_valid_index],
                    "head_bb": np.array(head_bb_list),
                    "body_bb": np.array(body_bb_list),
                    # "R_cam": self.R_cam,
                    # "t_cam": self.t_cam,
                    "index": part_valid_index
                }

                with open(os.path.join(target_dir, f"annotations_{n}.pkl"), "wb") as f:
                    print("Saving annotations to: annotations.pkl")
                    pickle.dump(data, f)

                # WE NEED SPACE!!!!
                del data
                del image_dict

                head_bb_list = []
                body_bb_list = []
                part_valid_index = []
                image_dict = dict()
                n += 1


def main(root_dir="data/raw_data/", target_dir="data/preprocessed/"):

    # all dataset
    exp_names = [
        "lab/1013_1",
        "lab/1014_1",
        "lab/1013_2",
        #################
        "kitchen/1022_4",
        "kitchen/1015_4",
        "kitchen/1022_2",
        #################
        "living_room/004",
        "living_room/005",
        "living_room/006",
        #################
        "library/1026_3",
        "library/1028_2",
        "library/1028_5",
        "library/1029_2",
        #################
        "courtyard/002",
        "courtyard/003",
        "courtyard/004",
        "courtyard/005",
    ]

    exp_dirs = [os.path.join(root_dir, en) for en in exp_names]
    for ed in exp_dirs:
        print("Scene:", ed)

        annot_files = sorted(glob(os.path.join(ed, "*.pkl")))
        video_dirs = [af.replace(".pkl", "") for af in annot_files]

        for vd, af in zip(video_dirs, annot_files):
            dset = GazeSeqDataset(vd, af)
            dirname = os.path.join(target_dir, "/".join(os.path.normpath(vd).split(os.sep)[-3:]))
            if os.path.exists(dirname):
                print("Skipped", vd)
                continue
            else:
                try:
                    os.makedirs(dirname)
                except FileExistsError:
                    continue

            try:
                if len(dset) > 8000:
                    print("Longer than 8000:", len(dset))
                    dset.run_preprocessing_thrice(target_dir=dirname)
                else:
                    dset.run_preprocessing(target_dir=dirname)
            except AssertionError as ae:
                    print(f"AssertionError: {ae}")
                    continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                # shutil.rmtree(dirname)
                continue

if __name__=="__main__":
    main()
