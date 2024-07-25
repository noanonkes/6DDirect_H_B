import os
import pickle

import torch
from torch.utils.data import Dataset, ConcatDataset


class GazeSeqDataset(Dataset):
    def __init__(self, video_path):

        self.video_path = video_path

        # load annotation
        with open(os.path.join(video_path, "6D_gaze_data.pkl"), "rb") as f:
            anno_data = pickle.load(f)
        
        self.gaze_input = [subsublist for sublist in anno_data["input"] for subsublist in sublist]
        self.gaze_target = torch.tensor([subsublist for sublist in anno_data["target"] for subsublist in sublist])

    def __len__(self):
        return len(self.gaze_input)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"index {idx} >= len { len(self) }")

        head_dirs = torch.tensor(self.gaze_input[idx]["head_dirs"])
        body_dirs = torch.tensor(self.gaze_input[idx]["body_dirs"])
        head_scores = torch.tensor(self.gaze_input[idx]["head_scores"]).unsqueeze(dim=-1)
        body_scores = torch.tensor(self.gaze_input[idx]["body_scores"]).unsqueeze(dim=-1)
        indices = torch.tensor(self.gaze_input[idx]["indices"])
        video = self.gaze_input[idx]["video"]

        # n_frames x 6
        gaze_input = {
            "head_dirs": head_dirs,
            "body_dirs": body_dirs,
            "head_scores": head_scores,
            "body_scores": body_scores,
            "indices": indices,
            "video": video
            }

        gaze_dir = self.gaze_target[idx]
        return gaze_input, gaze_dir


def create_gafa_dataset(exp_names, root_dir="./data/preprocessed"):
    exp_dirs = [os.path.join(root_dir, en) for en in exp_names]

    dset_list = []
    for ed in exp_dirs:
        if not os.path.exists(os.path.join(ed, "6D_gaze_data.pkl")):
            print(f"Scene {ed} has no data.")
            continue

        dset = GazeSeqDataset(ed)

        if len(dset) == 0:
            continue

        dset_list.append(dset)

    return ConcatDataset(dset_list)