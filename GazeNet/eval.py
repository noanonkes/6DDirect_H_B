"""
This code is based on Dynamic 3D Gaze from Afar:
    https://github.com/kyotovision-public/dynamic-3d-gaze-from-afar
"""

import argparse

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from dataloader.gafa import create_gafa_dataset
from models.gazenet import GazeNet


def main(opt):
    print("Arguments:", opt)

    # settings for dataset
    test_exp_names = [
        "library/1029_2",
        "lab/1013_2",
        "kitchen/1022_2",
        "living_room/006",
        "courtyard/002",
        "courtyard/003",
    ]

    if opt.scene != "all":
        test_exp_names = [scene for scene in test_exp_names if scene.startswith(opt.scene)]

    # load model
    model = GazeNet(n_frames=opt.n_frames)
    model.load_state_dict(torch.load(
        opt.checkpoint, map_location=torch.device("cpu") if opt.gpu == "cpu" else torch.device(opt.gpu))["state_dict"])

    # make dataloader
    test_dset = create_gafa_dataset(
        exp_names=test_exp_names,
    )

    test_loader = DataLoader(test_dset, batch_size=128,
                             num_workers=4, shuffle=False)

    trainer = Trainer(
        benchmark=True,
        gpus=opt.gpus,
        precision=16,
        accelerator="ddp"
    )

    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames", type=int, default=7)
    parser.add_argument("--checkpoint", type=str, default="output/gazenet.ckpt")
    parser.add_argument("--scene", default="all", choices=["all", "lab", "library", "living_room", "kitchen", "courtyard"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    opt = parser.parse_args()

    main(opt)
