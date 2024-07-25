"""
This code is based on Dynamic 3D Gaze from Afar:
    https://github.com/kyotovision-public/dynamic-3d-gaze-from-afar
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models.utils import compute_mae
from models.loss import compute_basic_cos_loss, compute_kappa_vMF3_loss


class GazeModule(pl.LightningModule):
    def __init__(self, n_frames, n_hidden=128, n_dir=6):
        super().__init__()
        assert n_frames % 2 == 1
        self.n_frames = n_frames

        # LSTM                     < +1 for confidence score
        self.lstm = nn.LSTM((n_dir + 1) * 2, n_hidden, bidirectional=True, num_layers=2)
        self.direction_layer = nn.Sequential(
            nn.Linear(2 * n_hidden * n_frames, 64),
            nn.ReLU(), #   < 3 for direction
            nn.Linear(64, 3 * n_frames),
        )
        self.kappa_layer = nn.Sequential(
            nn.Linear(2 * n_hidden * n_frames, 64),
            nn.ReLU(),
            nn.Linear(64, n_frames),
            nn.Softplus()
        )
    def forward(self, x):
        # LSTM
        fc_out, _ = self.lstm(x)
        fc_out = F.relu(fc_out).view(fc_out.shape[0], -1)
        # estimate mean of vMF
        direction = self.direction_layer(fc_out)
        direction = direction.reshape(x.shape[0], x.shape[1], 3)
        direction = direction / torch.norm(direction, dim=-1, keepdim=True)
        kappa = self.kappa_layer(fc_out).reshape(x.shape[0], x.shape[1], 1)
        output = {
            "direction": direction,
            "kappa": kappa
        }
        return output


class GazeNet(pl.LightningModule):
    def __init__(self, n_frames=7, lr=1e-4):
        super().__init__()
        self.n_frames = n_frames
        self.lr = lr
        self.gazemodule = GazeModule(n_frames)
        self.automatic_optimization = False

    def forward(self, input):

        head_dirs = input["head_dirs"]
        body_dirs = input["body_dirs"]
        head_scores = input["head_scores"]
        body_scores = input["body_scores"]

        # concat head, body 6D with confidence
        gaze_input = torch.cat((head_dirs, head_scores, body_dirs, body_scores), dim=-1)

        # get gaze direction and kappa
        gaze_res = self.gazemodule(gaze_input)

        return gaze_res

    def configure_optimizers(self):
        opt_direction = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        opt_kappa = torch.optim.Adam(filter(lambda p: p.requires_grad, self.gazemodule.kappa_layer.parameters()), lr=self.lr)
        return opt_direction , opt_kappa

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        gaze_res = self.forward(inputs)

        # take loss for gaze, head and body orientations
        opt_direction, opt_kappa = self.optimizers()
        if batch_idx % 10 != 0:
            loss = compute_basic_cos_loss(gaze_res, targets)
            opt_direction.zero_grad()
            self.manual_backward(loss)
            opt_direction.step()
            self.log_dict({"direction_loss": loss}, prog_bar=True)
        else:
            loss = compute_kappa_vMF3_loss(gaze_res, targets)
            opt_kappa.zero_grad()
            self.manual_backward(loss)
            opt_kappa.step()
            self.log_dict({"kappa_loss": loss}, prog_bar=True)

        mae = compute_mae(gaze_res["direction"], targets)
        self.log("train_mae", mae)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        gaze_res = self.forward(inputs)

        # loss for gaze
        loss = compute_kappa_vMF3_loss(gaze_res, targets)
        mae = compute_mae(gaze_res["direction"], targets)

        self.log("val_mae", mae)
        self.log("val_loss", loss)

        return mae

    def validation_epoch_end(self, outputs):
        val_mae = torch.stack([x for x in outputs]).flatten()
        val_mae_mean = val_mae[~torch.isnan(val_mae)].mean()
        print("MAE (validation): ", val_mae_mean)
        self.log("val_mae", val_mae_mean)

    def test_step(self, batch, batch_idx):

        inputs, gaze_label = batch

        gaze_res = self.forward(inputs)

        prediction = gaze_res["direction"]

        if gaze_label.shape[-1] == 3:
            front_index = torch.arange(gaze_label.shape[0]).to(self.device)[gaze_label[:, 0, -1] <= 0]
            back_index = torch.arange(gaze_label.shape[0]).to(self.device)[gaze_label[:, 0, -1] > 0]

            # 3D MAE
            mae = compute_mae(prediction, gaze_label)

            # 3D MAE for front facing
            front_mae = compute_mae(prediction[front_index], gaze_label[front_index])

            # 3D MAE for back facing
            back_mae = compute_mae(prediction[back_index], gaze_label[back_index])

            gaze_label_2d = gaze_label[..., :2] / torch.norm(gaze_label[..., :2], dim=-1, keepdim=True)
            prediction_2d = prediction[..., :2] / torch.norm(prediction[..., :2], dim=-1, keepdim=True)

            # 2D MAE
            mae_2d = compute_mae(prediction_2d, gaze_label_2d)
            front_mae_2d = compute_mae(prediction_2d[front_index], gaze_label_2d[front_index])
            back_mae_2d = compute_mae(prediction_2d[back_index], gaze_label_2d[back_index])

        elif gaze_label.shape[-1] == 2:
            gaze_label_2d = gaze_label[..., :2] / torch.norm(gaze_label[..., :2], dim=-1, keepdim=True)
            prediction_2d = prediction[..., :2] / torch.norm(prediction[..., :2], dim=-1, keepdim=True)
            mae = front_mae = back_mae = 0
            front_mae_2d = back_mae_2d = 0

            # 2D MAE
            mae_2d = compute_mae(prediction_2d, gaze_label_2d)
            print("MAE (test): ", mae_2d)

        return mae, mae_2d, front_mae, front_mae_2d, back_mae, back_mae_2d

    def test_epoch_end(self, outputs):
        mae = np.nanmean([x[0] for x in outputs])
        mae_2d = np.nanmean([x[1] for x in outputs])
        front_mae = np.nanmean([x[2] for x in outputs])
        front_mae_2d = np.nanmean([x[3] for x in outputs])
        back_mae = np.nanmean([x[4] for x in outputs])
        back_mae_2d = np.nanmean([x[5] for x in outputs])

        print("MAE (3D front): ", front_mae)
        print("MAE (2D front): ", front_mae_2d)
        print("MAE (3D back): ", back_mae)
        print("MAE (2D back): ", back_mae_2d)
        print("MAE (3D all): ", mae)
        print("MAE (2D all): ", mae_2d)