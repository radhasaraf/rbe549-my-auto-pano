"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    loss = nn.MSELoss()
    output = loss(out, labels)
    ###############################################
    return output


class HomographyModel(pl.LightningModule):
    def __init__(self, hparams):
        super(HomographyModel, self).__init__()
        self.hparams = hparams
        self.model = Net()

    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = LossFn(delta, img_a, patch_b, corners)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = LossFn(delta, img_a, patch_b, corners)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class Net(nn.Module):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 16 * 128, 1024)
        self.fc2 = nn.Linear(1024, 8)

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################
    def stn(self, x):
        "Spatial transformer network forward function"
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, xa, xb):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        x = torch.concat([xa, xb])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.conv4(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x