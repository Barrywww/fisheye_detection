import os
import time
import math
import shutil
import pickle

import numpy as np
from numpy.linalg import *

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from datasets import *
from torch.utils.data import DataLoader

from larecnet import LaRecNet
from resnet import BasicBlock

DATASET = "wireframe"
DATASET_PATH = "/Users/barrywang/Datasets/wireframe/"
EPOCHS = 3
LR = 0.0001
BATCH_SIZE = 4
# PARAM_WEIGHT = torch.Tensor([0.1, 0.1, 0.5, 1, 1, 0.1, 0.1, 0.1, 0.1]).resize_(9, 1)


def r_f(angle, k, num_params=5):
    result = 0
    for i in range(num_params):
        result += k[i] * angle ** (2 * (i + 1) - 1)

    return result


class LaRecNetLoss(nn.Module):
    def __init__(self, weights=[], lambda_fus=2, lambda_global=1, lambda_local=1,
                 lambda_m=2, lambda_geo=100, lambda_pix=1, lambda_para=1):
        super(LaRecNetLoss, self).__init__()
        # params for MCM
        self.weights = np.ones(9)
        self.lambda_fus = lambda_fus
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local

        # params for overall network
        self.lambda_m = lambda_m
        self.lambda_geo = lambda_geo
        self.lambda_pix = lambda_pix
        self.lambda_para = lambda_para

        # fix focal length: True ? False
        self.fix_focal = False
        self.focal = 0

    def fix_focal_length(self, f):
        self.fix_focal = True
        self.focal = f

    def forward(self, x, gt):
        k_local, k_global, k_hat = x[0], x[1], x[2]
        loss_global = 1 / 9 * (np.dot(self.weights, (k_global, gt["distortion"]))) ** 2
        loss_local = 1 / 25 * (self.weights[0:5] * np.dot(k_local, gt["distortion"][0:5])) ** 2
        loss_fused = 1 / 9 * (np.dot(self.weights, (k_hat, gt["distortion"]))) ** 2

        # loss_para = self.lambda_fus * loss_fused + self.lambda_global * loss_global + self.lambda_local * loss_local
        #
        # geometric_err = 0
        # image_size = len(gt["img"][0])

        return loss_fused + loss_local + loss_global


def train(model, inputs, ground_truth):
    # loss_func = LaRecNetLoss(weights)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_loss = 0
    inputs = torch.reshape(inputs, (BATCH_SIZE, 3, 320, 320))
    prediction = model(inputs)
    # print("Model Output:", prediction)
    loss = loss_func(prediction, ground_truth)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    return total_loss


def test(model, test_loader):
    return


def main():
    if DATASET == "wireframe":
        wireframe = Wireframe("/Users/barrywang/datasets/wireframe/v1.1/test1.txt")
        dataset_loader = DataLoader(dataset=wireframe, batch_size=BATCH_SIZE, shuffle=True)
    else:
        dataset_loader = None

    model = LaRecNet(block=BasicBlock, layers=[2, 2, 2, 2], batch_size=BATCH_SIZE)
    losses = []

    for i in range(EPOCHS):
        for idx, data in enumerate(dataset_loader):
            inputs, ground_truth = data
            loss = train(model, inputs, ground_truth)
            losses.append(loss)
        print(losses)
        print("EPOCH %d FINISH" % i)
    return sum(losses) / len(losses)


if __name__ == "__main__":
    main()
