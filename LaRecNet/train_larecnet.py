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
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader

from larecnet import LaRecNet

DATASET_PATH = "/Users/barrywang/Datasets/wireframe/"
EPOCHS = 150
LR = 0.001
BATCH_SIZE = 15
PARAM_WEIGHT = torch.Tensor([0.1, 0.1, 0.5, 1, 1, 0.1, 0.1, 0.1, 0.1]).resize_(9,1)


def rf(angle, k, num_params=5):
    result = 0
    for i in range(num_params):
        result += k[i] * angle^(2*(i+1) - 1)

    return result


class LaRecNetLoss(nn.Module):
    def __init__(self, w, lambda_fus=2, lambda_global=1, lambda_local=1, 
                 lambda_m=2, lambda_geo=100, lambda_pix=1, lambda_para=1):
        super(LaRecNetLoss, self).__init__()
        # params for MCM
        self.w = w
        self.lambda_fus = lambda_fus
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        
        #params for overall network
        self.lambda_m = lambda_m
        self.lambda_geo = lambda_geo
        self.lambda_pix = lambda_pix
        self.lambda_para = lambda_para
        
    def forward(self, x, gt):
        k_local, k_global, k_hat = x[0], x[1], x[2]
        loss_global = 1/9 * (np.dot(self.w, (k_global, gt["distortion"])))^2
        loss_local = 1/25 * ((self.w[0:5] * np.dot(k_local, gt["distortion"][0:5])))^2
        loss_fused = 1/9 * (np.dot(self.w, (k_hat, gt["distortion"])))^2
        
        loss_para = self.lambda_fus * loss_fused + self.lambda_global * loss_global + self.lambda_local * loss_local
        
        geometric_err = 0
        image_size = len(gt["img"][0])

        return
    
 
def train(model, train_loader):
    loss_func = LaRecNetLoss()
    optimizer = torch.optim.SGD(lr=LR)
    
    total_loss = 0
    
    for idx, pkl in enumerate(train_loader):
        prediction = model(pkl[idx]["img"])
        loss = loss_func(prediction, pkl[idx]["heatmap"])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss


def test(model, test_loader):
    return


def main():
    filename = "00030043.pkl"
    # wireframe_dataset = DataLoader()
    img_file = open(DATASET_PATH + filename, "rb")
    img_file.close()
    
    img_pkl = [pickle.load(img_file)]
    model = LaRecNet(depth=110, alpha=48, nstacks=5)
    
    losses = []
    for i in range(EPOCHS):
        loss = train(model, img_pkl)
        losses.append(loss)
    
    return losses

if __name__ == "__main__":
    main()       