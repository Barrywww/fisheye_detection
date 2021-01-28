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

from dlp import DLP

DATASET_PATH = "/Users/barrywang/Datasets/wireframe/"
EPOCHS = 150
LR = 0.001
BATCH_SIZE = 15


class DLPLoss(nn.Module):
    def __init__(self):
        super(DLPLoss, self).__init__()

    def forward(self, y_pred, gt):
        """
        x,gt: torch.Tensor, shape:(320,320) (y, x)
        """
        positive_set = 0
        negative_set = 0
        dp_positive = 0
        dp_negative = 0
        
        for y in range(len(y_pred)):
            for x in range(len(y_pred)):
                line_length = calc_length((y, x), gt["lines"])
                if y_pred[y][x] != 0:
                    positive_set += 1
                    dp_positive += (gt["line_length"][y][x] - line_length)**2
                else:
                    negative_set += 1
                    dp_negative += (gt["line_length"][y][x] - line_length)**2
        
        tot = dp_positive + dp_negative
        loss = dp_negative / tot * dp_negative + dp_positive / tot * dp_positive
        
        return loss


def calc_length(p, lines):
    """
    Calculate the minimal distance from p to line in lines
    p: tuple(y, x)
    lines: list of lines in tuple((y1, x1), (y2,x2))
    """
    p = np.array(p)
    min_cord1 = (0,0)
    min_cord2 = (0,0)
    min_dist = 1
    for idx, line in enumerate(lines):
        line_point = (np.array(lines[0]), np.array(lines[1]))
        distance = norm(np.cross((line_point[1] - line_point[0]), (line_point[0] - p))) / norm(line_point[1] - line_point[0])
        if idx == 0 or (distance <= 1 and distance <= min_dist):
            min_dist = distance
            min_cord1 = line_point[0]
            min_cord2 = line_point[1]
            
    return math.sqrt((min_cord2[0] - min_cord1[0])^2 + (min_cord2[1] - min_cord1[1]^2))


def train(model, train_loader):
    loss_func = DLPLoss()
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
    model = DLP(depth=110, alpha=48, nstacks=5)
    
    losses = []
    for i in range(EPOCHS):
        loss = train(model, img_pkl)
        losses.append(loss)
    
    return losses


if __name__ == "__main__":
    main()
