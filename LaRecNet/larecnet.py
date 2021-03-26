import math
import torch
import torch.nn as nn
import numpy as np
from dlp import DLP
from torch import Tensor
from resnet import BasicBlock, conv1x1, conv3x3
from calibration import CalibrationGlobal, CalibrationLocal
from typing import Type, Any, Callable, Union, List, Optional


class LaRecNet(nn.Module):
    def __init__(self, block, layers, batch_size, groups=1, width_per_group= 64):
        # receives modified dataset after DLP module
        # 320 * 320 * 4 (RGB + heatmap channel)

        super(LaRecNet, self).__init__()
        self.batch_size = batch_size
        self.inplanes = 256
        self.groups = groups
        self.base_width = width_per_group
        self.block = BasicBlock
        self._norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Resnet L1-L4
        self.layer1 = self._make_layer(block, 256, layers[0])
        self.layer2 = self._make_layer(block, 512, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 1024, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 2048, layers[3], stride=4)

        self.calibration_global = CalibrationGlobal(self.batch_size)
        self.calibration_local = CalibrationLocal(corner_side=5, ctr_side=6, batch_size=self.batch_size)

        self.linear = torch.nn.Linear(9, 1)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)
    

    def forward(self, x):
        """
        output: torch.Tensor shape = (9)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # print("Before ResNet:", x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print("After ResNet:", x.shape)

        k_global = self.calibration_global(x).reshape(self.batch_size, 9)
        # k_local = torch.from_numpy(np.random.rand(5))
        k_local = self.calibration_local(x)
        # print("k shape:", k_global.shape, k_local.shape)
        k_local = torch.cat([k_local, k_global[:, 5:]], dim=1)
        k_avg = k_local + k_global / 2
        # rectified = self.rectification_layer(k_local)
        f = self.linear(k_avg)
        print("F: ", f.reshape(self.batch_size))
        return f


