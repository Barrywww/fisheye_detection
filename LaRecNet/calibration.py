import math
import torch
import torch.nn as nn


class CalibrationGlobal(nn.Module):
    def __init__(self, batch_size):
        super(CalibrationGlobal, self).__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(6)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = x.reshape(self.batch_size, 1024)
        x = self.fc2(x)
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)

        return x


class CalibrationLocal(nn.Module):
    def __init__(self, corner_side, ctr_side, batch_size):
        super(CalibrationLocal, self).__init__()
        self.batch_size = batch_size
        self.corner_side = corner_side
        self.ctr_side = ctr_side

        self.pool_tl = nn.MaxPool2d(corner_side)
        self.pool_tr = nn.MaxPool2d(corner_side)
        self.pool_bl = nn.MaxPool2d(corner_side)
        self.pool_br = nn.MaxPool2d(corner_side)
        self.pool_ctr = nn.MaxPool2d(ctr_side)

        self.fc_s1 = nn.Linear(2048, 1024)
        self.fc_s2 = nn.Linear(1024, 512)
        self.fc_s3 = nn.Linear(512, 9)
        self.filter = nn.Linear(9, 5)

    def forward(self, x):
        # print(x.shape)
        tl = x[:, :, :self.corner_side, :self.corner_side]
        tr = x[:, :, self.corner_side:, -self.corner_side:]
        bl = x[:, :, -self.corner_side:, :self.corner_side]
        br = x[:, :, -self.corner_side:, -self.corner_side:]
        ctr_point = x.shape[2]//2
        ctr = x[:, :, ctr_point - self.ctr_side//2:ctr_point + self.ctr_side//2,
              ctr_point - self.ctr_side//2:ctr_point + self.ctr_side//2]

        x_shape = x.shape[1]
        # del x

        tl = self.pool_tl(tl)
        tr = self.pool_tr(tr)
        bl = self.pool_bl(bl)
        br = self.pool_br(br)
        ctr = self.pool_ctr(ctr)
        # print(tl.shape, tr.shape, bl.shape, br.shape, ctr.shape)
        # ftr_concat = torch.cat([tl, tr, bl, br, ctr])
        ftr_concat = (tl+tr+bl+br+ctr) / 5
        # print(ftr_concat.shape)
        ftr_concat = torch.reshape(ftr_concat, (self.batch_size, x_shape))

        out = self.fc_s1(ftr_concat)
        out = self.fc_s2(out)
        out = self.fc_s3(out)
        out = self.filter(out)
        # print(out.shape)

        return out



