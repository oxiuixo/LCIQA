import torch
from torch import nn
import numpy as np


class SEBlock(nn.Module):
    def __init__(self, input_channels, reduced_dim):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, reduced_dim, 1),
            nn.ReLU(),
            nn.Conv2d(reduced_dim, input_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_se = self.se(x)
        return x * x_se

class QRegressionVanilla(nn.Module):  # 目前最好，到96
    def __init__(self):
        super(QRegressionVanilla, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dense1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.dense3 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.dense4 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.qmerge = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, ft1, ft2, ft3, ft4):
        qft4_min = self.avg_pool(ft4).view(ft4.size(0), -1)
        #qft4_max = self.max_pool(ft4).view(ft4.size(0), -1)
        #qvec4 = torch.cat((qft4_min, qft4_max), dim=1)
        qvec4 = self.dense4(qft4_min)
        qsc = self.qmerge(qvec4)
        return qsc

class QRegressionSepA(nn.Module):  # 目前最好，到96
    def __init__(self):
        super(QRegressionSepA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.qconv1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, groups=512),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(),
            SEBlock(input_channels=512, reduced_dim=128),

            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
        )
        self.qconv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, groups=512),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(),
            SEBlock(input_channels=512, reduced_dim=128),

            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
        )
        self.qconv3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU6(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, groups=512),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(),
            SEBlock(input_channels=512, reduced_dim=128),

            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
        )
        self.qconv4 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU6(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, groups=512),
            nn.BatchNorm2d(512),
            nn.ReLU6(),
            SEBlock(input_channels=512, reduced_dim=128),
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
        )
        self.qdense1=nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.qdense2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.qdense3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.qdense4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.merg_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False, groups=256),
            nn.BatchNorm2d(256),
            nn.ReLU6(),
            SEBlock(input_channels=256, reduced_dim=128),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU6(),
        )
        self.qdense = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, ft1, ft2, ft3, ft4):
        qft1, qft2, qft3, qft4 = self.qconv1(ft1), self.qconv2(ft2), self.qconv3(ft3), self.qconv4(ft4)
        qft1min = self.avg_pool(qft1).view(qft1.size(0), -1)
        qft1max = self.max_pool(qft1).view(qft1.size(0), -1)
        qft2min = self.avg_pool(qft2).view(qft2.size(0), -1)
        qft2max = self.max_pool(qft2).view(qft2.size(0), -1)
        qft3min = self.avg_pool(qft3).view(qft3.size(0), -1)
        qft3max = self.max_pool(qft3).view(qft3.size(0), -1)
        qft4min = self.avg_pool(qft4).view(qft4.size(0), -1)
        qft4max = self.max_pool(qft4).view(qft4.size(0), -1)

        qvec1 = torch.cat((qft1min, qft1max), dim=1)
        qvec2 = torch.cat((qft2min, qft2max), dim=1)
        qvec3 = torch.cat((qft3min, qft3max), dim=1)
        qvec4 = torch.cat((qft4min, qft4max), dim=1)

        qsc1 = self.qdense1(qvec1)
        qsc2 = self.qdense2(qvec2)
        qsc3 = self.qdense3(qvec3)
        qsc4 = self.qdense4(qvec4)

        qft = self.merg_conv(torch.cat((qft1.detach(), qft2.detach(), qft3.detach(), qft4.detach()), dim=1))
        qft_min = self.avg_pool(qft).view(qft.size(0), -1)
        qft_max = self.max_pool(qft).view(qft.size(0), -1)
        qvec = torch.cat((qft_min, qft_max), dim=1)
        qsc = self.qdense(qvec)
        return qsc, qsc1, qsc2, qsc3, qsc4


class QRegressionSep(nn.Module):  # 目前最好，到96
    def __init__(self):
        super(QRegressionSep, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.qconv1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, groups=512),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(),
            SEBlock(input_channels=512, reduced_dim=128),

            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
        )
        self.qconv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, groups=512),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(),
            SEBlock(input_channels=512, reduced_dim=128),

            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
        )
        self.qconv3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU6(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, groups=512),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(),
            SEBlock(input_channels=512, reduced_dim=128),

            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
        )
        self.qconv4 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU6(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, groups=512),
            nn.BatchNorm2d(512),
            nn.ReLU6(),
            SEBlock(input_channels=512, reduced_dim=128),
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(),
        )
        self.qdense1=nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.qdense1b = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.qdense2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.qdense2b = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.qdense3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.qdense3b = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.qdense4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.qdense4b = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        #self.merg_conv = nn.Sequential(
        #    nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
        #    nn.BatchNorm2d(256),
        #    nn.ReLU6(),
        #    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False, groups=256),
        #    nn.BatchNorm2d(256),
        #    nn.ReLU6(),
        #    SEBlock(input_channels=256, reduced_dim=128),
        #    nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
        #    nn.BatchNorm2d(512),
        #    nn.ReLU6(),
        #)
        self.qdense = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, ft1, ft2, ft3, ft4):
        qft1, qft2, qft3, qft4 = self.qconv1(ft1), self.qconv2(ft2), self.qconv3(ft3), self.qconv4(ft4)
        qft1min = self.avg_pool(qft1).view(qft1.size(0), -1)
        qft1max = self.max_pool(qft1).view(qft1.size(0), -1)
        qft2min = self.avg_pool(qft2).view(qft2.size(0), -1)
        qft2max = self.max_pool(qft2).view(qft2.size(0), -1)
        qft3min = self.avg_pool(qft3).view(qft3.size(0), -1)
        qft3max = self.max_pool(qft3).view(qft3.size(0), -1)
        qft4min = self.avg_pool(qft4).view(qft4.size(0), -1)
        qft4max = self.max_pool(qft4).view(qft4.size(0), -1)

        qvec1 = torch.cat((qft1min, qft1max), dim=1)
        qvec2 = torch.cat((qft2min, qft2max), dim=1)
        qvec3 = torch.cat((qft3min, qft3max), dim=1)
        qvec4 = torch.cat((qft4min, qft4max), dim=1)

        qembed1 = self.qdense1(qvec1)
        qembed2 = self.qdense2(qvec2)
        qembed3 = self.qdense3(qvec3)
        qembed4 = self.qdense4(qvec4)

        qsc1 = self.qdense1b(qembed1)
        qsc2 = self.qdense2b(qembed2)
        qsc3 = self.qdense3b(qembed3)
        qsc4 = self.qdense4b(qembed4)

        qvec = (torch.cat((qembed1.detach(), qembed2.detach(), qembed3.detach(), qembed4.detach()), dim=1))
        #qft_min = self.avg_pool(qft).view(qft.size(0), -1)
        #qft_max = self.max_pool(qft).view(qft.size(0), -1)
        #qvec = torch.cat((qft_min, qft_max), dim=1)
        qsc = self.qdense(qvec)

        return qsc, qsc1, qsc2, qsc3, qsc4
