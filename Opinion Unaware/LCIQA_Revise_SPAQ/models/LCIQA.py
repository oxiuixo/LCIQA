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

class MultiLayerMerge(nn.Module):
    def __init__(self):
        super(MultiLayerMerge, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=128*4, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.merge = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=1024, dropout=0.1)
        self.qdense = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, qmerge):
        qmerge = self.conv(qmerge)
        qmerge = qmerge.view(qmerge.size(0), qmerge.size(1), -1) # B C H W => B C HW
        qmerge = qmerge.permute(2, 0, 1) # B C HW => HW B C
        qmerge = self.merge(qmerge)
        q_max, _ = qmerge.max(dim=0)
        q_mean = qmerge.mean(dim=0)
        qft = torch.cat((q_mean, q_max), dim=1)
        qsc = self.qdense(qft)
        return qsc

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
        self.merge = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, ft1, ft2, ft3, ft4):
        qft1, qft2, qft3, qft4 = self.qconv1(ft1), self.qconv2(ft2), self.qconv3(ft3), self.qconv4(ft4)
        qmerge = (torch.cat((qft1, qft2, qft3, qft4), dim=1))
        qft = self.merge(qmerge)
        qftA = self.avg_pool(qft).view(qft.size(0), -1)
        qftB = self.max_pool(qft).view(qft.size(0), -1)
        qft = torch.cat((qftA, qftB), dim=1)
        qsc = self.dense(qft)
        return qsc

def getIQAConv_Dense(in_ch=256, out_ch=128, down=[True, True, True]):
    layers = [nn.Conv2d(in_ch, 512, kernel_size=1, stride=1, padding=0, bias=False)]
    if down[0]:
        layers.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
    layers.extend([
        nn.BatchNorm2d(512),
        nn.ReLU6(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, groups=512),
        ])
    if down[1]:
        layers.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
    layers.extend([
        nn.BatchNorm2d(512),
        nn.ReLU6(),
        SEBlock(input_channels=512, reduced_dim=128),
        nn.Conv2d(512, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
    ])
    if down[2]:
        layers.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
    layers.extend([
        nn.BatchNorm2d(128),
        nn.ReLU6(),
    ])
    qdense = nn.Sequential(
        nn.Linear(out_ch*2, out_ch),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(out_ch, 1),
        nn.Sigmoid()
    )
    return nn.Sequential(*layers), qdense

class QSep(nn.Module):  # 目前最好，到96
    def __init__(self, in_ch=512, out_ch=128, down=[True, True, False]):
        super(QSep, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.qconv, self.qdense = getIQAConv_Dense(in_ch=in_ch, out_ch=out_ch, down=down)

    def forward(self, ft):
        qft = self.qconv(ft)
        qftmin = self.avg_pool(qft).view(qft.size(0), -1)
        qftmax = self.max_pool(qft).view(qft.size(0), -1)

        qsc = self.qdense(torch.cat((qftmin, qftmax), dim=1))
        return qsc, qftmin, qftmax

def getQSep(in_ch=256, out_ch=128, down=[True, True, True], pretrain_url=''):
    SingQConv = QSep(in_ch=in_ch, out_ch=out_ch, down=down)
    SingQConv.load_state_dict(torch.load(pretrain_url, map_location='cpu'))
    return SingQConv


class QMerge(nn.Module):
    def __init__(self):
        super(QMerge, self).__init__()
        self.conv1 = getQSep(in_ch=256, out_ch=128, down=[True, True, True], pretrain_url='./weights/Reg1_042.pth')
        self.conv2 = getQSep(in_ch=512, out_ch=128, down=[True, True, False], pretrain_url='./weights/Reg2_047.pth')
        self.conv3 = getQSep(in_ch=1024, out_ch=128, down=[True, False, False], pretrain_url='./weights/Reg3_019.pth')
        self.conv4 = getQSep(in_ch=2048, out_ch=128, down=[True, False, False], pretrain_url='./weights/Reg4_037.pth')
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.merg = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.att = nn.Sequential(
            nn.Linear(128 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, 4),
            nn.Sigmoid(),
        )
        self.conv1.train(False)
        self.conv2.train(False)
        self.conv3.train(False)
        self.conv4.train(False)
        self.conv1.eval()
        self.conv2.eval()
        self.conv3.eval()
        self.conv4.eval()

    def forward(self, f1, f2, f3, f4):
        _, qvec1, _ = self.conv1(f1)
        _, qvec2, _ = self.conv2(f2)
        _, qvec3, _ = self.conv3(f3)
        _, qvec4, _ = self.conv4(f4)
        qvec1 = qvec1.detach()
        qvec2 = qvec2.detach()
        qvec3 = qvec3.detach()
        qvec4 = qvec4.detach()
        qembed = torch.cat((qvec1, qvec2, qvec3, qvec4), dim=1)
        qatt = self.att(qembed)
        qatt = qatt.unsqueeze(1)
        qvec = torch.cat((qvec1.unsqueeze(2), qvec2.unsqueeze(2), qvec3.unsqueeze(2), qvec4.unsqueeze(2)), dim=2)
        qvec = qvec * qatt
        qvec = qvec.mean(dim=2, keepdim=False)
        qsc = self.merg(qvec)

        return qsc

