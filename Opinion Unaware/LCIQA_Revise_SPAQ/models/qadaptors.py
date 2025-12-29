import torch as torch
import torch.nn as nn

class UpScaleNet(nn.Module):
    def __init__(self, in_CH, out_CH):
        super(UpScaleNet, self).__init__()
        self.upLayer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=in_CH, out_channels=in_CH // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_CH // 2),
            nn.PReLU(in_CH // 2),
            nn.Conv2d(in_channels=in_CH // 2, out_channels=in_CH // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_CH // 2),
            nn.PReLU(in_CH // 2),
            nn.Conv2d(in_channels=in_CH // 2, out_channels=out_CH, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, ft):
        ft_rec = self.upLayer(ft)
        return ft_rec

class UpScaleNet4Res(nn.Module):
    def __init__(self, in_CH, out_CH):
        super(UpScaleNet4Res, self).__init__()
        self.upLayer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=in_CH, out_channels=in_CH // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_CH // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_CH // 2, out_channels=out_CH, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_CH),
            nn.ReLU(),
        )

    def forward(self, ft):
        ft_rec = self.upLayer(ft)
        return ft_rec