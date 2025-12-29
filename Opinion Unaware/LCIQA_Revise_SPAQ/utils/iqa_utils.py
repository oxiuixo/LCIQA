import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import random
import torch.nn as nn


class Hybrid_Loss(nn.Module):
    def __init__(self, threshold=1.0):
        super(Hybrid_Loss, self).__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, input, target):
        # 计算绝对误差
        abs_error = torch.abs(input - target)

        # 创建掩码以区分大于或小于阈值的误差
        mask = abs_error > self.threshold

        # 对于小于阈值的部分使用MSE
        mse_loss = self.mse(input[~mask], target[~mask]) if (~mask).any() else 0.0

        # 对于大于阈值的部分使用L1
        l1_loss = self.l1(input[mask], target[mask]) if mask.any() else 0.0

        # 结合两部分的损失
        loss = mse_loss + l1_loss
        return loss

class PLCC_Loss(nn.Module):
    def __init__(self):
        super(PLCC_Loss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred_mean = torch.mean(y_pred)
        y_true_mean = torch.mean(y_true)

        sum_covariance = torch.sum((y_pred - y_pred_mean) * (y_true - y_true_mean))
        sum_variance_pred = torch.sum((y_pred - y_pred_mean) ** 2)
        sum_variance_true = torch.sum((y_true - y_true_mean) ** 2)

        plcc = sum_covariance / torch.sqrt(sum_variance_pred * sum_variance_true)
        return 1 - plcc  # We minimize this loss to maximize PLCC

class Config(dict):

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = Config(value)
        return value

def getFineLoss(R_pred, G_pred, P_pred):
    diff1 = G_pred - R_pred
    diff1[diff1 < 0] = 0
    lossProjRG = diff1.mean()
    diff2 = P_pred - G_pred
    diff2[diff2 < 0] = 0
    lossProjGP = diff2.mean()
    loss = lossProjRG + lossProjGP
    return loss

def getCoarseLoss(q_ft, config):
    q_ft = q_ft / q_ft.norm(dim=1, keepdim=True)
    q_mat = q_ft @ q_ft.t()
    deltMat = -1 * q_mat
    deltMat[deltMat < 0] = 0

    q_matSS, q_matGG, q_matPP, q_matSG, q_matSP, q_matGP = q_mat[:config.batch_size, :config.batch_size], \
        q_mat[config.batch_size:2 * config.batch_size, config.batch_size:2 * config.batch_size], \
        q_mat[2 * config.batch_size:, 2 * config.batch_size:], \
        q_mat[:config.batch_size, config.batch_size:2 * config.batch_size], \
        q_mat[:config.batch_size, 2 * config.batch_size:], \
        q_mat[config.batch_size:2 * config.batch_size, 2 * config.batch_size:]

    deltSS = np.cos(config.THa) - q_matSS
    deltGG = np.cos(config.THb) - q_matGG
    deltPP = np.cos(config.THr) - q_matPP
    if config.THa +config.THr <= np.pi /2.0:
        deltSP = q_matSP - np.sin(config.THa + config.THr)
    else:
        deltSP = torch.zeros_like(q_matSP)

    deltSS[deltSS < 0] = 0
    deltGG[deltGG < 0] = 0
    deltPP[deltPP < 0] = 0
    deltSP[deltSP < 0] = 0

    loss = torch.mean(deltSS + deltGG + deltPP + deltSP )  + torch.mean(deltMat) * 0.1

    return  loss

