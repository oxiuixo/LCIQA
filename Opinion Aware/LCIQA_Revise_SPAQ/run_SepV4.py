
import random
import os
import scipy.io
from tqdm import tqdm
import numpy as np
from utils.dataloaders import DataLoaderIQA
from models.LCIQA import QSep
from models.qadaptors import UpScaleNet4Res
from models.resnet_backbone import resnet50_backbone
import torch
from utils.iqa_utils import Config, PLCC_Loss
from scipy import stats
import torch.nn.functional as F


def DataSetup(root, batch_size, data_lens, trainPCT):
    scn_idxs = [x for x in range(data_lens)]
    np.random.shuffle(scn_idxs)
    scn_idxs_train = scn_idxs[:int(trainPCT * data_lens)]
    scn_idxs_test = scn_idxs[int(trainPCT * data_lens):]
    loader_train = DataLoaderIQA('spaq', root, scn_idxs_train, batch_size=batch_size, istrain=True).get_data()
    loader_test = DataLoaderIQA('spaq', root, scn_idxs_test, batch_size=batch_size, istrain=False).get_data()
    return loader_train, loader_test


import torch.nn as nn


class AdaptorSingle(nn.Module):
    def __init__(self):
        super(AdaptorSingle, self).__init__()
        self.k = 3

    def forward(self, s1, r1):
        d1 = (s1 - r1).abs()

        ED1 = torch.mean(torch.pow(d1, self.k), dim=1, keepdim=True)
        ED1 = torch.pow(ED1, 1 / self.k)
        ED1W = ED1 / (ED1.mean(dim=(2, 3), keepdim=True))  # 距离权重，重建差别越大权重越大
        s1_new = s1 * ED1W  # B N D * B N 1给原特征中不好重建的地方大的权重
        return s1_new


def test_model(models, loaders, config, cnt):
    torch.cuda.empty_cache()
    my_device = torch.device('cuda:0')
    models.bb.train(False)
    models.rr.train(False)
    models.bb.eval()
    models.rr.eval()
    mos_vals = np.empty((0, 1))
    predM_vals = np.empty((0, 1))
    test_time = 1
    for cnt in range(test_time):
        with torch.no_grad():
            for inputs, labels in loaders.test:
                inputs, labels = inputs.float().to(my_device), labels.float().to(my_device)
                bb_fts = models.bb(inputs)
                san = bb_fts['f4']
                predM = models.rr(san)
                if cnt == 0:
                    mos_vals = np.append(mos_vals, labels[:, None].detach().cpu().numpy(), axis=0)
                predM_vals = np.append(predM_vals, predM.detach().cpu().numpy(), axis=0)

    Len, _ = predM_vals.shape
    predM_vals_R = predM_vals.reshape((test_time, Len // test_time))
    predM_Mean = predM_vals_R.mean(axis=0)

    scipy.io.savemat('./results/test_gt_%s_cnt%02d.mat' % (config.type, cnt), {'gt': mos_vals})
    scipy.io.savemat('./results/test_pred_%s_cnt%02d.mat' % (config.type, cnt), {'pred': predM_vals})

    srcc_valM, _ = stats.spearmanr(predM_Mean.squeeze(), mos_vals.squeeze())
    plcc_valM, _ = stats.pearsonr(predM_Mean.squeeze(), mos_vals.squeeze())
    return srcc_valM, plcc_valM


def train_model(models, loaders, optims, config):
    torch.cuda.empty_cache()
    models.bb.train(False)
    models.bb.eval()
    models.rr.train(True)
    my_device = torch.device('cuda:0')
    srccs, plccs = [], []
    for t in range(config.nepoch):
        pred_vals = np.empty((0, 1))
        gt_vals = np.empty((0, 1))
        epoch_loss = []
        models.bb.train(False)
        models.bb.eval()
        models.rr.train(True)
        for inputs, labels in tqdm(loaders.train):
            inputs, labels = inputs.float().to(my_device), labels.float().to(my_device)
            bb_fts = models.bb(inputs)
            san = bb_fts['f4']
            pred = models.rr(san)

            lossQ = optims.c1(pred.squeeze(), labels.detach().squeeze()) + \
                    optims.c2(pred.squeeze(), labels.detach().squeeze())
            optims.optimQ.zero_grad()
            lossQ.backward()
            optims.optimQ.step()
            optims.schedQ.step()

            epoch_loss.append(lossQ.item())
            pred_vals = np.append(pred_vals, pred.detach().cpu().numpy(), axis=0)
            gt_vals = np.append(gt_vals, labels[:, None].detach().cpu().numpy(), axis=0)

        print('testing....')
        srcc_val_t, _ = stats.spearmanr(pred_vals.squeeze(), gt_vals.squeeze())
        plcc_val_t, _ = stats.pearsonr(pred_vals.squeeze(), gt_vals.squeeze())
        torch.save(models.rr.state_dict(), './weights/Reg3_%03d.pth' % t)
        srcc_valM, plcc_valM = test_model(
            models, loaders, config, t)
        srccs.append(srcc_valM)
        plccs.append(plcc_valM)

        print(
            f'EP:{t:03d}| TrainSRCC: {srcc_val_t:.3f} TrainPLCC: {plcc_val_t:.3f} | TestSRCCM: {srcc_valM: .3f} TestPLCCM: {plcc_valM: .3f}')
    print(f'MaxSRCC: {max(srccs) : .3f} MaxPLCC: {max(plccs) : .3f}\n')


def getHyperParams(pct, sd):
    myconfigs = {
        'lr': 1e-4,
        'weight_decay': 5e-4,
        'T_MAX': 50,
        'eta_min': 0,
        'nepoch': 50,
        'batch_size': 12,
        'data_lens': 11125,
        'roots': r'/root/IQADatasets/SPAQ',
        #'roots': r'E:\ImageDatabase\SPAQ',
        'type': 'LCIQA',
    }
    return Config(myconfigs)


def main():
    trainPCT = 0.8
    myseed = 0
    random.seed(myseed)
    os.environ['PYTHONHASHSEED'] = str(myseed)
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    torch.backends.cudnn.deterministic = True
    myconfig = getHyperParams(trainPCT, myseed)

    #model_urls = {
    #   'UP2to1': r'.\weights\QAdaptor\UP2to1_00.pth',
    #   'UP3to2': r'.\weights\QAdaptor\UP3to2_00.pth',
    #   'UP4to3': r'.\weights\QAdaptor\UP4to3_00.pth',
    #}
    model_urls = {
        'UP2to1': r'./weights/QAdaptor/UP2to1_00.pth',
        'UP3to2': r'./weights/QAdaptor/UP3to2_00.pth',
        'UP4to3': r'./weights/QAdaptor/UP4to3_00.pth',
    }

    mres = resnet50_backbone(pretrained=True).cuda()
    mres.eval()
    madp = AdaptorSingle().cuda()
    madp.train(False)
    madp.eval()

    regress_net = QSep().cuda()
    regress_net.train(True)


    paramQ = [
        {'params': regress_net.parameters(), 'lr': myconfig.lr},
    ]

    optimQ = torch.optim.Adam(paramQ, weight_decay=myconfig.weight_decay)
    schedQ = torch.optim.lr_scheduler.CosineAnnealingLR(optimQ, myconfig.T_MAX, myconfig.eta_min)

    train_loader, test_loader = DataSetup(myconfig.roots, myconfig.batch_size, myconfig.data_lens, trainPCT)

    criterion1 = torch.nn.MSELoss()
    criterion2 = PLCC_Loss()
    optim_params = Config({'c1': criterion1,
                           'c2': criterion2,
                           'optimQ': optimQ,
                           'schedQ': schedQ,
                           })
    models_params = Config({
        'bb': mres,
        'rr': regress_net,
    })

    data_loaders = Config({'train': train_loader,
                           'test': test_loader})

    train_model(models_params, data_loaders, optim_params, myconfig)

    print('OK..')

if __name__ == '__main__':
    main()


