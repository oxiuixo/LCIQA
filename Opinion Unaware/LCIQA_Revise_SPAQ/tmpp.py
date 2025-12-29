import random
import os
import scipy.io
from tqdm import tqdm
import numpy as np
from utils.dataloaders import DataLoaderIQA
from models.LCIQA import QRegressionSep
from models.qadaptors import UpScaleNet4Res
from models.resnet_backbone import resnet50_backbone
import torch
from utils.iqa_utils import Config, PLCC_Loss
from scipy import stats
import torch.nn.functional as F

def DataSetup(root, batch_size, data_lens,trainPCT):
    scn_idxs = [x for x in range(data_lens)]
    np.random.shuffle(scn_idxs)
    scn_idxs_train = scn_idxs[:int(trainPCT * data_lens)]
    scn_idxs_test = scn_idxs[int(trainPCT * data_lens):]
    loader_train = DataLoaderIQA('spaq', root, scn_idxs_train, batch_size=batch_size, istrain=True).get_data()
    loader_test = DataLoaderIQA('spaq', root, scn_idxs_test, batch_size=batch_size, istrain=False).get_data()
    return loader_train, loader_test

import torch.nn as nn

class Adaptor(nn.Module):
    def __init__(self):
        super(Adaptor, self).__init__()
        self.k = 3
    def forward(self, s1, s2, s3, r1, r2, r3):

        d1, d2, d3 = (s1 - r1).abs(), (s2 - r2).abs(), (s3 - r3).abs()

        ED1, ED2, ED3 = torch.mean(torch.pow(d1, self.k), dim=1, keepdim=True), \
                        torch.mean(torch.pow(d2, self.k), dim=1, keepdim=True), \
                        torch.mean(torch.pow(d3, self.k), dim=1, keepdim=True)
        ED1, ED2, ED3 = torch.pow(ED1, 1 / self.k), torch.pow(ED2, 1 / self.k), torch.pow(ED3, 1 / self.k)
        ED1W = ED1 / (ED1.mean(dim=(2,3), keepdim=True))  # 距离权重，重建差别越大权重越大
        ED2W = ED2 / (ED2.mean(dim=(2,3), keepdim=True))
        ED3W = ED3 / (ED3.mean(dim=(2,3), keepdim=True))
        s1_new, s2_new, s3_new = s1 * ED1W, s2 * ED2W, s3 * ED3W  # B N D * B N 1给原特征中不好重建的地方大的权重
        return s1_new, s2_new, s3_new

def test_model(models, loaders, config, cnt):
    torch.cuda.empty_cache()
    my_device = torch.device('cuda:0')
    models.bb.train(False)
    models.rr.train(False)
    models.bb.eval()
    models.rr.eval()
    #C = 0.1
    # Extract Features and test
    mos_vals = np.empty((0, 1))
    pred1_vals = np.empty((0, 1))
    pred2_vals = np.empty((0, 1))
    pred3_vals = np.empty((0, 1))
    pred4_vals = np.empty((0, 1))
    predM_vals = np.empty((0, 1))
    test_time = 20
    for cnt in range(test_time):
        with torch.no_grad():
            for inputs, labels in loaders.test:
                inputs, labels = inputs.float().to(my_device), labels.float().to(my_device)
                bb_fts = models.bb(inputs)
                s1, s2, s3, s4 = bb_fts['f1'], bb_fts['f2'], bb_fts['f3'], bb_fts['f4']
                r1, r2, r3 = models.up2to1(s2), models.up3to2(s3), models.up4to3(s4)
                s1n, s2n, s3n = models.adp(s1, s2, s3, r1, r2, r3)

                predM, pred1, pred2, pred3, pred4 = models.rr(s1n, s2n, s3n, s4)
                if cnt == 0:
                    mos_vals = np.append(mos_vals, labels[:, None].detach().cpu().numpy(), axis=0)
                # pred1_vals = np.append(pred1_vals, pred1.detach().cpu().numpy(), axis=0)
                # pred2_vals = np.append(pred2_vals, pred2.detach().cpu().numpy(), axis=0)
                # pred3_vals = np.append(pred3_vals, pred3.detach().cpu().numpy(), axis=0)
                # pred4_vals = np.append(pred4_vals, pred4.detach().cpu().numpy(), axis=0)
                predM_vals = np.append(predM_vals, predM.detach().cpu().numpy(), axis=0)

    Len, _ = predM_vals.shape
    predM_vals_R = predM_vals.reshape((test_time, Len//test_time))
    predM_Mean = predM_vals_R.mean(axis=0)


    scipy.io.savemat('./results/test_gt_%s_cnt%02d.mat' % (config.type, cnt), {'gt': mos_vals})
    scipy.io.savemat('./results/test_pred_%s_cnt%02d.mat' % (config.type, cnt), {'pred': predM_vals})

    #srcc_val1, _ = stats.spearmanr(pred1_vals.squeeze(), mos_vals.squeeze())
    #plcc_val1, _ = stats.pearsonr(pred1_vals.squeeze(), mos_vals.squeeze())

    #srcc_val2, _ = stats.spearmanr(pred2_vals.squeeze(), mos_vals.squeeze())
    #plcc_val2, _ = stats.pearsonr(pred2_vals.squeeze(), mos_vals.squeeze())

    #srcc_val3, _ = stats.spearmanr(pred3_vals.squeeze(), mos_vals.squeeze())
    #plcc_val3, _ = stats.pearsonr(pred3_vals.squeeze(), mos_vals.squeeze())

    #srcc_val4, _ = stats.spearmanr(pred4_vals.squeeze(), mos_vals.squeeze())
    #plcc_val4, _ = stats.pearsonr(pred4_vals.squeeze(), mos_vals.squeeze())

    srcc_valM, _ = stats.spearmanr(predM_Mean.squeeze(), mos_vals.squeeze())
    plcc_valM, _ = stats.pearsonr(predM_Mean.squeeze(), mos_vals.squeeze())
    srcc_val1, plcc_val1, srcc_val2, plcc_val2, srcc_val3, plcc_val3, srcc_val4, plcc_val4 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return srcc_val1, plcc_val1, srcc_val2, plcc_val2, srcc_val3, plcc_val3, srcc_val4, plcc_val4, srcc_valM, plcc_valM

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
            s1, s2, s3, s4 = bb_fts['f1'], bb_fts['f2'], bb_fts['f3'], bb_fts['f4']
            r1, r2, r3 = models.up2to1(s2), models.up3to2(s3), models.up4to3(s4)
            s1n, s2n, s3n = models.adp(s1, s2, s3, r1, r2, r3)

            predM, pred1, pred2, pred3, pred4 = models.rr(s1n, s2n, s3n, s4)

            lossQ = optims.c1(predM.squeeze(), labels.detach().squeeze()) + \
                    optims.c2(predM.squeeze(), labels.detach().squeeze()) + \
                    0.5 * optims.c1(pred1.squeeze(), labels.detach().squeeze()) + \
                    0.5 * optims.c1(pred2.squeeze(), labels.detach().squeeze()) + \
                    0.5 * optims.c1(pred3.squeeze(), labels.detach().squeeze()) + \
                    0.5 * optims.c1(pred4.squeeze(), labels.detach().squeeze()) + \
                    0.5 * optims.c2(pred1.squeeze(), labels.detach().squeeze()) + \
                    0.5 * optims.c2(pred2.squeeze(), labels.detach().squeeze()) + \
                    0.5 * optims.c2(pred3.squeeze(), labels.detach().squeeze()) + \
                    0.5 * optims.c2(pred4.squeeze(), labels.detach().squeeze())
            optims.optimQ.zero_grad()
            lossQ.backward()
            optims.optimQ.step()
            optims.schedQ.step()

            epoch_loss.append(lossQ.item())
            pred_vals = np.append(pred_vals, predM.detach().cpu().numpy(), axis=0)
            gt_vals = np.append(gt_vals, labels[:, None].detach().cpu().numpy(), axis=0)

        print('testing....')
        srcc_val_t, _ = stats.spearmanr(pred_vals.squeeze(), gt_vals.squeeze())
        plcc_val_t, _ = stats.pearsonr(pred_vals.squeeze(), gt_vals.squeeze())

        srcc_val1, plcc_val1, srcc_val2, plcc_val2, srcc_val3, plcc_val3, srcc_val4, plcc_val4, srcc_valM, plcc_valM = test_model(models, loaders, config, t)
        srccs.append(srcc_valM)
        plccs.append(plcc_valM)

        print(
            f'EP:{t:03d}| TrainSRCC: {srcc_val_t:.3f} TrainPLCC: {plcc_val_t:.3f} | TestSRCC1: {srcc_val1: .3f} TestPLCC1: {plcc_val1: .3f}'
        + f'TestSRCC2: {srcc_val2: .3f} TestPLCC2: {plcc_val2: .3f} TestSRCC3: {srcc_val3: .3f} TestPLCC3: {plcc_val3: .3f}'
        + f'TestSRCC4: {srcc_val4: .3f} TestPLCC4: {plcc_val4: .3f} TestSRCCM: {srcc_valM: .3f} TestPLCCM: {plcc_valM: .3f}')
        #+ f'TestSRCCMB: {srcc_valMB: .3f} TestPLCCMB: {plcc_valMB: .3f} TestSRCCM: {srcc_valM: .3f} TestPLCCM: {plcc_valM: .3f}')

    print(f'MaxSRCC: {max(srccs) : .3f} MaxPLCC: {max(plccs) : .3f}\n')



def getHyperParams(pct,sd):
    myconfigs = {
        'lr': 1e-4,
        'weight_decay': 5e-4,
        'T_MAX': 50,
        'eta_min': 0,
        'nepoch': 50,
        'batch_size': 12,
        'data_lens': 11125,
        'roots': r'/root/IQADatasets/SPAQ',
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
    myconfig = getHyperParams(trainPCT,myseed)

    #model_urls = {
    #    'UP2to1': r'.\weights\QAdaptor\UP2to1_00.pth',
    #    'UP3to2': r'.\weights\QAdaptor\UP3to2_00.pth',
    #    'UP4to3': r'.\weights\QAdaptor\UP4to3_00.pth',
    #}
    model_urls = {
        'UP2to1': r'./weights/QAdaptor/UP2to1_00.pth',
        'UP3to2': r'./weights/QAdaptor/UP3to2_00.pth',
        'UP4to3': r'./weights/QAdaptor/UP4to3_00.pth',
    }

    mres = resnet50_backbone(pretrained=True).cuda()
    mres.eval()
    madp = Adaptor().cuda()
    madp.train(False)
    madp.eval()

    regress_net = QRegressionSep().cuda()
    regress_net.train(True)

    upnet2to1 = UpScaleNet4Res(in_CH=512, out_CH=256)
    upnet2to1.load_state_dict(torch.load(model_urls['UP2to1'], map_location='cpu'))
    upnet2to1 = upnet2to1.cuda()
    upnet2to1.eval()

    upnet3to2 = UpScaleNet4Res(in_CH=1024, out_CH=512)
    upnet3to2.load_state_dict(torch.load(model_urls['UP3to2'], map_location='cpu'))
    upnet3to2 = upnet3to2.cuda()
    upnet3to2.eval()

    upnet4to3 = UpScaleNet4Res(in_CH=2048, out_CH=1024)
    upnet4to3.load_state_dict(torch.load(model_urls['UP4to3'], map_location='cpu'))
    upnet4to3 = upnet4to3.cuda()
    upnet4to3.eval()

    paramQ = [
        {'params': regress_net.parameters(), 'lr': myconfig.lr},
    ]

    optimQ = torch.optim.Adam(paramQ, weight_decay=myconfig.weight_decay)
    schedQ = torch.optim.lr_scheduler.CosineAnnealingLR(optimQ, myconfig.T_MAX, myconfig.eta_min)

    train_loader, test_loader = DataSetup(myconfig.roots, myconfig.batch_size, myconfig.data_lens,trainPCT)


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
        'up2to1': upnet2to1,
        'up3to2': upnet3to2,
        'up4to3': upnet4to3,
        'adp': madp
    })

    data_loaders = Config({'train': train_loader,
                           'test': test_loader})

    train_model(models_params, data_loaders, optim_params, myconfig)

    print('OK..')


import argparse
if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--trainpct', dest='pct', type=float, default=0.8, help='training percent')
    #parser.add_argument('--seed', dest='sd', type=int, default=0, help='random seed')
    #pcfg = parser.parse_args()
    #cProfile.run('main()', 'time_analy.prof')
    main()
