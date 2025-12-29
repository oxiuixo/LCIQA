import random
import torch.utils.data as data
import os
import os.path
import numpy as np
import scipy.io as scio
from PIL import Image
import os
import torch
import torchvision
import csv
import cv2
from torchvision import transforms

class SPQAFolder(data.Dataset):


    def __init__(self, root, index, transform):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'mos.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = np.array(float(row['MOS'])).astype(np.float32)
                mos_all.append(mos)
        labels = np.array(mos_all).astype(np.float32)
        labels = (labels - np.min(labels)) / (np.max(labels) - np.min(labels))
        sample = []
        for i, item in enumerate(index):
            sample.append((os.path.join(root, 'TestImage', imgname[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """


        path, target = self.samples[index]


        image = pil_loader(path)

        # 调整图像大小，保持宽高比不变，较小的边为512
        w, h =  image.size
        if h <= w:
            new_h = 512
            new_w = int(w * (512 / h))
        else:
            new_w = 512
            new_h = int(h * (512 / w))
        image = image.resize((new_w, new_h))
        '''
        path, target = self.samples[index]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

        # 调整图像大小，保持宽高比不变，较小的边为512
        h, w, _ = image.shape
        if h < w:
            new_h = 512
            new_w = int(w * (512 / h))
        else:
            new_w = 512
            new_h = int(h * (512 / w))
        image = cv2.resize(image, (new_w, new_h))

        # 转换为PIL Image以便使用torchvision.transforms
        image = transforms.ToPILImage()(image)
        '''
        sample = self.transform(image)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class Koniq_10kFolder(data.Dataset):

    def __init__(self, root, index, transform):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = np.array(float(row['MOS_zscore'])).astype(np.float32)
                mos_all.append(mos)
        labels = np.array(mos_all).astype(np.float32)
        labels = (labels - np.min(labels)) / (np.max(labels) - np.min(labels))
        sample = []
        for ind in index:
            sample.append((os.path.join(root, '1024x768', imgname[ind]), labels[ind]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class DataLoaderIQA(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, img_indx, batch_size=1, istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain

        if dataset == 'koniq':
            if istrain:
                transforms = torchvision.transforms.Compose([
                    #torchvision.transforms.Resize((512, 768)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomCrop((512, 768)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                     std=(0.5, 0.5, 0.5))])
            else:
                transforms = torchvision.transforms.Compose([
                    #torchvision.transforms.Resize((512, 768)),
                    torchvision.transforms.RandomCrop((512, 768)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                     std=(0.5, 0.5, 0.5))])

        if dataset == 'spaq':
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomCrop((512, 512)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                     std=(0.5, 0.5, 0.5))])
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.CenterCrop((512, 512)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                     std=(0.5, 0.5, 0.5))])

        if dataset == 'koniq':
            self.data = Koniq_10kFolder(root=path, index=img_indx, transform=transforms)
        if dataset == 'spaq':
            self.data = SPQAFolder(root=path, index=img_indx, transform=transforms)


    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=12) #Shuffle 改成False了
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=12)
        return dataloader



