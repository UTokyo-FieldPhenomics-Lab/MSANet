import os
import random
from scipy import spatial
import networkx as nx

import skimage
from skimage import filters
from skimage.morphology import ellipse

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import imageio.v3 as iio
import glob
import scipy.io as io
import ast
from pathlib import Path

class Soy(Dataset):
    def __init__(self, data_root, transform=None, train=False):
        self.root_path = data_root
        if train:
            images_path = f'{data_root}/images_a'
        else:
            images_path = f'{data_root}/images_b'

        
        pa = Path(images_path)
        self.img_list = []
        self.label_list = []

        for child in pa.glob('*.png'):
            filepath = str(child.resolve())
            self.img_list.append(filepath)

            label_path = filepath.replace('images_', 'labels_').replace('.png', '.txt')
            self.label_list.append(label_path)

        self.nSamples = len(self.img_list)
        
        self.transform = transform
        self.train = train


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.label_list[index]
        # 
        img, point = load_data(img_path, gt_path)
        h, w, _ = img.shape
        target = {
                'name': img_path.split('/')[-1],
                'height': h,
                'width': w
            }
        #
        if self.train:
            mask = np.zeros_like(img)[... ,0]
            # print(mask.shape)
            
            point_int = np.trunc(point).astype(np.int32)
            # h, w, _ = img.shape
            x = point_int[:, 0]
            y = point_int[:, 1]
            # print(x.max(), y.max())
            mask[y, x] = 1

            mask = skimage.morphology.dilation(mask, ellipse(5, 5))
            mask = filters.gaussian(mask,sigma=5, mode='nearest',preserve_range=True, truncate=1)
            mask[mask>0.9] = 1

            
            if self.transform:
                transformed = self.transform(image=img, mask=mask)
                # mask = self.transforms(mask)
                img = transformed['image']
                mask = transformed['mask']

            return img, mask, target
        
        else:
            h, w, _ = img.shape

            mask = np.zeros_like(img)[... ,0]
            point_int = np.trunc(point).astype(np.int32)
            # h, w, _ = img.shape
            x = point_int[:, 0]
            y = point_int[:, 1]
            # print(x.max(), y.max())
            mask[y, x] = 1

            mask = skimage.morphology.dilation(mask, ellipse(5, 5))
            mask = filters.gaussian(mask,sigma=5, mode='nearest',preserve_range=True, truncate=1)
            mask[mask>0.9] = 1
            if self.transform:
                transformed = self.transform(image=img, mask=mask)
                mask = transformed['mask']
                img = transformed['image']
            return img, point, mask, target
        #  need to adapt your own image names
        # target = {}

        # target['point'] = torch.Tensor(point)
        # target['mask'] = mask

        


def load_data(img_path, gt_path):
    # load the images
    img_np = iio.imread(img_path)

    w, h, _ = img_np.shape
    coords = []
    with open(gt_path, 'r') as f:
        for line in f:
           x, y = ast.literal_eval(line[:-1])
           if not ((x<0) | (y<0) | (x>h) | (y>w)):
               coords.append([x, y])
    coords_np = np.array(coords)
    coords_round = np.trunc(coords_np).astype(np.int32)

    return img_np, coords_round

