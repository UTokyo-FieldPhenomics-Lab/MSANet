import json
import cv2


from skimage import filters
from skimage.morphology import ellipse

import numpy as np
from torch.utils.data import Dataset
import imageio.v3 as iio

from pathlib import Path
from utils import get_points

class Soy(Dataset):
    def __init__(self, data_root, transform=None, train=False):
        self.root_path = data_root
        
        pa = Path(data_root)
        self.img_list = []
        self.label_list = []
        if train:
            for child in pa.glob('*_a*.png'):
                filepath = str(child.resolve())
                self.img_list.append(filepath)

                label_path = filepath.replace('.png', '.json')
                self.label_list.append(label_path)
        else:
            for child in pa.glob('*_b*.png'):
                filepath = str(child.resolve())
                self.img_list.append(filepath)

                label_path = filepath.replace('.png', '.json')
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
            y, x = point_int[:, 0], point_int[:, 1]
            print(y.max(), x.max())
            print(mask.shape)
            mask[y, x] = 1

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel)

            mask = filters.gaussian(mask, sigma=5, mode='nearest', preserve_range=True)
            mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
            # mask = skimage.morphology.dilation(mask, ellipse(3, 3))
            # mask = filters.gaussian(mask,sigma=5, mode='nearest',preserve_range=True, truncate=1)
            # print(np.unique(mask))
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
            y, x = point_int[:, 0], point_int[:, 1]
            # print(x.max(), y.max())
            mask[x, y] = 1

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel)
            # print(np.unique(mask))
            mask = filters.gaussian(mask,sigma=5, mode='nearest',preserve_range=True, truncate=1)
            mask[mask>0.9] = 1
            if self.transform:
                transformed = self.transform(image=img, mask=mask)
                mask = transformed['mask']
                img = transformed['image']

            return img, point_int, mask, target
        #  need to adapt your own image names
        # target = {}

        # target['point'] = torch.Tensor(point)
        # target['mask'] = mask

        


def load_data(img_path, gt_path):
    # load the images
    print(img_path, gt_path)
    img_np = iio.imread(img_path)

    coords = []
    with open(gt_path, 'r') as f:
        json_file = json.load(f)
    coords = get_points(json_file)
    coords_np = np.array(coords)
    coords_round = np.trunc(coords_np).astype(np.int32)

    return img_np, coords_round

