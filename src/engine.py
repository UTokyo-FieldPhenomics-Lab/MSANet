import os
import time
import random
import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import r2_score, mean_absolute_error
from skimage.feature import peak_local_max

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# from torchsummary import summary
from tensorboardX import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import SoyModel
from dataset import Soy
from utils import *

# from tqdm import tqdm

def denorm(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

class trainer:
    def __init__(self, args):
        self.args = args

        os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

        self.device = torch.device('cuda')

        # fix the seed for reproducibility
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # # the logger writer
        
        project_dir = f'{args.output_dir}/{args.project_name}'
        self.ckpt_dir = f'{project_dir}/ckpt'
        self.fig_dir = f'{project_dir}/figs'
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

        self.writer = SummaryWriter(project_dir)

        # get dataset
        train_set, val_set = self.get_datasets()

        # create dataloader
        self.dataloader_train = DataLoader(
            train_set,
            batch_size = args.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.dataloader_val = DataLoader(
            val_set,
            batch_size = 1,
            shuffle=False,
        )

        # others
        if self.args.resume != '':
            self.checkpoint = torch.load(args.resume)
            self.epoch = self.checkpoint['epoch']
            self.iter = self.checkpoint['iteration']
            # self.maes = self.checkpoint['maes']
            # self.r2s = self.checkpoint['r2s']
            self.best_mae = self.checkpoint['best_mae']
            self.best_mae_epoch = self.checkpoint['best_mae_epoch']

            self.best_eval_loss = self.checkpoint['best_eval_loss']
            self.best_eval_loss_epoch = self.checkpoint['best_eval_loss_epoch']
            lr_schedular_state_dict = self.checkpoint['lr_schedular_state_dict']
        else:
            self.best_mae = float('inf')
            self.best_eval_loss = float('inf')
            self.epoch = 0
            self.iter = 0
            # self.maes = []
            # self.r2s = []

        # set criterian
        self.criterian = self.get_criterian()

        # get model
        self.model = self.get_model()
        self.model.to(self.device)

        # get optimizer
        self.optim = self.get_optim()
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optim, start_factor=0.1, total_iters=5)
        if self.args.resume != '':
            self.lr_scheduler.load_state_dict(lr_schedular_state_dict)

        # print(summary(self.model, (3, 256, 256)))

    def get_optim(self):
        optim = torch.optim.Adam(self.model.parameters(),
                                 lr=self.args.lr,
                                 betas=(self.args.alpha, self.args.beta))

        if self.args.resume != '':
            optim.load_state_dict(self.checkpoint["optimizer_state_dict"])

        return optim
    
    def get_model(self):
        if self.args.resume != '':
            model = SoyModel(pretrained=False)
            model.load_state_dict(self.checkpoint["model_state_dict"])
            return model
        else:
            model = SoyModel(pretrained=True)
            # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=False, num_classes=1)
            return model

    def get_datasets(self):

        transform_train = A.Compose(
            [   
                A.ShiftScaleRotate(shift_limit=0.,
                        scale_limit=(-0.4, 0.4),
                        rotate_limit=90, p=0.5),
                # A.LongestMaxSize(max_size=512),
                A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.RandomCrop(512, 512, p=1.0),
                A.VerticalFlip(p=0.5),  
                A.Blur(p=0.5),

                A.RandomBrightness((-0.5, 0.5), p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        transform_val= A.Compose(
            [   
                A.PadIfNeeded(min_height=None, min_width=None,
                            pad_height_divisor=32, pad_width_divisor=32,
                            position=A.PadIfNeeded.PositionType.TOP_LEFT,
                            border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        # 
        train_set = Soy(self.args.data_root, train=True, transform=transform_train)
        # 
        val_set = Soy(self.args.data_root, train=False, transform=transform_val)

        return train_set, val_set

    def get_criterian(self):
        return nn.MSELoss()
        # return nn.BCELoss()
    
    def train_one_epoch(self):
        self.epoch += 1
        self.model.train()
        batch_loss = 0.
        len_train = len(self.dataloader_train)

        for img, label256, target in self.dataloader_train:
            self.iter += 1
            self.optim.zero_grad()

            img = img.to(self.device)
            label256 = torch.unsqueeze(label256, 1)
            label256 = label256.float().to(self.device)

            # print(img, label256)
            bx0, bx1, bx2, bx3, bx4 = self.model(img)
            # bx0 = self.model(img)['out']
    
            label64 = F.max_pool2d(label256, kernel_size=4).to(self.device)
            label32 = F.max_pool2d(label64, kernel_size=2).to(self.device)
            label16 = F.max_pool2d(label32, kernel_size=2).to(self.device)
            label8 = F.max_pool2d(label16, kernel_size=2).to(self.device)

            loss_256 = self.criterian(label256, bx0)
            loss_64 = self.criterian(label64, bx1)
            loss_32 = self.criterian(label32, bx2)
            loss_16 = self.criterian(label16, bx3)
            loss_8 = self.criterian(label8, bx4) 

            train_loss = loss_256 + loss_64 + loss_32 + loss_16 + loss_8
            # print(loss_256.item())
            # train_loss = loss_256
            # Update gradients
            train_loss.backward()
            print(f"iter {self.iter}, Train loss: {train_loss:.3f}")
            # Update optimizer
            self.optim.step()

            # Keep track of the losses
            batch_loss += train_loss.item()
            self.writer.add_scalars(
                'data/losses',
                {'total_loss': train_loss.item(),
                'loss_256': loss_256.item(),
                'loss_64': loss_64.item(),
                'loss_32': loss_32.item(),
                'loss_16': loss_16.item(),
                'loss_8': loss_8.item()
                },
                self.iter)
        self.lr_scheduler.step()
        self.writer.add_scalar('data/average_loss',
                               batch_loss / len_train,
                               self.epoch)
        
        return batch_loss / len_train
    
    def visualization(self):
        self.model.eval()

        img, label256, _ = next(iter(self.dataloader_train))

        img = img.to(self.device)
        label256 = torch.unsqueeze(label256, 1)
        label256 = label256.float().to(self.device)

        # print(img, label256)
        bx0, _, _, _, _ = self.model(img)

        ### Visualization code ###
        fig = plt.figure(figsize=(20,10), num=1, clear=True)

        for i in range(8):
            ax = fig.add_subplot(2, 4, i+1)
            ax.set_xticks([])
            ax.set_yticks([])
            image_tensor = denorm(img[i],
                                  mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225))
            image_tensor = image_tensor.detach().cpu().permute(1, 2, 0).squeeze().numpy()
            coords = label256[i].detach().cpu().permute(1, 2, 0).squeeze().numpy()
            preds = bx0[i].detach().cpu().permute(1, 2, 0).squeeze().numpy()

            points = np.argwhere(coords > .1)
            # print(np.unique(coords))
            pred_points  = np.argwhere(preds > .1)
            pred_center  = np.argwhere(preds > .5)
            # print('points:', points)
            x = points[:, 0]
            y = points[:, 1]
            px = pred_points[:, 0]
            py = pred_points[:, 1]
            pxc = pred_center[:, 0]
            pyc = pred_center[:, 1]

            ax.imshow(image_tensor)
            ax.plot(y, x, marker='.', color='r', ls='', alpha=.5)
            ax.plot(py, px, marker='.', color='b', ls='', alpha=.5)
            ax.plot(pyc, pxc, marker='.', color='g', ls='', alpha=.5)

        # self.writer.add_figure('figure', fig, self.epoch)

        plt.savefig(f'{self.fig_dir}/epoch_{self.epoch}.jpg')
        plt.close('fig')

    def evaluation(self):
        self.model.eval()
        gts = []
        preds = []  
        eval_loss = .0
        with torch.no_grad():
            for img, points, mask, target in self.dataloader_val:
        
                img = img.to(self.device)

                mask = torch.unsqueeze(mask, 1)
                mask = mask.float().to(self.device)

                bx0 = self.model(img, inference=True)
                # bx0 = self.model(img)['out']
                loss = self.criterian(mask, bx0)
                rst = bx0.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                # coordinates = peak_local_max(rst[..., 0], min_distance=10, threshold_abs=0.1)
                coordinates = extract_centroids(rst[..., 0], threshold=0.1)

                gts.append(len(points[0]))
                preds.append(len(coordinates))
                eval_loss += loss.item()
        r2 = r2_score(gts, preds)
        mae = mean_absolute_error(gts, preds)
        eval_loss /= len(self.dataloader_val)

        self.writer.add_scalars('data/metrics',
                                {
                                    'R2_score': r2,
                                    'MAE': mae,
                                    'eval_loss': eval_loss
                                },
                                self.epoch)
        # self.maes.append(round(mae, 4))
        # self.r2s.append(round(r2, 4))

        if mae < self.best_mae:
            print('replacing best model')
            self.best_mae = mae
            self.best_mae_epoch = self.epoch
            self.best_model = copy.deepcopy(self.get_checkpoint())
        # if eval_loss < self.best_eval_loss:
        #     print('replacing best model')
        #     self.best_eval_loss = eval_loss
        #     self.best_eval_loss_epoch = self.epoch
        #     self.best_model = copy.deepcopy(self.get_checkpoint())

        return r2, mae, eval_loss
    
    def get_checkpoint(self):
        return{
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'lr_schedular_state_dict': self.lr_scheduler.state_dict(),
                'epoch': self.epoch,
                'iteration': self.iter,
                # 'maes': self.maes,
                # 'r2s': self.r2s,
                'best_mae': self.best_mae,
                'best_mae_epoch': self.best_mae_epoch,
                # 'best_eval_loss': self.best_eval_loss,
                # 'best_eval_loss_epoch': self.best_eval_loss_epoch
            }