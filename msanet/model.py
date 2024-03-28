import torch.nn as nn
import torch
import timm
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

class SoyModel(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_small_in22k', pretrained=pretrained, features_only=True)

        dims = [96, 192, 384, 768]
        self.c1 = nn.Conv2d(dims[3], 1, kernel_size=3, padding=1, bias=False)
        self.c2 = nn.Conv2d(dims[3]+1, 1, kernel_size=3, padding=1, bias=False)
        self.c3 = nn.Conv2d(dims[2]+1, 1, kernel_size=3, padding=1, bias=False)
        self.c4 = nn.Conv2d(dims[1]+1, 1, kernel_size=3, padding=1, bias=False)
        self.c5 = nn.Conv2d(dims[0]+1, dims[0], kernel_size=3, padding=1, bias=False)
        self.c6 = nn.Conv2d(dims[0], 1, kernel_size=3, padding=1, bias=False)
        # self.

    def upsample_conv(self, x, conv):
        return conv(nn.UpsamplingNearest2d(scale_factor=2)(x))

    def forward(self, x, inference=False):
        act = nn.Sigmoid()
        x1 ,x2, x3, x4 = self.backbone(x)

        bx4 = self.c1(x4)
        bx3 = self.upsample_conv(torch.cat((bx4, x4), 1), self.c2)
        bx2 = self.upsample_conv(torch.cat((bx3, x3), 1), self.c3)
        bx1 = self.upsample_conv(torch.cat((bx2, x2), 1), self.c4)
        bx0 = self.upsample_conv(torch.cat((bx1, x1), 1), self.c5)
        bx0 = self.upsample_conv(bx0, self.c6)

        if inference:
            return act(bx0)
        else:
            return act(bx0), act(bx1), act(bx2), act(bx3), act(bx4)
