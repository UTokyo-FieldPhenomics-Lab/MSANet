#### 
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from p2psoy.layers import *
import time

vgg_conv1_2 = vgg_conv2_2 = vgg_conv3_3 = vgg_conv4_3 = vgg_conv5_3 = None

def conv_1_2_hook(module, input, output):
    global vgg_conv1_2
    vgg_conv1_2 = output
    return None

def conv_2_2_hook(module, input, output):
    global vgg_conv2_2
    vgg_conv2_2 = output
    return None

def conv_3_3_hook(module, input, output):
    global vgg_conv3_3
    vgg_conv3_3 = output
    return None

def conv_4_3_hook(module, input, output):
    global vgg_conv4_3
    vgg_conv4_3 = output
    return None

def conv_5_3_hook(module, input, output):
    global vgg_conv5_3
    vgg_conv5_3 = output
    return None

##
class CPFE_hl(nn.Module):
    def __init__(self, feature_layer=None, out_channels=8):
        super(CPFE_hl, self).__init__()

        self.dil_rates = [3, 5, 7]

        # Determine number of in_channels from VGG-16 feature layer
        if feature_layer == 'conv5_3':
            self.in_channels = 512
        elif feature_layer == 'conv4_3':
            self.in_channels = 512
        elif feature_layer == 'conv3_3':
            self.in_channels = 256
        elif feature_layer == 'conv2_3':
            self.in_channels = 128
        elif feature_layer == 'conv1_3':
            self.in_channels = 64

        # Define layers
        self.conv_1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.conv_dil_3 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[0], padding=self.dil_rates[0], bias=False)
        self.conv_dil_5 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[1], padding=self.dil_rates[1], bias=False)
        self.conv_dil_7 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[2], padding=self.dil_rates[2], bias=False)

        self.bn = nn.BatchNorm2d(out_channels*4)

    def forward(self, input_):
        # Extract features
        conv_1_1_feats = self.conv_1_1(input_)
        conv_dil_3_feats = self.conv_dil_3(input_)
        conv_dil_5_feats = self.conv_dil_5(input_)
        conv_dil_7_feats = self.conv_dil_7(input_)

        # Aggregate features
        concat_feats = torch.cat((conv_1_1_feats, conv_dil_3_feats, conv_dil_5_feats, conv_dil_7_feats), dim=1)
        bn_feats = F.relu(self.bn(concat_feats))

        return bn_feats
##
class SODModel(nn.Module):
    def __init__(self):
        super(SODModel, self).__init__()

        # 
        self.vgg16 = models.vgg16(pretrained=True).features

        # Extract and register intermediate features of VGG-16_bn
        self.vgg16[3].register_forward_hook(conv_1_2_hook)
        self.vgg16[8].register_forward_hook(conv_2_2_hook)
        self.vgg16[15].register_forward_hook(conv_3_3_hook)
        self.vgg16[22].register_forward_hook(conv_4_3_hook)
        self.vgg16[29].register_forward_hook(conv_5_3_hook)

        # Initialize layers for high level (hl) feature (conv3_3, conv4_3, conv5_3) processing
        self.cpfe_conv3_3 = CPFE_hl(feature_layer='conv3_3')
        self.cpfe_conv4_3 = CPFE_hl(feature_layer='conv4_3')
        self.cpfe_conv5_3 = CPFE_hl(feature_layer='conv5_3')
        #
        self.cpfe_conv1_3 = CPFE_hl(feature_layer='conv1_3')
        self.cpfe_conv2_3 = CPFE_hl(feature_layer='conv2_3')
        # 11,03,2022, remove channel attention
        self.cha_att = ChannelwiseAttention(in_channels=96)  # in_channels = 3 x (8 x 4)

        self.hl_conv1 = nn.Conv2d(96, 8, (3, 3), padding=1)
        self.hl_bn1 = nn.BatchNorm2d(8)

        # 
        self.ll_conv_1 = nn.Conv2d(64, 8, (3, 3), padding=1)
        self.ll_bn_1 = nn.BatchNorm2d(8)
        self.ll_conv_2 = nn.Conv2d(128, 8, (3, 3), padding=1)
        self.ll_bn_2 = nn.BatchNorm2d(8)
        self.ll_conv_3 = nn.Conv2d(64, 8, (3, 3), padding=1) 
        self.ll_bn_3 = nn.BatchNorm2d(8)

        self.spa_att = SpatialAttention(in_channels=8)

        # 
        self.ff_conv_1 = nn.Conv2d(16, 3, (3, 3), padding=1)
        self.ff_bn_1 = nn.BatchNorm2d(3)
    def forward(self, input_):
        global vgg_conv1_2, vgg_conv2_2, vgg_conv3_3, vgg_conv4_3, vgg_conv5_3

        # Pass input_ through vgg16 to generate intermediate features
        self.vgg16(input_)
        # Process high level features
        conv3_cpfe_feats = self.cpfe_conv3_3(vgg_conv3_3)
        conv4_cpfe_feats = self.cpfe_conv4_3(vgg_conv4_3)
        conv5_cpfe_feats = self.cpfe_conv5_3(vgg_conv5_3)

        conv4_cpfe_feats = F.interpolate(conv4_cpfe_feats, scale_factor=2, mode='bilinear', align_corners=True) # reduce spatial dimension by 2
        conv5_cpfe_feats = F.interpolate(conv5_cpfe_feats, scale_factor=4, mode='bilinear', align_corners=True)

        conv_345_feats = torch.cat((conv3_cpfe_feats, conv4_cpfe_feats, conv5_cpfe_feats), dim=1)
        
        # channel attention on high level features
        conv_345_ca, ca_act_reg = self.cha_att(conv_345_feats)
        conv_345_feats = torch.mul(conv_345_feats, conv_345_ca)

        conv_345_feats = self.hl_conv1(conv_345_feats)
        conv_345_feats = F.relu(self.hl_bn1(conv_345_feats))
        ##
        # Process low level features
        conv0_feats = input_ # the original input image
        conv1_cpfe_feats = self.cpfe_conv1_3(vgg_conv1_2)
        conv2_cpfe_feats = self.cpfe_conv2_3(vgg_conv2_2)

        conv0_feats = F.interpolate(conv0_feats, scale_factor=0.25, mode='bilinear', align_corners=True)
        conv1_cpfe_feats = F.interpolate(conv1_cpfe_feats, scale_factor=0.25, mode='bilinear', align_corners=True)
        conv2_cpfe_feats = F.interpolate(conv2_cpfe_feats, scale_factor=0.5, mode='bilinear', align_corners=True)

        #
        conv_12_feats = torch.cat((conv1_cpfe_feats, conv2_cpfe_feats), dim=1)
        conv_12_feats = self.ll_conv_3(conv_12_feats)
        conv_12_feats = F.relu(self.ll_bn_3(conv_12_feats))
        # spatial attention on low level features
        conv_12_sa = self.spa_att(conv_12_feats)
        conv_12_feats = torch.mul(conv_12_feats, conv_12_sa)

        # fuse the low and high level features
        fused_feats = torch.cat((conv_12_feats, conv_345_feats), dim=1)
        #
        fused_final = self.ff_conv_1(fused_feats)
        fused_final = F.relu(self.ff_bn_1(fused_final))
        # add the fused low and high level features to the original image
        fused_final_out = torch.add(fused_final, conv0_feats)
        fused_final_out = torch.sigmoid(fused_final_out)

        return fused_final_out

# build model p2pNet.py

# the network frmawork of the regression branch
class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=32):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1) # one point has two coordinates 
    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 2)

# the network frmawork of the classification branch
class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=32):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1) # one classes, only positives 
        self.output_act = nn.Sigmoid()
    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, _ = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

# generate the reference points in grid layout
def generate_anchor_points(stride=8, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    return anchor_points
# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5)* stride
    shift_y = (np.arange(0, shape[0]) + 0.5)* stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points

# 
class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels] # calcualtes the output size of the model (image of 128*128 to feature map of 16*16)

        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        # get reference points for each level
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2**p, row=self.row, line=self.line)
            shifted_anchor_points = shift(image_shapes[idx], self.strides[idx], anchor_points)
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)

        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        # send reference points to device
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))
##
# the defenition of the P2PNet model
class P2PNet(nn.Module):
    def __init__(self, row=2, line=2):
        super().__init__()
        self.num_classes = 2
        # the number of all anchor points
        num_anchor_points = row * line

        self.regression = RegressionModel(num_features_in=3, num_anchor_points=num_anchor_points)
        self.classification = ClassificationModel(num_features_in=3, \
                                            num_classes=self.num_classes, \
                                            num_anchor_points=num_anchor_points)

        self.anchor_points = AnchorPoints(pyramid_levels=[2,], row=row, line=line) # remember to change pyramid level when you change feature input

        self.fpn = SODModel()

    def forward(self, samples): #
        # 
        features_fpn = self.fpn(samples) # output = bach_size, channel, Height, Weight
        #
        batch_size = features_fpn.size()[0]
        # run the regression and classification branch
        regression = self.regression(features_fpn) * 100 # 8x
        classification = self.classification(features_fpn)
        #
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        output_coord = regression + anchor_points
        output_class = classification
        out = {'pred_logits': output_class, 'pred_points': output_coord}
        #
        return out