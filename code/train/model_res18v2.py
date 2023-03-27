# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PART OF THIS FILE AT ALL TIMES.

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable

import resnet as models
from data.config import solver
from layers import *


class conv_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MTNet(nn.Module):

    def __init__(self, num_classes, seg_classes, drivable_classes, max_depth, lane_class, freeze_backbone):
        super(MTNet, self).__init__()
        self.lane_class = lane_class
        self.max_depth = max_depth
        self.num_classes = num_classes
        self.seg_classes = seg_classes
        self.drivable_classes = drivable_classes
        resnet18_32s = models.resnet18(pretrained=True)
        resnet_block_expansion_rate = resnet18_32s.layer1[0].expansion
        aspect_ratios = solver['aspect_ratios']
        num_anchors = [len(i) * 2 + 2 for i in aspect_ratios]
        self.num_anchors = num_anchors

        self.resnet18_32s = resnet18_32s
        self.resnet18_32s = resnet18_32s
        if freeze_backbone:
            for child in self.resnet18_32s.children():
                for param in child.parameters():
                    param.requires_grad = False

        self.encode_layer0_seg = nn.Sequential(
            conv_bn_relu(64, 64, 3, 1, 1),
            conv_bn_relu(64, 64, 1, 1, 0),
        )
        self.conv_cat_seg2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, padding=1), nn.BatchNorm2d(64),
                                           nn.ReLU(inplace=True))
        self.conv_seg = nn.Sequential(nn.Conv2d(64, 64, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.reg_2s_seg = nn.Sequential(OrderedDict([
            ('reg_2s_seg_conv', nn.Conv2d(64, self.seg_classes, kernel_size=1)),
        ]))

        self.encode_layer0_det = nn.Sequential(
            conv_bn_relu(64, 64, 3, 1, 1),
            conv_bn_relu(64, 64, 1, 1, 0),
        )
        self.conv_cat_layer0_det2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, padding=1), nn.BatchNorm2d(64),
                                                  nn.ReLU(inplace=True))
        self.conv_det = nn.Sequential(nn.Conv2d(64, 64, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        if self.drivable_classes:
            self.encode_layer0_drivable = nn.Sequential(
                conv_bn_relu(64, 64, 3, 1, 1),
                conv_bn_relu(64, 64, 1, 1, 0),
            )
            self.conv_cat_drivable2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, padding=1), nn.BatchNorm2d(64),
                                                    nn.ReLU(inplace=True))
            self.conv_drivable = nn.Sequential(nn.Conv2d(64, 64, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
            self.score_2drivable = nn.Sequential(OrderedDict([
                ('score_2drivable_conv', nn.Conv2d(64, self.drivable_classes, kernel_size=1)),
                # ('score_2drivable_BN', nn.BatchNorm2d(self.drivable_classes)),
            ]))
        if self.max_depth:
            self.encode_layer0_depth = nn.Sequential(
                conv_bn_relu(64, 64, 3, 1, 1),
                conv_bn_relu(64, 64, 1, 1, 0),
            )
            self.conv_cat_depth_encode = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64),
                                                       nn.ReLU(inplace=True))
            self.conv_cat_depth2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, padding=1), nn.BatchNorm2d(64),
                                                 nn.ReLU(inplace=True))
            self.conv_depth = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                conv_bn_relu(64, 32, 1, 1, 0)
            )
            self.reg_2s_depth = nn.Sequential(OrderedDict([
                ('reg_2s_depth_conv_encode', conv_bn_relu(32, 32, 3, 1, 1)),
                ('reg_2s_depth_conv', nn.Conv2d(32, 1, kernel_size=1)),
                ('reg_2s_depth_sigmoid', nn.Sigmoid())
            ]))

        if self.lane_class:
            self.encode_fe3 = nn.Sequential(nn.Conv2d(512, 256, 1, 1), nn.ReLU(inplace=True))
            self.encode_feature = nn.Sequential(
                conv_bn_relu(256, 256, 1, 1),
                conv_bn_relu(256, 128, 1, 1),
                conv_bn_relu(128, 64, 1, 1))
            self.encode_layer0_lane = nn.Sequential(
                conv_bn_relu(64, 64, 3, 1, 1),
                conv_bn_relu(64, 64, 1, 1, 0),
            )
            self.conv_cat_lane2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, padding=1), nn.BatchNorm2d(64),
                                                nn.ReLU(inplace=True))
            self.conv_lane = nn.Sequential(nn.Conv2d(64, 64, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
            self.reg_2s_lane = nn.Sequential(OrderedDict([
                ('reg_2s_lane_conv', nn.Conv2d(64, self.lane_class, kernel_size=1)),
            ]))

        self.conv_block6 = nn.Sequential(OrderedDict([
            ('conv_block6_conv1', nn.Conv2d(512, 256, 1, padding=0)),
            ('conv_block6_BN1', nn.BatchNorm2d(256)),
            ('conv_block6_relu1', nn.ReLU(inplace=True)),
            ('conv_block6_conv2', nn.Conv2d(256, 512, 3, padding=1, stride=2)),
            ('conv_block6_BN2', nn.BatchNorm2d(512)),
            ('conv_block6_relu2', nn.ReLU(inplace=True)), ]))

        self.conv_block7 = nn.Sequential(OrderedDict([
            ('conv_block7_conv1', nn.Conv2d(512, 128, 1, padding=0)),
            ('conv_block7_BN1', nn.BatchNorm2d(128)),
            ('conv_block7_relu1', nn.ReLU(inplace=True)),
            ('conv_block7_conv2', nn.Conv2d(128, 256, 3, padding=0)),
            ('conv_block7_BN2', nn.BatchNorm2d(256)),
            ('conv_block7_relu2', nn.ReLU(inplace=True)), ]))

        self.conv_block8 = nn.Sequential(OrderedDict([
            ('conv_block8_conv1', nn.Conv2d(256, 128, 1, padding=0)),
            ('conv_block8_BN1', nn.BatchNorm2d(128)),
            ('conv_block8_relu1', nn.ReLU(inplace=True)),
            ('conv_block8_conv2', nn.Conv2d(128, 256, 3, padding=0)),
            ('conv_block8_BN2', nn.BatchNorm2d(256)),
            ('conv_block8_relu2', nn.ReLU(inplace=True)), ]))

        self.toplayer3 = nn.Sequential(OrderedDict([
            ('toplayer3_conv', nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)),
            ('toplayer3_BN', nn.BatchNorm2d(256)),
            ('toplayer3_relu', nn.ReLU(inplace=True)),
        ]))  # Reduce channels
        self.toplayer2 = nn.Sequential(OrderedDict([
            ('toplayer2_conv', nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)),
            ('toplayer2_BN', nn.BatchNorm2d(128)),
            ('toplayer2_relu', nn.ReLU(inplace=True)),
        ]))  # Reduce channels
        self.toplayer1 = nn.Sequential(OrderedDict([
            ('toplayer1_conv', nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)),
            ('toplayer1_BN', nn.BatchNorm2d(64)),
            ('toplayer1_relu', nn.ReLU(inplace=True)),
        ]))  # Reduce channels
        self.conv_cat_3 = conv_bn_relu(512, 256, 3, 1, 1)
        self.conv_cat_2 = conv_bn_relu(256, 128, 3, 1, 1)

        self.loc_0 = nn.Sequential(OrderedDict([
            ('loc_0_conv2', nn.Conv2d(64, num_anchors[0] * 4, 3, padding=1)),
        ]))
        self.loc_1 = nn.Sequential(OrderedDict([
            ('loc_1_conv2', nn.Conv2d(128, num_anchors[1] * 4, 3, padding=1)),
        ]))
        self.loc_2 = nn.Sequential(OrderedDict([
            ('loc_2_conv2', nn.Conv2d(256, num_anchors[2] * 4, 3, padding=1)),
        ]))
        self.loc_3 = nn.Sequential(OrderedDict([
            ('loc_3_conv2', nn.Conv2d(512, num_anchors[3] * 4, 3, padding=1)),
        ]))
        self.loc_4 = nn.Sequential(OrderedDict([
            ('loc_4_conv2', nn.Conv2d(512, num_anchors[4] * 4, 3, padding=1)),
        ]))
        self.loc_5 = nn.Sequential(OrderedDict([
            ('loc_5_conv2', nn.Conv2d(256, num_anchors[5] * 4, 3, padding=1)),
        ]))
        self.loc_6 = nn.Sequential(OrderedDict([
            ('loc_6_conv2', nn.Conv2d(256, num_anchors[6] * 4, 3, padding=1)),
        ]))

        self.conf_0 = nn.Sequential(OrderedDict([
            ('conf_0_conv2', nn.Conv2d(64, num_anchors[0] * self.num_classes, 3, padding=1)),
        ]))
        self.conf_1 = nn.Sequential(OrderedDict([
            ('conf_1_conv2', nn.Conv2d(128, num_anchors[1] * self.num_classes, 3, padding=1)),
        ]))

        self.conf_2 = nn.Sequential(OrderedDict([
            ('conf_2_conv2', nn.Conv2d(256, num_anchors[2] * self.num_classes, 3, padding=1)),
        ]))
        self.conf_3 = nn.Sequential(OrderedDict([
            ('conf_3_conv2', nn.Conv2d(512, num_anchors[3] * self.num_classes, 3, padding=1)),
        ]))
        self.conf_4 = nn.Sequential(OrderedDict([
            ('conf_4_conv2', nn.Conv2d(512, num_anchors[4] * self.num_classes, 3, padding=1)),
        ]))
        self.conf_5 = nn.Sequential(OrderedDict([
            ('conf_5_conv2', nn.Conv2d(256, num_anchors[5] * self.num_classes, 3, padding=1)),
        ]))
        self.conf_6 = nn.Sequential(OrderedDict([
            ('conf_6_conv2', nn.Conv2d(256, num_anchors[6] * self.num_classes, 3, padding=1)),
        ]))

        self.centerness_0 = nn.Sequential(OrderedDict([
            ('centerness_0_conv', nn.Conv2d(64, num_anchors[0] * 1, 3, padding=1)),
        ]))
        self.centerness_1 = nn.Sequential(OrderedDict([
            ('centerness_1_conv', nn.Conv2d(128, num_anchors[1] * 1, 3, padding=1)),
        ]))

        self.centerness_2 = nn.Sequential(OrderedDict([
            ('centerness_2_conv', nn.Conv2d(256, num_anchors[2] * 1, 3, padding=1)),
        ]))
        self.centerness_3 = nn.Sequential(OrderedDict([
            ('centerness_3_conv', nn.Conv2d(512, num_anchors[3] * 1, 3, padding=1)),
        ]))
        self.centerness_4 = nn.Sequential(OrderedDict([
            ('centerness_4_conv', nn.Conv2d(512, num_anchors[4] * 1, 3, padding=1)),
        ]))
        self.centerness_5 = nn.Sequential(OrderedDict([
            ('centerness_5_conv', nn.Conv2d(256, num_anchors[5] * 1, 3, padding=1)),
        ]))
        self.centerness_6 = nn.Sequential(OrderedDict([
            ('centerness_6_conv', nn.Conv2d(256, num_anchors[6] * 1, 3, padding=1)),
        ]))

        self.priorbox = PriorBox(solver, return_lv_info=True)
        self.priors, self.num_anchors_per_lv = self.priorbox.forward()
        self.priors = Variable(self.priors, requires_grad=False)

    def forward(self, x, **kwargs):
        loc = list()
        conf = list()
        centerness = list()

        x = self.resnet18_32s.conv1(x)
        x = self.resnet18_32s.bn1(x)
        x = self.resnet18_32s.relu(x)
        f_2s = x
        x = self.resnet18_32s.maxpool(x)

        x = self.resnet18_32s.layer1(x)
        f0 = x

        x = self.resnet18_32s.layer2(x)
        f1 = x

        x = self.resnet18_32s.layer3(x)
        f2 = x

        x = self.resnet18_32s.layer4(x)
        feature3 = x

        top3 = nn.functional.upsample(self.toplayer3(feature3), scale_factor=2, mode='bilinear')
        feature2 = self.conv_cat_3(torch.cat([top3, f2], 1))
        top2 = nn.functional.upsample(self.toplayer2(feature2), scale_factor=2, mode='bilinear')
        feature1 = self.conv_cat_2(torch.cat([top2, f1], 1))
        nonupsample_top1 = self.toplayer1(feature1)
        top1 = nn.functional.upsample(nonupsample_top1, scale_factor=2, mode='bilinear')

        feature4 = self.conv_block6(feature3)
        feature5 = self.conv_block7(feature4)
        feature6 = self.conv_block8(feature5)
        bs = feature6.size(0)
        enfe3 = self.encode_fe3(feature3)
        encode_feture = self.encode_feature(enfe3)
        encode_feture = nn.functional.upsample(encode_feture, scale_factor=4, mode='bilinear')
        up_encode_feture = nn.functional.upsample(encode_feture, scale_factor=2,
                                                  mode='bilinear')
        f0_encode_feature_seg = self.encode_layer0_seg(top1)
        # f0_encode_feature_seg = top1 + f0_encode_feature_seg + up_encode_feture
        f0_encode_feature_seg = self.conv_cat_seg2(f0_encode_feature_seg + up_encode_feture)
        seg_feature = self.conv_seg(f0_encode_feature_seg)
        seg_feature = nn.functional.upsample(seg_feature, scale_factor=2, mode='bilinear')
        logits_2s = self.reg_2s_seg(seg_feature)
        seg = nn.functional.upsample(logits_2s, scale_factor=2, mode='bilinear')

        if self.drivable_classes:
            f0_encode_feature = self.encode_layer0_drivable(top1)
            # drivable_feature = f0_encode_feature + up_encode_feture
            drivable_feature = self.conv_cat_drivable2(f0_encode_feature + up_encode_feture)
            drivable_feature = self.conv_drivable(drivable_feature)
            drivable_feature = nn.functional.upsample(drivable_feature, scale_factor=2, mode='bilinear')
            drivable = self.score_2drivable(drivable_feature)
        else:
            drivable = None

        if self.max_depth:
            f0_encode_feature_depth = self.encode_layer0_depth(nonupsample_top1)
            f0_encode_feature_depth = self.conv_cat_depth2(
                f0_encode_feature_depth + self.conv_cat_depth_encode(encode_feture))
            f0_encode_feature_depth = nn.functional.upsample(f0_encode_feature_depth, scale_factor=2, mode='bilinear')
            depth_feature = self.conv_depth(f0_encode_feature_depth)
            depth_feature = nn.functional.upsample(depth_feature, scale_factor=2, mode='bilinear')
            depth = self.reg_2s_depth(depth_feature) * self.max_depth
        else:
            depth = None

        if self.lane_class:
            f0_encode_feature_lane = self.encode_layer0_lane(top1)
            f0_encode_feature_lane = self.conv_cat_lane2(
                f0_encode_feature_lane + up_encode_feture)
            lane_feature = self.conv_lane(f0_encode_feature_lane)
            lane_feature = nn.functional.upsample(lane_feature, scale_factor=2, mode='bilinear')
            lane = self.reg_2s_lane(lane_feature)
            lane = nn.functional.upsample(lane, scale_factor=2, mode='bilinear')
        else:
            lane = None

        f0_encode_feature_det = self.encode_layer0_det(top1)
        f0_encode_feature_det = self.conv_cat_layer0_det2(f0_encode_feature_det + up_encode_feture)
        f0_encode_feature_det = self.conv_det(f0_encode_feature_det)
        # loc.append(self.loc_0(f0_encode_feature_det).permute(0, 2, 3, 1).contiguous())
        # loc.append(self.loc_1(feature1).permute(0, 2, 3, 1).contiguous())
        # loc.append(self.loc_2(feature2).permute(0, 2, 3, 1).contiguous())
        # loc.append(self.loc_3(feature3).permute(0, 2, 3, 1).contiguous())
        # loc.append(self.loc_4(feature4).permute(0, 2, 3, 1).contiguous())
        # loc.append(self.loc_5(feature5).permute(0, 2, 3, 1).contiguous())
        # loc.append(self.loc_6(feature6).permute(0, 2, 3, 1).contiguous())
        #
        # conf.append(self.conf_0(f0_encode_feature_det).permute(0, 2, 3, 1).contiguous())
        # conf.append(self.conf_1(feature1).permute(0, 2, 3, 1).contiguous())
        # conf.append(self.conf_2(feature2).permute(0, 2, 3, 1).contiguous())
        # conf.append(self.conf_3(feature3).permute(0, 2, 3, 1).contiguous())
        # conf.append(self.conf_4(feature4).permute(0, 2, 3, 1).contiguous())
        # conf.append(self.conf_5(feature5).permute(0, 2, 3, 1).contiguous())
        # conf.append(self.conf_6(feature6).permute(0, 2, 3, 1).contiguous())
        #
        # centerness.append(self.centerness_0(f0_encode_feature_det).permute(0, 2, 3, 1).contiguous())
        # centerness.append(self.centerness_1(feature1).permute(0, 2, 3, 1).contiguous())
        # centerness.append(self.centerness_2(feature2).permute(0, 2, 3, 1).contiguous())
        # centerness.append(self.centerness_3(feature3).permute(0, 2, 3, 1).contiguous())
        # centerness.append(self.centerness_4(feature4).permute(0, 2, 3, 1).contiguous())
        # centerness.append(self.centerness_5(feature5).permute(0, 2, 3, 1).contiguous())
        # centerness.append(self.centerness_6(feature6).permute(0, 2, 3, 1).contiguous())

        loc.append(self.loc_0(f0_encode_feature_det))
        loc.append(self.loc_1(feature1))
        loc.append(self.loc_2(feature2))
        loc.append(self.loc_3(feature3))
        loc.append(self.loc_4(feature4))
        loc.append(self.loc_5(feature5))
        loc.append(self.loc_6(feature6))

        conf.append(self.conf_0(f0_encode_feature_det))
        conf.append(self.conf_1(feature1))
        conf.append(self.conf_2(feature2))
        conf.append(self.conf_3(feature3))
        conf.append(self.conf_4(feature4))
        conf.append(self.conf_5(feature5))
        conf.append(self.conf_6(feature6))

        centerness.append(self.centerness_0(f0_encode_feature_det))
        centerness.append(self.centerness_1(feature1))
        centerness.append(self.centerness_2(feature2))
        centerness.append(self.centerness_3(feature3))
        centerness.append(self.centerness_4(feature4))
        centerness.append(self.centerness_5(feature5))
        centerness.append(self.centerness_6(feature6))

        # loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        # conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # centerness = torch.cat([o.view(o.size(0), -1) for o in centerness], 1)
        #
        # output = (
        #     loc.view(loc.size(0), -1, 4),
        #     [conf.view(conf.size(0), -1, self.num_classes), centerness.view(centerness.size(0), -1)],
        #     seg,
        #     self.priors,
        #     drivable,
        #     depth
        # )

        output = (
            loc,
            [conf, centerness],
            seg,
            self.priors,
            drivable,
            depth,
            lane
        )
        return output


def build_model(det_classes, seg_classes, drivable_classes, max_depth, lane_class, freeze_backbone=False):
    model = MTNet(det_classes, seg_classes, drivable_classes, max_depth, lane_class, freeze_backbone)
    return model
