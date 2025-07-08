# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from torch.utils.checkpoint import checkpoint
from mmcv.cnn.bricks import ConvModule
from mmdet.models import NECKS
import torch.nn.functional as F

@NECKS.register_module()
class FPN_LSS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 ):
        super(FPN_LSS, self).__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.out_channels = out_channels
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)

        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor, kernel_size=3,
                      padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
        )

        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
            )

        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral)[1],
                nn.ReLU(inplace=True)
            )

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        if self.extra_upsample:
            x = self.up2(x)
        return x

@NECKS.register_module()
class Custom_FPN_LSS(nn.Module):
    def __init__(self,
                 catconv_in_channels1,
                 catconv_in_channels2,
                 out_channels,
                 scale_factor=2,
                 input_feature_index=(0, 1, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 with_cp=False):
        super(Custom_FPN_LSS, self).__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.out_channels = out_channels
        self.with_cp = with_cp
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)

        channels_factor = 2 if self.extra_upsample else 1

        self.cat_conv1 = self._make_cat_conv(
            catconv_in_channels1, out_channels * channels_factor, norm_cfg)
        self.cat_conv2 = self._make_cat_conv(
            catconv_in_channels2, out_channels * channels_factor, norm_cfg)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
            nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        )

    def _make_cat_conv(self, in_channels, out_channels, norm_cfg):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        x3, x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]], feats[self.input_feature_index[2]]
        x1_up = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)

        x2_cat = torch.cat([x2, x1_up], dim=1)
        x2_cat = self._apply_checkpoint(self.cat_conv1, x2_cat)

        x2_up = self._apply_checkpoint(self.up, x2_cat)

        x3_cat = torch.cat([x3, x2_up], dim=1)
        x3_cat = self._apply_checkpoint(self.cat_conv2, x3_cat)

        if self.extra_upsample:
            x4_out = self.up2(x3_cat)
        else:
            x4_out = None

        return [x4_out, x4_out, x4_out, x4_out]

    def _apply_checkpoint(self, module, *inputs):
        if self.with_cp:
            return checkpoint(module, *inputs)
        else:
            return module(*inputs)