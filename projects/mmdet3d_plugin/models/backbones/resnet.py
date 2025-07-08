import torch.utils.checkpoint as checkpoint
from torch import nn

from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet3d.models import BACKBONES
from typing import Optional, Union, Sequence
import torch

@BACKBONES.register_module()
class CustomResNet(nn.Module):
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
    ):
        super(CustomResNet, self).__init__()
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids

        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [Bottleneck(inplanes=curr_numC, planes=num_channels[i]//4, stride=stride[i],
                                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([Bottleneck(inplanes=curr_numC, planes=num_channels[i]//4, stride=1,
                                         downsample=None, norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [BasicBlock(inplanes=curr_numC, planes=num_channels[i], stride=stride[i],
                                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([BasicBlock(inplanes=curr_numC, planes=num_channels[i], stride=1,
                                          downsample=None, norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)])
                layers.append(nn.Sequential(*layer))
        else:
            assert False

        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


class grn(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(grn, self).__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def init_fn(self, shape, fill_value):
        return torch.full(shape, fill_value)

    def forward(self, inputs, mask=None):
        x = inputs
        if mask is not None:
            x = x * (1. - mask)
        Gx = torch.sqrt(torch.sum(torch.pow(x, 2), dim=(1, 2), keepdim=True) + self.eps)
        Nx = Gx / (torch.mean(Gx, dim=-1, keepdim=True) + self.eps)
        return self.gamma * (Nx * inputs) + self.beta + inputs

class Block(nn.Module):
    def __init__(self, dim, kernel_size=7, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = grn(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x

@BACKBONES.register_module()
class CustomBEVBackbone(nn.Module):
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=True,
            ConvNext_kernel_size=7,
    ):
        super(CustomBEVBackbone, self).__init__()
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        curr_numC = numC_input
        cur = 0
        for i in range(len(num_layer)):
            layer = [BasicBlock(inplanes=curr_numC, planes=num_channels[i], stride=stride[i],
                                downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                                norm_cfg=norm_cfg)]
            curr_numC = num_channels[i]
            layer.extend([Block(dim=curr_numC,kernel_size=ConvNext_kernel_size) for _ in range(num_layer[i])])
            layers.append(nn.Sequential(*layer))
            cur += num_layer[i]
        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

class BasicBlock3D(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = ConvModule(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.conv2 = ConvModule(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=None)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)

@BACKBONES.register_module()
class CustomResNet3D(nn.Module):
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            with_cp=False,
    ):
        super(CustomResNet3D, self).__init__()
        assert len(num_layer) == len(stride)
        num_channels = [numC_input * 2 ** (i + 1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        curr_numC = numC_input
        for i in range(len(num_layer)):
            layer = [
                BasicBlock3D(
                    curr_numC,
                    num_channels[i],
                    stride=stride[i],
                    downsample=ConvModule(
                        curr_numC,
                        num_channels[i],
                        kernel_size=3,
                        stride=stride[i],
                        padding=1,
                        bias=False,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=dict(type='BN3d', ),
                        act_cfg=None)
                    )
            ]
            curr_numC = num_channels[i]
            layer.extend([
                BasicBlock3D(curr_numC, curr_numC)
                for _ in range(num_layer[i] - 1)
            ])
            layers.append(nn.Sequential(*layer))
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats
