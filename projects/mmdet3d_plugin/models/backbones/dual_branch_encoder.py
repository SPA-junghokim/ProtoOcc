from torch import nn

from mmdet3d.models import BACKBONES
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmdet3d.models import builder
import torch

class voxelize_module(nn.Module):
    def __init__(
            self,
            bev_z = 16,
            in_dim=48,
    ):
        super(voxelize_module, self).__init__()
        self.bev_z = bev_z
        self.linear = nn.Sequential(
                nn.Linear(in_dim, 2 * in_dim),
                nn.ReLU(),
                nn.Linear(2*in_dim, self.bev_z * in_dim),
            )
    def forward(self, x):
        x_linear = self.linear(x.permute(0,2,3,1))
        B, H, W, C_Z = x_linear.shape
        x = x_linear.reshape(B, H, W, self.bev_z, -1).permute(0,4,1,2,3)
        return x


@BACKBONES.register_module()
class Dual_Branch_Encoder(nn.Module):
    def __init__(
        self,
        z_size = [4, 8, 16],
        vox_feat1=80,
        vox_feat2=24,
        vox_feat3=48*4,
        vox_feat4=48*8,
        voxel_out_channels=48,
        bev_feat_ch1=160,
        bev_feat_ch2=320,
        bev_feat_ch3=640,
        vox_kernel_size=3,
        with_cp=True,
        bev_encoder_backbone=None,
        bev_encoder_neck=None,
        down_sample_for_3d_pooling=None,
    ):
        super().__init__()
        self.with_cp = with_cp
        
        # BEV encoder
        self.down_sample_for_3d_pooling = \
                    nn.Conv2d(down_sample_for_3d_pooling[0],
                    down_sample_for_3d_pooling[1],
                    kernel_size=1,
                    padding=0,
                    stride=1)
        self.bev_encoder_backbone = builder.build_backbone(bev_encoder_backbone)

        # Voxel encoder
        self.vox_branch_first = self._make_vox_branch_first(vox_feat1, vox_feat2, vox_kernel_size)
        self.vox_branch1 = self._make_downsample_branch(vox_feat2, vox_feat2, vox_kernel_size)
        self.vox_branch2 = self._make_vox_branch(vox_feat2, vox_feat3, vox_kernel_size, downsample=True)
        self.vox_branch3 = self._make_vox_branch(vox_feat3, vox_feat4, vox_kernel_size, downsample=True)
        self.vox_branch4 = self._make_vox_branch(vox_feat4, vox_feat3, vox_kernel_size)

        # Hierarchical Fusion Module
        self.vox_up_branch2 = self._make_upsample_branch(vox_feat3, vox_feat2, vox_kernel_size)
        self.vox_up_branch3 = self._make_upsample_branch(vox_feat2, vox_feat2, vox_kernel_size)

        self.vox_up_branch_final = nn.Sequential(
            nn.Conv3d(vox_feat2, voxel_out_channels * 2, kernel_size=1, padding=0, stride=1),
            nn.Softplus(),
            nn.Conv3d(voxel_out_channels * 2, voxel_out_channels, kernel_size=1, padding=0, stride=1),
        )

        self.bev_ch1 = self._make_bev_branch(bev_feat_ch3, vox_feat3, z_size[0])
        self.bev_ch2 = self._make_bev_branch(bev_feat_ch2, vox_feat2, z_size[1])
        self.bev_ch3 = self._make_bev_branch(bev_feat_ch1, vox_feat2, z_size[2])

        # for high performance
        self.bev_encoder_neck = builder.build_neck(bev_encoder_neck)
        self.voxelize_module = voxelize_module(
            in_dim = voxel_out_channels,
            )
        
    def _make_vox_branch_first(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def _make_vox_branch(self, in_channels, out_channels, kernel_size, downsample=False):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        ]
        if downsample:
            layers.extend([
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=2),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
            ])
        return nn.Sequential(*layers)

    def _make_downsample_branch(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def _make_upsample_branch(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def _make_bev_branch(self, bev_channels, vox_channels, z_size):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(bev_channels, vox_channels * z_size, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(vox_channels * z_size),
            nn.ReLU(),
        )


    def apply_checkpoint(self, layer, x):
        if self.with_cp and self.training:
            return cp.checkpoint(layer, x)
        else:
            return layer(x)

    def forward(self, x):
        # Voxel Branch
        # checkpoint is crucial for reducing GPU memory usage during training, but require longer training time.
        vox_res = self.apply_checkpoint(self.vox_branch_first, x)
        vox1 = self.apply_checkpoint(self.vox_branch1, vox_res)
        vox2 = self.apply_checkpoint(self.vox_branch2, vox1)
        vox3 = self.apply_checkpoint(self.vox_branch3, vox2)
        vox3 = F.interpolate(vox3, size=vox2.shape[2:], mode='trilinear', align_corners=True)
        vox3 = self.apply_checkpoint(self.vox_branch4, vox3)
        
        # BEV Branch
        pooled_x = torch.cat(x.unbind(dim=2), 1)
        pooled_x = self.down_sample_for_3d_pooling(pooled_x)
        multi_scale_bev = self.bev_encoder_backbone(pooled_x)

        # Hierarchical Fusion Module
        B, C, Z, H, W = vox3.shape
        vox3 = vox3 + vox2 + self.bev_ch1(multi_scale_bev[2]).view(B, -1, Z, H, W).contiguous()
        vox_2 = self.apply_checkpoint(self.vox_up_branch2, vox3)
        B, C, Z, H, W = vox_2.shape
        vox_2 = vox_2 + vox1 + self.bev_ch2(multi_scale_bev[1]).view(B, -1, Z, H, W).contiguous()
        vox_raw = self.apply_checkpoint(self.vox_up_branch3, vox_2)
        B, C, Z, H, W = vox_res.shape
        vox_raw = self.apply_checkpoint(self.vox_up_branch_final, vox_res + vox_raw + self.bev_ch3(multi_scale_bev[0]).view(B, -1, Z, H, W).contiguous())
        vox_raw = vox_raw.permute(0,1,3,4,2)

        # residual connection (for high performance)
        bev = self.bev_encoder_neck(multi_scale_bev)
        vox = self.voxelize_module(bev[0])
        comprehensive_voxel_feature = vox + vox_raw

        return comprehensive_voxel_feature
