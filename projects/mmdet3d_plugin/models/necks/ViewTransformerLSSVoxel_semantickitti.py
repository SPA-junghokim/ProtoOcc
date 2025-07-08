# Copyright (c) Phigent Robotics. All rights reserved.
import torch
from mmdet3d.models.builder import NECKS
from projects.mmdet3d_plugin.ops.bev_pool import bev_pool
from mmcv.runner import force_fp32
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
import pdb

from .view_transformer_semantickitti import *

@NECKS.register_module()
class ViewTransformerLiftSplatShootVoxel_SemanticKITTI(ViewTransformerLSSBEVDepth_SemanticKITTI):
    def __init__(
            self, 
            loss_depth_weight,
            point_cloud_range=None,
            loss_depth_type='bce', 
            num_class = 20,
            **kwargs,
        ):
        super(ViewTransformerLiftSplatShootVoxel_SemanticKITTI, self).__init__(loss_depth_weight=loss_depth_weight, **kwargs)
        
        self.loss_depth_type = loss_depth_type
        self.cam_depth_range = self.grid_config['dbound']
        self.point_cloud_range = point_cloud_range
        self.num_class =num_class

        
    def get_downsampled_gt_depth(self, gt_depths):
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)
        
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()
        
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        
        return gt_depths_vals, gt_depths.float()
    
    @force_fp32()
    def get_bce_depth_loss(self, depth_labels, depth_preds):
        _, depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none').sum() / max(1.0, fg_mask.sum())
        
        return depth_loss
    
    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        if self.loss_depth_type == 'bce':
            depth_loss = self.get_bce_depth_loss(depth_labels, depth_preds)
        
        else:
            pdb.set_trace()
        
        return self.loss_depth_weight * depth_loss
        
    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        x = x.reshape(Nprime, C)

        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        final = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
        final = final.permute(0, 1, 3, 4, 2)

        return final

    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...]
        img_feat = x[:, self.D:self.D + self.numC_Trans, ...]
            
        depth_prob = self.get_depth_dist(depth_digit)

        # Lift
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, -1, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans, bda)
        bev_feat = self.voxel_pooling(geom, volume)
        
        return bev_feat, depth_prob

    def get_downsampled_gt_depth_and_semantic(self, gt_depths, gt_semantics):
        gt_semantics[gt_depths < self.grid_config['depth'][0]] = 0
        gt_semantics[gt_depths > self.grid_config['depth'][1]] = 0
        gt_depths[gt_depths < self.grid_config['depth'][0]] = 0
        gt_depths[gt_depths > self.grid_config['depth'][1]] = 0

        B, N, H, W = gt_semantics.shape
        one_hot = torch.nn.functional.one_hot(gt_semantics.to(torch.int64), num_classes=self.num_class)
        
        semantic_downsample = self.downsample
            
        one_hot = one_hot.view(B, N, H // semantic_downsample, semantic_downsample, W // semantic_downsample, semantic_downsample, self.num_class)
        class_counts = one_hot.sum(dim=(3, 5)).to(gt_semantics)
        class_counts[..., 0] = 0
        class_counts = class_counts.to(gt_semantics)
            
        _, most_frequent_classes = class_counts.max(dim=-1)
        gt_semantics = most_frequent_classes.view(B * N, H // semantic_downsample, W // semantic_downsample)
        gt_semantics = F.one_hot(gt_semantics.long(), num_classes=self.num_class).permute(0,3,1,2).float().contiguous()

        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)
        gt_depths = (gt_depths - (self.grid_config['depth'][0] - self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths_onehot = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:].float()
        return gt_depths_onehot, gt_semantics