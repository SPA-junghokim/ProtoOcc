# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss
from ..losses.semkitti_loss import sem_scal_loss, geo_scal_loss
from ..losses.lovasz_softmax import lovasz_softmax
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

nusc_class_frequencies = np.array([
 944004,
 1897170,
 152386,
 2391677,
 16957802,
 724139,
 189027,
 2074468,
 413451,
 2384460,
 5916653,
 175883646,
 4275424,
 51393615,
 61411620,
 105975596,
 116424404,
 1892500630
 ])
class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:19] + 0.001)).float()

@HEADS.register_module()
class cnn3d_decoder(BaseModule):
    def __init__(self,
                 in_dim=32,
                 out_dim=32,
                 use_mask=True,
                 num_classes=18,
                 class_wise=False,
                 loss_occ=None,
                 loss_geo=None,
                 loss_geo_weight=1,
                 empty_idx=0,
                 loss_sem=None,
                 loss_sem_weight=1,
                 loss_weight=1.,
                 with_cp = False,
                 pred_upsample = False,
                 ):
        super(cnn3d_decoder, self).__init__()
        self.empty_idx = empty_idx
        self.out_dim = 32
        self.final_conv = ConvModule(
            in_dim,
            out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d')
        )
        self.predicter = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim*2),
            nn.Softplus(),
            nn.Linear(self.out_dim*2, num_classes),
        )

        self.num_classes = num_classes
        self.use_mask = use_mask
        self.class_wise = class_wise
        self.loss_weight=loss_weight

        self.weight = torch.Tensor([1.0]*self.num_classes)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=255, reduction="mean")
        self.lovasz_softmax_loss = lovasz_softmax
        self.loss_occ = build_loss(loss_occ)
        self.with_cp = with_cp
        self.pred_upsample = pred_upsample # for semantic kitti
        
        self.loss_geo = loss_geo
        self.loss_sem = loss_sem
        if self.loss_geo:
            self.loss_geo = geo_scal_loss
            self.loss_geo_weight = loss_geo_weight

        if self.loss_sem:
            self.loss_sem = sem_scal_loss
            self.loss_sem_weight = loss_sem_weight
        else:
            self.loss_sem_weight = False

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx)

        Returns:

        """
        if self.with_cp:
            occ_feat = checkpoint(self.final_conv ,img_feats)
            occ_feat = occ_feat.permute(0, 4, 3, 2, 1)
        else:
            occ_feat = self.final_conv(img_feats).permute(0, 4, 3, 2, 1)
        
        mask_feat = occ_feat
        if self.with_cp:
            occ_pred = checkpoint(self.predicter, occ_feat)
        else:
            occ_pred = self.predicter(occ_feat)
            
        return occ_pred, mask_feat

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        if self.pred_upsample: # for semantic kitti
            voxel_semantics = self.downsample_voxel(voxel_semantics)
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        
        occ_pred = occ_pred.permute(0,4,1,2,3)
        if self.loss_geo:
            loss_geo_scal = self.loss_geo(occ_pred, voxel_semantics, ignore_index=255, non_empty_idx=self.empty_idx)
            loss['loss_geo_scal_prototype'] = loss_geo_scal * self.loss_geo_weight
        if self.loss_sem_weight:
            loss_sem_scal = self.loss_sem(occ_pred, voxel_semantics, ignore_index=255)
            loss['loss_sem_scal_prototype'] = loss_sem_scal * self.loss_sem_weight
        voxel_loss = self.cross_entropy_loss(occ_pred, voxel_semantics.long())
        lovasz_softmax_loss = self.lovasz_softmax_loss(F.softmax(occ_pred, dim=1), voxel_semantics, ignore=255)
        loss['loss_CE_prototype'] = voxel_loss * self.loss_weight
        loss['lovasz_softmax_loss_prototype'] = lovasz_softmax_loss * self.loss_weight
        return loss
    
    def downsample_voxel(self, input_tensor): # for semantic kitti
        B, H, W, C = input_tensor.shape 
        tensor = input_tensor.view(B, H // 2, 2, W // 2, 2, C // 2, 2).permute(0,1,3,5,2,4,6).flatten(4)
        downsampled_tensor, _ = torch.mode(tensor, dim=(-1), keepdim=False)
        temp = downsampled_tensor.clone()
        all_empty_or_255_mask = ((tensor == 0) | (tensor == 255)).all(dim=(-1))
        non_empty_and_non_255_mask = ~all_empty_or_255_mask
        filtered_tensor = tensor[non_empty_and_non_255_mask]
        
        filtered_tensor[filtered_tensor == 255] = -1
        filtered_tensor = filtered_tensor.sort(dim=-1)[0][:,-1]
        
        if filtered_tensor.numel() > 0:
            downsampled_tensor[non_empty_and_non_255_mask] = filtered_tensor
            
        return downsampled_tensor