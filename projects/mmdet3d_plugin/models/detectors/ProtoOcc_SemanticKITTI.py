# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVDet
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head
import torch
from mmdet3d.models import builder

 
@DETECTORS.register_module()
class ProtoOcc_SemanticKITTI(BEVDet):
    def __init__(self,
                 pc_range = [0, -25.6, -2, 51.2, 25.6, 4.4],
                 grid_size = [128, 128, 16],
                 dual_branch_encoder=None,
                 cnn3d_decoder=None,
                 prototype_query_decoder=None,
                 **kwargs):
        super(ProtoOcc_SemanticKITTI, self).__init__(**kwargs)
        self.pts_bbox_head = None # useless
        self.pc_range = torch.tensor(pc_range)
        self.grid_size = torch.tensor(grid_size)


        self.dual_branch_encoder = builder.build_backbone(dual_branch_encoder)
        self.cnn3d_decoder = build_head(cnn3d_decoder)
        self.prototype_query_decoder = build_head(prototype_query_decoder)

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat
        
    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        x, _ = self.image_encoder(img_inputs[0])

        rots, trans, intrins, post_rots, post_trans, bda = img_inputs[1:7]
        mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
        
        voxel_feat, depth = self.img_view_transformer([x] + geo_inputs)
        voxel_feat = voxel_feat.permute(0,1,4,2,3)
        
        return voxel_feat, depth

    def extract_feat(self, img_inputs, img_metas, **kwargs):
        voxel_feat, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        return voxel_feat, depth

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      non_vis_semantic_voxel=None,
                      **kwargs):
        # 2D to 3D view transformation
        voxel_feat, depth = self.extract_feat(img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        # Dual Branch Encoder (DBE)
        comprehensive_voxel_feature = self.dual_branch_encoder(voxel_feat) 

        # 3d CNN for generating Prototype
        prototoype_occ_pred, mask_feat = self.cnn3d_decoder(comprehensive_voxel_feature.permute(0,1,4,2,3))
            
        # Prototype Query Decoder (PQD)
        B = comprehensive_voxel_feature.shape[0]
        voxel_semantics = kwargs['gt_occ'] 
        mask_camera = torch.ones_like(voxel_semantics).bool() 
        img_metas = [{"pc_range": self.pc_range, "occ_size":self.grid_size} for i in range(B)]
        losses = self.prototype_query_decoder.forward_train(comprehensive_voxel_feature, img_metas, voxel_semantics, mask_camera, mask_feat, prototoype_occ_pred)
        
        # aggregate loss values
        loss_prototype = self.cnn3d_decoder.loss(prototoype_occ_pred, voxel_semantics, mask_camera)
        losses.update(loss_prototype)
        gt_depth = img_inputs[7]
        losses['loss_depth']  = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        
        return losses

    def forward_test(self,
                     points=None,
                     img_inputs=None,
                     img_metas=None,
                     **kwargs):
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas, img_inputs,
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    **kwargs):
        # 2D to 3D view transformation
        voxel_feat, depth = self.extract_feat(img_inputs=img, img_metas=img_metas, **kwargs)

        # Dual Branch Encoder (DBE)
        comprehensive_voxel_feature = self.dual_branch_encoder(voxel_feat)

        # Prototype Query Generator
        prototoype_occ_pred, mask_feat = self.cnn3d_decoder(comprehensive_voxel_feature.permute(0,1,4,2,3))

        # Prototype Query Decoder (PQD)
        B = comprehensive_voxel_feature.shape[0]
        img_metas_occ = [{"pc_range": self.pc_range, "occ_size":self.grid_size * 2} for i in range(B)]
        prototoype_occ_pred, mask_feat = self.cnn3d_decoder(comprehensive_voxel_feature.permute(0,1,4,2,3))
        occ_preds = self.prototype_query_decoder.simple_test(comprehensive_voxel_feature, img_metas_occ, mask_feat, prototoype_occ_pred)

        output = {}
        output['output_voxels'] = torch.tensor(occ_preds[0])
        output['target_voxels'] = kwargs['gt_occ']
        output['output_voxel_refine'] = None

        return output

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        return None
