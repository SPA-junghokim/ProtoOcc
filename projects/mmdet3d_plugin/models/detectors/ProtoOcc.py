# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVDet
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head
import torch
from mmdet3d.models import builder

 
@DETECTORS.register_module()
class ProtoOcc(BEVDet):
    def __init__(self,
                 pc_range = [-40.0, -40.0, -1, 40.0, 40.0, 5.4],
                 grid_size = [200, 200, 16],
                 depth_net=None,
                 dual_branch_encoder=None,
                 cnn3d_decoder=None,
                 prototype_query_decoder=None,
                 **kwargs):
        super(ProtoOcc, self).__init__(**kwargs)
        self.pts_bbox_head = None # useless
        self.pc_range = torch.tensor(pc_range)
        self.grid_size = torch.tensor(grid_size)

        self.depth_net = build_head(depth_net)

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
        img_inputs = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(img_inputs[0])
        cam_params = img_inputs[1:7]

        mlp_input = self.depth_net.get_mlp_input(*cam_params)
        pv_feat, depth = self.depth_net(x, mlp_input)
        voxel_feat, depth = self.img_view_transformer(depth, pv_feat, [x] + img_inputs[1:7])

        return voxel_feat, depth, pv_feat

    def extract_feat(self, img_inputs, img_metas, **kwargs):
        voxel_feat, depth, pv_feat = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        return voxel_feat, depth, pv_feat

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      voxel_semantics=None,
                      mask_camera = None,
                      sa_gt_depth=None,
                      sa_gt_semantic=None,
                      non_vis_semantic_voxel=None,
                      **kwargs):
        # 2D to 3D view transformation
        voxel_feat, depth, pv_feat = self.extract_feat(img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        # Dual Branch Encoder (DBE)
        comprehensive_voxel_feature = self.dual_branch_encoder(voxel_feat) 

        # 3d CNN for generating Prototype
        prototoype_occ_pred, mask_feat = self.cnn3d_decoder(comprehensive_voxel_feature.permute(0,1,4,2,3))
            
        # Prototype Query Decoder (PQD)
        B = comprehensive_voxel_feature.shape[0]
        img_metas = [{"pc_range": self.pc_range, "occ_size":self.grid_size} for i in range(B)]
        losses = self.prototype_query_decoder.forward_train(comprehensive_voxel_feature, img_metas, voxel_semantics, mask_camera, mask_feat, prototoype_occ_pred)
        
        # aggregate loss values
        loss_prototype = self.cnn3d_decoder.loss(prototoype_occ_pred, voxel_semantics, mask_camera)
        losses.update(loss_prototype)
        loss_depth = self.depth_net.get_PV_loss(pv_feat, depth, sa_gt_depth, sa_gt_semantic)
        losses.update(loss_depth)
        
        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # 2D to 3D view transformation
        voxel_feat, depth, pv_feat = self.extract_feat(img_inputs=img, img_metas=img_metas, **kwargs)

        # Dual Branch Encoder (DBE)
        comprehensive_voxel_feature = self.dual_branch_encoder(voxel_feat)

        # Prototype Query Generator
        prototoype_occ_pred, mask_feat = self.cnn3d_decoder(comprehensive_voxel_feature.permute(0,1,4,2,3))

        # Prototype Query Decoder (PQD)
        B = comprehensive_voxel_feature.shape[0]
        img_metas_occ = [{"pc_range": self.pc_range, "occ_size":self.grid_size} for i in range(B)]
        prototoype_occ_pred, mask_feat = self.cnn3d_decoder(comprehensive_voxel_feature.permute(0,1,4,2,3))
        occ_preds = self.prototype_query_decoder.simple_test(comprehensive_voxel_feature, img_metas_occ, mask_feat, prototoype_occ_pred)
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        return None
