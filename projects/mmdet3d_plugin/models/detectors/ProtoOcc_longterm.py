# Copyright (c) Phigent Robotics. All rights reserved.
from .bevstereo4d import BEVStereo4D
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head
import torch.nn as nn
import torch
from mmdet3d.models import builder
from torch.utils.checkpoint import checkpoint


@DETECTORS.register_module()
class ProtoOcc_Longterm(BEVStereo4D):
    def __init__(self,
                 pc_range = [-40.0, -40.0, -1, 40.0, 40.0, 5.4],
                 grid_size = [200, 200, 16],
                 with_cp = False,
                 align_after_view_transfromation=False,
                 num_adj=None,
                 down_sample_before_bevencoding=None,
                 dual_branch_encoder=None,
                 cnn3d_decoder=None,
                 prototype_query_decoder=None,
                 context_channels = 64,
                 **kwargs):
        super(ProtoOcc_Longterm, self).__init__(**kwargs)
        self.pts_bbox_head = None # useless
        self.pc_range = torch.tensor(pc_range)
        self.grid_size = torch.tensor(grid_size)
        self.with_cp = with_cp
                 
        # for stereo
        self.num_adj = num_adj
        self.num_frame = self.num_adj + 1
        self.temporal_frame = self.num_frame
        self.extra_ref_frame = 1
        self.num_frame += self.extra_ref_frame
        self.align_after_view_transfromation = align_after_view_transfromation

        # for longterm
        self.down_sample_before_bevencoding = nn.Sequential(
            nn.Conv3d(down_sample_before_bevencoding[0], down_sample_before_bevencoding[1], kernel_size=1, padding=0, stride=1,)
        )
        self.PV_class_predictor = nn.Sequential(
                nn.Conv2d(context_channels // 2 , context_channels , kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(context_channels),
                nn.ReLU(),
                nn.Conv2d(context_channels, context_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(context_channels),
                nn.ReLU(),
                nn.Conv2d(context_channels, 18, kernel_size=1, stride=1, padding=0)
                )
        
        # ProtoOcc
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
        
    def prepare_voxel_feat(self, img, sensor2keyego, ego2global, intrin,
                         post_rot, post_tran, bda, mlp_input, feat_prev_iv,
                         k2s_sensor, extra_ref_frame):
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)     # (B*N_views, C_stereo, fH_stereo, fW_stereo)
            return None, None, stereo_feat, None
        # x: (B, N_views, C, fH, fW)
        # stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo)
        x, stereo_feat = self.image_encoder(img, stereo=True)

        # 建立cost volume 所需的信息.
        metas = dict(k2s_sensor=k2s_sensor,     # (B, N_views, 4, 4)
                     intrins=intrin,            # (B, N_views, 3, 3)
                     post_rots=post_rot,        # (B, N_views, 3, 3)
                     post_trans=post_tran,      # (B, N_views, 3)
                     frustum=self.img_view_transformer.cv_frustum.to(x),    # (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                     cv_downsample=4,
                     downsample=self.img_view_transformer.downsample,
                     grid_config=self.img_view_transformer.grid_config,
                     cv_feat_list=[feat_prev_iv, stereo_feat]
                     )
        # voxel_feat: (B, C * Dz(=1), Dy, Dx)
        # depth: (B * N, D, fH, fW)
        voxel_feat, depth, tran_feat = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], metas)
        if self.pre_process:
            voxel_feat = self.pre_process_net(voxel_feat)[0]    # (B, C, Dy, Dx)
        return voxel_feat, depth, stereo_feat, tran_feat

    def prepare_voxel_feat_w_context(self, img, sensor2keyego, ego2global, intrin,
                         post_rot, post_tran, bda, mlp_input, feat_prev_iv,
                         k2s_sensor, extra_ref_frame):
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)     # (B*N_views, C_stereo, fH_stereo, fW_stereo)
            return None, None, stereo_feat, None
        # x: (B, N_views, C, fH, fW)
        # stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo)
        x, stereo_feat = self.image_encoder(img, stereo=True)

        # 建立cost volume 所需的信息.
        metas = dict(k2s_sensor=k2s_sensor,     # (B, N_views, 4, 4)
                     intrins=intrin,            # (B, N_views, 3, 3)
                     post_rots=post_rot,        # (B, N_views, 3, 3)
                     post_trans=post_tran,      # (B, N_views, 3)
                     frustum=self.img_view_transformer.cv_frustum.to(x),    # (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                     cv_downsample=4,
                     downsample=self.img_view_transformer.downsample,
                     grid_config=self.img_view_transformer.grid_config,
                     cv_feat_list=[feat_prev_iv, stereo_feat]
                     )
        # voxel_feat: (B, C * Dz(=1), Dy, Dx)
        # depth: (B * N, D, fH, fW)
        # import pdb; pdb.set_trace()
        voxel_feat, depth, tran_feat = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], metas) # voxel_feat = [8, 1280, 200, 200],depth = [48, 88, 16, 44]
        
        
        if self.pre_process:
            # import pdb; pdb.set_trace()
            voxel_feat = self.pre_process_net(voxel_feat)[0]    # (B, C, Dy, Dx)
        return voxel_feat, depth, stereo_feat, tran_feat
    


    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)
        """Extract features of images."""
        voxel_feat_list = []
        depth_key_frame = None
        pv_feat_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame-1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame-self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)
                if key_frame:
                    voxel_feat, depth, feat_curr_iv, trans_feat = \
                        self.prepare_voxel_feat_w_context(*inputs_curr)
                    depth_key_frame = depth
                    pv_feat_key_frame=trans_feat
                else:
                    with torch.no_grad():
                        voxel_feat, depth, feat_curr_iv, _ = \
                            self.prepare_voxel_feat(*inputs_curr)
                if not extra_ref_frame:
                    voxel_feat_list.append(voxel_feat)
                if not key_frame:
                    feat_prev_iv = feat_curr_iv
        if pred_prev:
            # Todo
            assert False
        if not self.with_prev:
            voxel_feat_key = voxel_feat_list[0]
            if len(voxel_feat_key.shape) ==4:
                b,c,h,w = voxel_feat_key.shape
                voxel_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(voxel_feat_key), voxel_feat_key]
            else:
                b, c, z, h, w = voxel_feat_key.shape
                voxel_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(voxel_feat_key), voxel_feat_key]
        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame-2):
                voxel_feat_list[adj_id] = self.shift_feature(
                    voxel_feat_list[adj_id],
                    [sensor2keyegos[0],
                    sensor2keyegos[self.num_frame-2-adj_id]],
                    bda)
        voxel_feat = torch.cat(voxel_feat_list, dim=1)
        voxel_feat = checkpoint(self.down_sample_before_bevencoding, voxel_feat) if self.with_cp else self.down_sample_before_bevencoding(voxel_feat)
    
        return voxel_feat, depth_key_frame, pv_feat_key_frame

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
                      gt_depth=None,
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
        pv_seg_pred = checkpoint(self.PV_class_predictor, pv_feat) if self.with_cp else self.PV_class_predictor(pv_feat)
        loss_pv_depth = self.img_view_transformer.get_PV_loss(pv_seg_pred, depth, sa_gt_depth, sa_gt_semantic) # auxiliary loss for high performance
        losses.update(loss_pv_depth)
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth
        
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
