# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.core import build_assigner, build_sampler, reduce_mean, multi_apply
from mmdet.models.builder import HEADS, build_loss

from .base.mmdet_utils import preprocess_occupancy_gt

from mmcv.cnn import build_norm_layer
from .base.anchor_free_head import AnchorFreeHead
from .base.mask_head import MaskHead
from torch.utils.checkpoint import checkpoint
from mmdet.models.utils import build_transformer
 
@HEADS.register_module()
class Prototype_Query_Decoder_nuScenes(MaskHead):
    def __init__(self,
                 feat_channels,
                 out_channels,
                 num_occupancy_classes=20,
                 num_queries=100,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 post_norm_cfg=dict(type='LN'),
                 query_self_attn_cfg = None,
                 prototpye_EMA_weight = 0.01,
                 RPL_Groups=5,
                 RPL_mask_noise_scale = 0.2,
                 mask_noise_scale = 0.1,
                 shift_noise = [10,10,3],
                 mask_size=[200,200,16],
                 with_cp = False,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.with_cp = with_cp
        self.num_occupancy_classes = num_occupancy_classes
        self.num_classes = self.num_occupancy_classes
        self.num_queries = num_queries
        
        self.post_norm = build_norm_layer(post_norm_cfg, feat_channels)[1]
        
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        self.proto_first_flag = torch.ones(self.num_classes).cuda().bool()

        self.prototpye_EMA_weight = prototpye_EMA_weight
        self.prototype_EMA_feat = nn.Embedding(self.num_classes,32) 
        self.global_protoEMA_agg = nn.Sequential(
            nn.Linear(32, feat_channels), 
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), 
            )
        self.local_protoEMA_agg = nn.Sequential(
            nn.Linear(32, feat_channels), 
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), 
            )
        self.for_query_embed = nn.Sequential(
        nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
        nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
        nn.Linear(feat_channels, feat_channels))
        
        self.cls_embed = nn.Linear(feat_channels, self.num_occupancy_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)
        
        self.preprocess_occupancy_gt = preprocess_occupancy_gt
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        
        self.class_weight = loss_cls.class_weight
        
        # align_corners
        self.align_corners = True
        self.voxel_coord = None
        self.loss_dice = build_loss(loss_dice)
        
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        
        self.RPL_Groups=RPL_Groups
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.RPL_mask_noise_scale = RPL_mask_noise_scale
        self.mask_noise_scale = mask_noise_scale
        self.mask_size = mask_size
        self.shift_noise = torch.tensor(shift_noise).cuda()
        self.cur_iter = 0

        self.query_self_attn = build_transformer(query_self_attn_cfg)

    def init_weights(self):
        pass

    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, gt_binary_list, img_metas):
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list, mask_preds_list, 
                    gt_labels_list, gt_masks_list, gt_binary_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks, gt_binary, img_metas):
        # sample points
        gt_labels = gt_labels.long()
        
        # label target
        labels = gt_labels.new_full((self.num_classes, ), self.num_classes, dtype=torch.long)
        labels[gt_labels] = gt_labels
        labels = labels.repeat(self.num_queries // self.num_classes)
        
        assert labels.shape != self.num_queries
        
        pos_inds = torch.where(labels != float(self.num_classes))[0]
        neg_inds = torch.where(labels == float(self.num_classes))[0]

        label_weights = labels.new_ones(self.num_queries).type_as(cls_score)
        class_weights_tensor = torch.tensor(self.class_weight).type_as(cls_score)
        
        mask_targets = gt_masks
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = class_weights_tensor[labels[pos_inds]]
        return (labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds)
    
    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list, 
             gt_masks_list, gt_binary_list, gt_occ, mask_camera, img_metas, RPL_args):
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_binary_list = [gt_binary_list for _ in range(num_dec_layers)]
        all_mask_camera = [mask_camera for _ in range(num_dec_layers)]
        
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        
        losses_cls, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, all_gt_binary_list, all_mask_camera, img_metas_list)
        
        loss_dict = dict()
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i        
            num_dec_layer += 1
        
        if self.training:
            loss_dict['loss_cls_RPL'] = 0
            loss_dict['loss_mask_RPL'] = 0
            loss_dict['loss_dice_RPL'] = 0
            for cur_group in range(self.RPL_Groups):
                all_cls_scores = [RPL_args['cls_pred_list'][dec][:, (cur_group)*self.num_classes : (cur_group+1)*self.num_classes] for dec in range(num_dec_layers)]
                all_mask_preds = [RPL_args['mask_pred_list'][dec][:, (cur_group)*self.num_classes : (cur_group+1)*self.num_classes] for dec in range(num_dec_layers)]
                losses_cls_RPL, losses_mask_RPL, losses_dice_RPL = multi_apply(
                    self.loss_single, all_cls_scores, all_mask_preds,
                    all_gt_labels_list, all_gt_masks_list, all_gt_binary_list, all_mask_camera, img_metas_list)
                loss_dict['loss_cls_RPL'] += losses_cls_RPL[-1]
                loss_dict['loss_mask_RPL'] += losses_mask_RPL[-1]
                loss_dict['loss_dice_RPL'] += losses_dice_RPL[-1]
                
        return loss_dict


    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, gt_binary_list, mask_camera, img_metas):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos, num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list, gt_labels_list, 
                    gt_masks_list, gt_binary_list, img_metas)
        labels = torch.stack(labels_list, dim=0)
        label_weights = torch.stack(label_weights_list, dim=0)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        loss_cls = None
        if mask_targets.shape[0] == 0:
            loss_mask = mask_preds.sum()
            loss_dice = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        batch_size = len(gt_labels_list)
        
        loss_cls = None
        loss_dice = None
        loss_mask = None
        
        # dice loss
        batch_index = 0
        for bs in range(batch_size):
            
            cls_score = cls_scores[bs]
            label = labels[bs]
            label_weight = label_weights[bs]
            mask_target = mask_targets_list[bs]
            mask_weight = mask_weights[bs]
            
            class_weight = cls_score.new_tensor(self.class_weight)
            if loss_cls is None:
                loss_cls = self.loss_cls(cls_score,label,label_weight,avg_factor=class_weight[labels].sum(),)
            else:
                loss_cls += self.loss_cls(cls_score,label,label_weight,avg_factor=class_weight[labels].sum(),)
            
            num_gt = len(gt_labels_list[bs])
            num_total_masks = reduce_mean(cls_score.new_tensor([num_gt]))
            num_total_masks = max(num_total_masks, 1)
            mask_pred = mask_preds[bs][mask_weight > 0]
            mask_preds_flatten = mask_pred[:, mask_camera[bs].bool()]
            mask_targets_flatten = mask_target[:, mask_camera[bs].bool()]
            
            hwz = mask_preds_flatten.shape[1]
            if loss_dice is None:
                loss_dice = self.loss_dice(mask_preds_flatten, mask_targets_flatten, avg_factor=num_total_masks)
            else:
                loss_dice += self.loss_dice(mask_preds_flatten, mask_targets_flatten, avg_factor=num_total_masks)
            
            mask_preds_flatten = mask_preds_flatten.reshape(-1, 1)
            mask_targets_flatten = mask_targets_flatten.reshape(-1)
            if loss_mask is None:
                loss_mask = self.loss_mask(mask_preds_flatten,1-mask_targets_flatten,avg_factor=num_total_masks*hwz)
            else:
                loss_mask += self.loss_mask(mask_preds_flatten,1-mask_targets_flatten,avg_factor=num_total_masks*hwz)
        return loss_cls, loss_mask, loss_dice

        
    def prepare_for_RPL_shift(self, pred_mask):
        pred_bin_mask = []
        for i in range(self.num_classes):
            cur_mask = pred_mask == i
            pred_bin_mask.append(cur_mask)
        pred_bin_mask = torch.stack(pred_bin_mask, dim=1).repeat(1, self.RPL_Groups, 1,1,1)
        
        bs, c, w, h, z = pred_bin_mask.shape
        masks = pred_bin_mask.reshape(-1, w, h, z).float()
        
        masks = masks > 1e-8
        areas= (masks).flatten(1).sum(1)
        noise_ratio=areas*self.RPL_mask_noise_scale/(self.mask_size[0]*self.mask_size[1]*self.mask_size[2])
        delta_mask = torch.rand_like(masks,dtype=torch.float)<noise_ratio[:,None,None,None]
        masks=torch.logical_xor(masks,delta_mask)
        masks = masks.float()
        delta_masks = ((torch.rand(masks.shape[0],3) * 2 - 1.0).cuda() * self.shift_noise[None]).to(masks)
        is_scale = torch.rand(masks.shape[0])
        new_masks = []
        scale_noise = (1-torch.rand(masks.shape[0]).cuda()*2) * self.mask_noise_scale 
            
        scale_size = (torch.tensor(self.mask_size).float().to(masks)[None]*(1+scale_noise)[:,None]).long()+1
        delta_center = ((torch.tensor(self.mask_size)[None].to(masks)-scale_size)*torch.tensor(0.5)).to(masks)
        scale_size = scale_size.tolist()
        for mask, delta_mask, sc, noise_scale, dc in zip(masks, delta_masks, is_scale, scale_size, delta_center):
            if sc > 0.5:
                mask_scale = F.interpolate(mask[None][None],noise_scale[0], mode="nearest")[0][0]
                x_, y_, z_ = torch.where(mask_scale > 0.5)
                x_+=dc[0].long()
                y_+=dc[1].long()
                z_+=dc[2].long()
            else:
                x_, y_, z_ = torch.where(mask > 0.5)
            x_ = x_.clamp(min=0, max=self.mask_size[-3] - 1)
            y_ = y_.clamp(min=0, max=self.mask_size[-2] - 1)
            z_ = z_.clamp(min=0, max=self.mask_size[-1] - 1)
            mask = torch.zeros_like(mask,dtype=torch.bool)
            mask[x_.long(), y_.long(), z_.long()] = True
            new_masks.append(mask)   
             
        new_masks = torch.stack(new_masks) > 0.5
        new_masks = new_masks.reshape(bs , self.RPL_Groups * self.num_classes, w,h,z)
        RPL_pad_size = self.RPL_Groups * self.num_classes
        single_pad = self.num_classes
        tgt_size = RPL_pad_size + self.num_queries
        self_attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        self_attn_mask[RPL_pad_size:, :RPL_pad_size] = True
        for i in range(self.RPL_Groups):
            self_attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):RPL_pad_size] = True
            self_attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        return new_masks, RPL_pad_size, self_attn_mask
        

    def forward_head(self, decoder_out, mask_feature):
        decoder_out = self.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        cls_pred = self.cls_embed(decoder_out)
        mask_embed = self.mask_embed(decoder_out)
        mask_pred = torch.einsum('bqc,bcxyz->bqxyz', mask_embed, mask_feature)
        return cls_pred, mask_pred

    def preprocess_gt(self, gt_occ, img_metas):
        num_class_list = [self.num_occupancy_classes] * len(img_metas)
        labels, masks, binary_mask = multi_apply(self.preprocess_occupancy_gt, gt_occ, num_class_list)
        return labels, masks, binary_mask

    def forward_train(self,
            voxel_feats,
            img_metas,
            gt_occ,
            mask_camera,
            mask_feat,
            occ_pred,
            **kwargs,
        ):
        gt_labels, gt_masks, gt_binaries = self.preprocess_gt(gt_occ, img_metas, )
        all_cls_scores, all_mask_preds, RPL_args = self(voxel_feats, img_metas, mask_feat, occ_pred, )
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks, gt_binaries, gt_occ, mask_camera, img_metas, RPL_args)

        return losses

    def forward(self, 
            voxel_feats,
            img_metas,
            mask_feat,
            occ_pred=None,
            **kwargs,
        ):
        batch_size = len(img_metas)
        mask_features = voxel_feats.permute(0,1,3,2,4)
        query_feat = self.query_feat.weight

        # Scene-Adaptive Prototype Generator
        mask_target = occ_pred.clone().detach().permute(0,4,1,2,3)
        mask_ = F.softmax(mask_target, dim=1)
        top2_values, top2_indices = torch.topk(mask_, 2, dim=1)
        difference = top2_values[:, 0] - top2_values[:, 1]
        scores = 1.0 - difference

        vis_mask = scores >= 0.9
        mask = mask_.argmax(dim=1)
        for_query = []
        for i in range(self.num_classes):
            mask_cls = mask == i
            mask_cls *= ~vis_mask
            if mask_cls.sum() == 0:
                query_ = mask_feat.new_zeros((1, mask_feat.shape[-1]))
                for_query.append(query_)
            else:
                query_ = mask_feat[mask_cls, :]
                for_query.append(query_.mean(dim=0).unsqueeze(0))
        for_query = torch.cat(for_query)
                
        # Robust Prototype Learning
        RPL_args = {}
        if self.training: 
            new_masks, RPL_pad_size, self_attn_mask  = self.prepare_for_RPL_shift(mask)
            RPL_for_query = []
            for b_ in range(self.RPL_Groups):
                cur_group_for_query = []
                for i in range(self.num_classes):
                    mask_cls = new_masks[:, (b_*self.num_classes) + i]
                    if mask_cls.sum() == 0:
                        query_ = mask_feat.new_zeros((1, mask_feat.shape[-1]))
                        cur_group_for_query.append(query_)
                    else:
                        query_ = mask_feat[mask_cls, :]
                        cur_group_for_query.append(query_.mean(dim=0).unsqueeze(0))
                cur_group_for_query = torch.cat(cur_group_for_query)
                RPL_for_query.append(cur_group_for_query)
            RPL_for_query = torch.cat(RPL_for_query)
            for_query_ = torch.cat((RPL_for_query, for_query), dim=0)
            RPL_query = self.query_feat.weight.repeat(self.RPL_Groups,1)
            query_feat = torch.cat((RPL_query, query_feat),dim=0)
        else:
            self_attn_mask = None
            query_feat = self.query_feat.weight
            RPL_pad_size = 0
            for_query_ = for_query
        

        # Scene-Agnostic Prototype Generator
        if self.training:
            with torch.no_grad():
                cur_query_assign_flag = for_query[-RPL_pad_size:].sum(1) != 0
                no_assign_flag = self.proto_first_flag * cur_query_assign_flag
                assign_query = for_query[-RPL_pad_size:][no_assign_flag]
                if no_assign_flag.sum() != 0:
                    self.prototype_EMA_feat.weight[no_assign_flag] = assign_query
                    self.proto_first_flag[no_assign_flag] = False
                proto_assign_flag = self.proto_first_flag == False
                assign_flag = proto_assign_flag * cur_query_assign_flag
                if assign_flag.shape[0] != 0:
                    self.prototype_EMA_feat.weight[assign_flag] = self.prototype_EMA_feat.weight[assign_flag] * (1-self.prototpye_EMA_weight) + for_query[-RPL_pad_size:][assign_flag].detach()*self.prototpye_EMA_weight
            ema_query = self.prototype_EMA_feat.weight.repeat(self.RPL_Groups+1, 1)
            learnable_query = self.query_feat.weight.repeat(self.RPL_Groups+1,1)
        else:
            ema_query = self.prototype_EMA_feat.weight
            learnable_query = self.query_feat.weight
            
        if self.with_cp:
            global_for_query_ = checkpoint(self.global_protoEMA_agg, ema_query)
            local_for_query_ =  checkpoint(self.local_protoEMA_agg, for_query_)
            query_feat = learnable_query + global_for_query_ + local_for_query_
            query_feat = checkpoint(self.for_query_embed, query_feat).unsqueeze(1).repeat((1, batch_size, 1))
        else:
            global_for_query_ = self.global_protoEMA_agg(ema_query)
            local_for_query_ =  self.local_protoEMA_agg(for_query_)
            query_feat = learnable_query + global_for_query_ + local_for_query_
            query_feat = self.for_query_embed(query_feat).unsqueeze(1).repeat((1, batch_size, 1))

        query_feat = self.query_self_attn(query_feat, [self_attn_mask])
        

        # Preidct final occupancy
        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred = self.forward_head(query_feat, mask_features)

        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        cls_pred_list_ = []
        mask_pred_list_ = []
        RPL_cls_pred_list = []
        RPL_mask_pred_list = []
        for i in range(len(cls_pred_list)):
            cls_pred_list_.append(cls_pred_list[i][:, RPL_pad_size:])
            mask_pred_list_.append(mask_pred_list[i][:, RPL_pad_size:])
            RPL_cls_pred_list.append(cls_pred_list[i][:, :RPL_pad_size])
            RPL_mask_pred_list.append(mask_pred_list[i][:, :RPL_pad_size])
            
        RPL_args['cls_pred_list'] = RPL_cls_pred_list
        RPL_args['mask_pred_list'] = RPL_mask_pred_list

        return cls_pred_list_, mask_pred_list_, RPL_args

    def format_results(self, mask_cls_results, mask_pred_results):
        mask_cls = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        output_voxels = torch.einsum("bqc, bqxyz->bcxyz", mask_cls, mask_pred)
        return output_voxels


    def simple_test(self, 
            voxel_feats,
            img_metas,
            mask_feat,
            occ_pred=None,
            **kwargs,
        ):
        all_cls_scores, all_mask_preds, _ = self(voxel_feats, img_metas, mask_feat, occ_pred)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        
        output_voxels = self.format_results(mask_cls_results, mask_pred_results)
        occ_score = output_voxels.permute(0,2,3,4,1).softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)
        return list(occ_res)
