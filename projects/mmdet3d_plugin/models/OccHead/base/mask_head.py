# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import force_fp32

from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead

# modified due to mmdet version
from .mmdet_utils import preprocess_panoptic_gt

class MaskHead(AnchorFreeHead):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_queries=100,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=[1.0] * 133 + [0.1]),
                 loss_mask=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=20.0),
                 loss_dice=dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     naive_dice=True,
                     loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries

        pixel_decoder.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder)[1]
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        pixel_decoder_type = pixel_decoder.get('type')
        if pixel_decoder_type == 'PixelDecoder' and (
                self.decoder_embed_dims != in_channels[-1]
                or enforce_decoder_input_project):
            self.decoder_input_proj = Conv2d(
                in_channels[-1], self.decoder_embed_dims, kernel_size=1)
        else:
            self.decoder_input_proj = nn.Identity()
        self.decoder_pe = build_positional_encoding(positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, out_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(train_cfg.get('assigner', None))
            self.sampler = build_sampler(
                train_cfg.get('sampler', None), context=self)

        self.class_weight = loss_cls.get('class_weight', None)
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

    def init_weights(self):
        if isinstance(self.decoder_input_proj, Conv2d):
            caffe2_xavier_init(self.decoder_input_proj, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def preprocess_gt(self, gt_labels_list, gt_masks_list, gt_semantic_segs,
                      img_metas):
        num_things_list = [self.num_things_classes] * len(gt_labels_list)
        num_stuff_list = [self.num_stuff_classes] * len(gt_labels_list)
        if gt_semantic_segs is None:
            gt_semantic_segs = [None] * len(gt_labels_list)

        targets = multi_apply(preprocess_panoptic_gt, gt_labels_list,
                              gt_masks_list, gt_semantic_segs, num_things_list,
                              num_stuff_list, img_metas)
        labels, masks = targets
        return labels, masks

    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, img_metas):
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks,
                           img_metas):
        target_shape = mask_pred.shape[-2:]
        if gt_masks.shape[0] > 0:
            gt_masks_downsampled = F.interpolate(
                gt_masks.unsqueeze(1).float(), target_shape,
                mode='nearest').squeeze(1).long()
        else:
            gt_masks_downsampled = gt_masks

        assign_result = self.assigner.assign(cls_score, mask_pred, gt_labels,
                                             gt_masks_downsampled, img_metas)
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones(self.num_queries)

        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list,
             gt_masks_list, img_metas):
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, img_metas):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                           gt_labels_list, gt_masks_list,
                                           img_metas)
        labels = torch.stack(labels_list, dim=0)
        label_weights = torch.stack(label_weights_list, dim=0)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)


        mask_preds = mask_preds[mask_weights > 0]
        target_shape = mask_targets.shape[-2:]

        if mask_targets.shape[0] == 0:
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        mask_preds = F.interpolate(
            mask_preds.unsqueeze(1),
            target_shape,
            mode='bilinear',
            align_corners=False).squeeze(1)

        loss_dice = self.loss_dice(
            mask_preds, mask_targets, avg_factor=num_total_masks)

        h, w = mask_preds.shape[-2:]
        mask_preds = mask_preds.reshape(-1, 1)
        mask_targets = mask_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_preds, 1 - mask_targets, avg_factor=num_total_masks * h * w)

        return loss_cls, loss_mask, loss_dice

    def forward(self, feats, img_metas):
        batch_size = len(img_metas)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        padding_mask = feats[-1].new_ones(
            (batch_size, input_img_h, input_img_w), dtype=torch.float32)
        for i in range(batch_size):
            img_h, img_w, _ = img_metas[i]['img_shape']
            padding_mask[i, :img_h, :img_w] = 0
        padding_mask = F.interpolate(
            padding_mask.unsqueeze(1),
            size=feats[-1].shape[-2:],
            mode='nearest').to(torch.bool).squeeze(1)
        mask_features, memory = self.pixel_decoder(feats, img_metas)
        pos_embed = self.decoder_pe(padding_mask)
        memory = self.decoder_input_proj(memory)
        memory = memory.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        padding_mask = padding_mask.flatten(1)
        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        target = torch.zeros_like(query_embed)
        out_dec = self.transformer_decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=padding_mask)
        out_dec = out_dec.transpose(1, 2)

        all_cls_scores = self.cls_embed(out_dec)
        mask_embed = self.mask_embed(out_dec)
        all_mask_preds = torch.einsum('lbqc,bchw->lbqhw', mask_embed,
                                      mask_features)

        return all_cls_scores, all_mask_preds

    def forward_train(self,
                      feats,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg,
                      gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None

        all_cls_scores, all_mask_preds = self(feats, img_metas)
        gt_labels, gt_masks = self.preprocess_gt(gt_labels, gt_masks,
                                                 gt_semantic_seg, img_metas)

        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks,
                           img_metas)

        return losses

    def simple_test(self, feats, img_metas, **kwargs):
        all_cls_scores, all_mask_preds = self(feats, img_metas)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        img_shape = img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        return mask_cls_results, mask_pred_results
