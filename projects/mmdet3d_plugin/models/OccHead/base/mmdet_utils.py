# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import numpy as np
import random

from mmcv.ops import point_sample

import pdb

def denormalize(grid):
    return grid * 2.0 - 1.0

def point_sample_3d(input, points, align_corners=False, **kwargs):
    add_dim = False
    if points.dim() == 3:
        add_dim = True
        points = points.unsqueeze(2).unsqueeze(2)

    output = F.grid_sample(input, denormalize(points), align_corners=align_corners, **kwargs)
    
    if add_dim:
        output = output.squeeze(3).squeeze(3)
    
    return output

def point_sample_2d(input, points, align_corners=False, **kwargs):
    add_dim = False
    if points.dim() == 3:
        add_dim = True
        points = points.unsqueeze(1)
    output = F.grid_sample(input, denormalize(points), align_corners=align_corners, **kwargs)
    
    if add_dim:
        output = output.squeeze(1)
    
    return output

def get_uncertainty(mask_pred, labels):
    if mask_pred.shape[1] == 1:
        gt_class_logits = mask_pred.clone()
    else:
        inds = torch.arange(mask_pred.shape[0], device=mask_pred.device)
        gt_class_logits = mask_pred[inds, labels].unsqueeze(1)
    
    return -torch.abs(gt_class_logits)

def unravel_indices(indices, shape):
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = torch.div(indices, dim, rounding_mode='floor')

    coord = torch.stack(coord[::-1], dim=-1)

    return coord

@torch.no_grad()
def sample_valid_coords_with_frequencies(num_points, gt_labels, 
        gt_masks, sample_weights=None):
    
    assert sample_weights is not None
    sample_weights = torch.tensor(sample_weights).to(gt_masks.device)
    point_sampling_weights = sample_weights[gt_labels].view(-1, 1, 1, 1) * gt_masks
    point_sampling_weights = point_sampling_weights.sum(dim=0).view(-1)
    
    # clamp min_p to ensure enough points can be sampled
    point_indices = torch.multinomial(point_sampling_weights, 
                        num_samples=num_points, replacement=False)
    
    point_coords = unravel_indices(point_indices, shape=gt_masks.shape[1:]).float()
    coords_norm = torch.tensor(gt_masks.shape[1:]).type_as(gt_masks).view(1, 1, -1)
    point_coords = point_coords / (coords_norm - 1).float()
    
    return point_indices, point_coords

@torch.no_grad()
def batch_sample_valid_coords_with_frequencies(num_points, gt_labels_list, 
        gt_masks_list, sample_weights=None):
    
    assert sample_weights is not None
    
    device = gt_labels_list[0].device
    sample_weights = torch.tensor(sample_weights).to(device).float()
    batch_sampling_weights = []
    for gt_labels, gt_masks in zip(gt_labels_list, gt_masks_list):
        num_gt = gt_labels.shape[0]
        point_sampling_weights = sample_weights[gt_labels].view(-1, 1, 1, 1) * gt_masks
        point_sampling_weights = point_sampling_weights.sum(dim=0).view(-1)
        
        point_sampling_weights = point_sampling_weights[None].repeat(num_gt, 1)
        batch_sampling_weights.append(point_sampling_weights)
    
    batch_sampling_weights = torch.cat(batch_sampling_weights, dim=0)
    point_indices = torch.multinomial(batch_sampling_weights, 
                        num_samples=num_points, replacement=False)
    
    point_coords = unravel_indices(point_indices, shape=gt_masks.shape[1:]).float()
    coords_norm = torch.tensor(gt_masks.shape[1:]).type_as(gt_masks).view(1, 1, -1)
    point_coords = point_coords / (coords_norm - 1).float()
    
    return point_indices, point_coords

def get_nusc_lidarseg_point_coords(mask_pred, labels, gt_binary, voxel_coord, num_points, oversample_ratio, 
        importance_sample_ratio, point_cloud_range, mask_camera, padding_mode='border', remove_noise_lidarseg=False, consider_visible_mask=False):

    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    batch_size = mask_pred.shape[0]
    num_oversampled = int(num_points * oversample_ratio) * 2
    num_sampled = int(num_points * oversample_ratio)
    
    point_cloud_range = torch.tensor(point_cloud_range).type_as(mask_pred)
    batch_point_coords = []

    if consider_visible_mask:
        for index, (gt_bin, vis_mask) in enumerate(zip(gt_binary, mask_camera)):
            point_coords = voxel_coord[gt_bin.flatten()][:, :3]
            num_lidarseg = min(num_oversampled // 2, point_coords.shape[0])
            if num_lidarseg < point_coords.shape[0]:
                point_coords = point_coords[torch.randperm(point_coords.shape[0])[:num_lidarseg]]
            num_rand = num_oversampled - point_coords.shape[0]
            if num_rand > 0:
                while True:
                    rand_point_coords = torch.rand((num_rand, 3), device=mask_pred.device)
                    rand_vis_mask = point_sample_3d(vis_mask[None, None].float(), rand_point_coords[None, ..., [2,1,0]]).bool().squeeze(0).squeeze(0)
                    rand_point_coords = rand_point_coords[rand_vis_mask]
                    point_coords = torch.cat((point_coords, rand_point_coords), dim=0)
                    
                    if point_coords.shape[0] >= num_sampled:
                        point_coords = point_coords[:num_sampled]
                        break
            batch_point_coords.extend([point_coords] * labels[index].shape[0])
    else:
        for index, gt_bin in enumerate(gt_binary):
            point_coords = voxel_coord[gt_bin.flatten()][:, :3]
            num_lidarseg = min(num_sampled // 2, point_coords.shape[0])
            if num_lidarseg < point_coords.shape[0]:
                point_coords = point_coords[torch.randperm(point_coords.shape[0])[:num_lidarseg]]
            num_rand = num_sampled - point_coords.shape[0]
            if num_rand > 0:
                rand_point_coords = torch.rand((num_rand, 3), device=mask_pred.device)
                point_coords = torch.cat((point_coords, rand_point_coords), dim=0)
            batch_point_coords.extend([point_coords] * labels[index].shape[0])
        
    point_coords = torch.stack(batch_point_coords, dim=0)
    point_logits = point_sample_3d(mask_pred, point_coords[..., [2, 1, 0]], 
                                padding_mode=padding_mode).squeeze(1)
    
    point_uncertainties = get_uncertainty(point_logits.unsqueeze(1), None)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        batch_size, dtype=torch.long, device=mask_pred.device)
    idx += shift[:, None]
    
    # get top-k points
    point_coords = point_coords.view(-1, 3)[idx.view(-1)].view(batch_size, num_uncertain_points, 3)
    if num_random_points > 0:
        rand_point_coords = torch.rand((batch_size, num_random_points, 3), device=mask_pred.device)
        point_coords = torch.cat((point_coords, rand_point_coords), dim=1)
    
    return point_coords


def get_nusc_lidarseg_point_coords_2d(mask_pred, labels, gt_binary, voxel_coord, num_points, oversample_ratio, 
        importance_sample_ratio, point_cloud_range, mask_camera, padding_mode='border', remove_noise_lidarseg=False, consider_visible_mask=False):

    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1

    batch_size = mask_pred.shape[0]
    num_oversampled = int(num_points * oversample_ratio) * 2
    num_sampled = int(num_points * oversample_ratio)
    
    point_cloud_range = torch.tensor(point_cloud_range).type_as(mask_pred)
    batch_point_coords = []

    if consider_visible_mask:
        for index, (gt_bin, vis_mask) in enumerate(zip(gt_binary, mask_camera)):
            gt_bin = gt_bin.sum(2).bool()
            vis_mask = vis_mask.sum(2).bool().to(torch.uint8)
            point_coords = voxel_coord[gt_bin.flatten()][:, :2]        
            num_lidarseg = min(num_oversampled // 2, point_coords.shape[0])
            if num_lidarseg < point_coords.shape[0]:
                point_coords = point_coords[torch.randperm(point_coords.shape[0])[:num_lidarseg]]
            num_rand = num_oversampled - point_coords.shape[0]
            if num_rand > 0:
                while True:
                    rand_point_coords = torch.rand((num_rand, 2), device=mask_pred.device)
                    rand_vis_mask = point_sample_2d(vis_mask[None, None].float(), rand_point_coords[None, ..., [1,0]]).bool().squeeze(0).squeeze(0)
                    rand_point_coords = rand_point_coords[rand_vis_mask]
                    point_coords = torch.cat((point_coords, rand_point_coords), dim=0)
                    
                    if point_coords.shape[0] >= num_sampled:
                        point_coords = point_coords[:num_sampled]
                        break
            batch_point_coords.extend([point_coords] * labels[index].shape[0])
    else:
        for index, gt_bin in enumerate(gt_binary):
            gt_bin = gt_bin.sum(2).bool()
            point_coords = voxel_coord[gt_bin.flatten()][:, :2]
            num_lidarseg = min(num_oversampled // 2, point_coords.shape[0])
            if num_lidarseg < point_coords.shape[0]:
                point_coords = point_coords[torch.randperm(point_coords.shape[0])[:num_lidarseg]]
            num_rand = num_oversampled - point_coords.shape[0]
            if num_rand > 0:
                rand_point_coords = torch.rand((num_rand, 2), device=mask_pred.device)
                point_coords = torch.cat((point_coords, rand_point_coords), dim=0)
            batch_point_coords.extend([point_coords] * labels[index].shape[0])
        
    point_coords = torch.stack(batch_point_coords, dim=0)
    point_logits = point_sample_2d(mask_pred, point_coords[..., [1, 0]], 
                                padding_mode=padding_mode).squeeze(1)
    
    point_uncertainties = get_uncertainty(point_logits.unsqueeze(1), None)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        batch_size, dtype=torch.long, device=mask_pred.device)
    idx += shift[:, None]
    
    # get top-k points
    point_coords = point_coords.view(-1, 2)[idx.view(-1)].view(batch_size, num_uncertain_points, 2)
    if num_random_points > 0:
        rand_point_coords = torch.rand((batch_size, num_random_points, 2), device=mask_pred.device)
        point_coords = torch.cat((point_coords, rand_point_coords), dim=1)
    
    return point_coords


def get_uncertain_point_coords_3d_with_frequency(
    mask_pred, labels, gt_labels_list, gt_masks_list, sample_weights, num_points,
    oversample_ratio, importance_sample_ratio):
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    batch_size = mask_pred.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    
    point_indices, point_coords = batch_sample_valid_coords_with_frequencies(
        num_sampled, gt_labels_list, gt_masks_list, sample_weights,
    )
    
    if mask_pred.shape[-3:] == gt_masks_list[0].shape[1:]:
        point_logits = torch.gather(mask_pred.view(mask_pred.shape[0], -1), dim=1, index=point_indices)
    else:
        point_logits = point_sample_3d(mask_pred, point_coords[..., [2, 1, 0]], align_corners=True).squeeze(1)
    point_uncertainties = get_uncertainty(point_logits.unsqueeze(1), labels)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        batch_size, dtype=torch.long, device=mask_pred.device)
    idx += shift[:, None]
    
    # get top-k points
    point_indices = point_indices.view(-1)[idx.view(-1)].view(batch_size, num_uncertain_points)
    point_coords = point_coords.view(-1, 3)[idx.view(-1)].view(batch_size, num_uncertain_points, 3)
    
    if num_random_points > 0:
        uniform_sample_weights = np.ones_like(sample_weights)
        rand_indices, rand_coords = batch_sample_valid_coords_with_frequencies(
            num_random_points, gt_labels_list, gt_masks_list, 
            sample_weights=uniform_sample_weights,
        )
        point_indices = torch.cat((point_indices, rand_indices), dim=1)
        point_coords = torch.cat((point_coords, rand_coords), dim=1)
    
    return point_indices, point_coords

def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates.
    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).
    Returns:
        tuple: Filtered results
            - scores (Tensor): The scores after being filtered, \
                shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape \
                (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape \
                (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional): \
                The filtered results. The shape of each item is \
                (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    # torch.sort is actually faster than .topk (at least on GPUs)
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                      f'but get {type(results)}.')
    
    return scores, labels, keep_idxs, filtered_results

def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
    """Extract a multi-scale single image tensor from a multi-scale batch
    tensor based on batch index.
    Note: The default value of detach is True, because the proposal gradient
    needs to be detached during the training of the two-stage model. E.g
    Cascade Mask R-CNN.
    Args:
        mlvl_tensors (list[Tensor]): Batch tensor for all scale levels,
           each is a 4D-tensor.
        batch_id (int): Batch index.
        detach (bool): Whether detach gradient. Default True.
    Returns:
        list[Tensor]: Multi-scale single image tensor.
    """
    assert isinstance(mlvl_tensors, (list, tuple))
    num_levels = len(mlvl_tensors)

    if detach:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id].detach() for i in range(num_levels)
        ]
    else:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id] for i in range(num_levels)
        ]
    return mlvl_tensor_list

def preprocess_panoptic_gt(gt_labels, gt_masks, gt_semantic_seg, num_things,
                           num_stuff, img_metas):
    """Preprocess the ground truth for a image.
    Args:
        gt_labels (Tensor): Ground truth labels of each bbox,
            with shape (num_gts, ).
        gt_masks (BitmapMasks): Ground truth masks of each instances
            of a image, shape (num_gts, h, w).
        gt_semantic_seg (Tensor | None): Ground truth of semantic
            segmentation with the shape (1, h, w).
            [0, num_thing_class - 1] means things,
            [num_thing_class, num_class-1] means stuff,
            255 means VOID. It's None when training instance segmentation.
        img_metas (dict): List of image meta information.
    Returns:
        tuple: a tuple containing the following targets.
            - labels (Tensor): Ground truth class indices for a
                image, with shape (n, ), n is the sum of number
                of stuff type and number of instance in a image.
            - masks (Tensor): Ground truth mask for a image, with
                shape (n, h, w). Contains stuff and things when training
                panoptic segmentation, and things only when training
                instance segmentation.
    """
    num_classes = num_things + num_stuff

    things_masks = gt_masks.pad(img_metas['pad_shape'][:2], pad_val=0)\
        .to_tensor(dtype=torch.bool, device=gt_labels.device)

    if gt_semantic_seg is None:
        masks = things_masks.long()
        return gt_labels, masks

    things_labels = gt_labels
    gt_semantic_seg = gt_semantic_seg.squeeze(0)

    semantic_labels = torch.unique(
        gt_semantic_seg,
        sorted=False,
        return_inverse=False,
        return_counts=False)
    stuff_masks_list = []
    stuff_labels_list = []
    for label in semantic_labels:
        if label < num_things or label >= num_classes:
            continue
        stuff_mask = gt_semantic_seg == label
        stuff_masks_list.append(stuff_mask)
        stuff_labels_list.append(label)

    if len(stuff_masks_list) > 0:
        stuff_masks = torch.stack(stuff_masks_list, dim=0)
        stuff_labels = torch.stack(stuff_labels_list, dim=0)
        labels = torch.cat([things_labels, stuff_labels], dim=0)
        masks = torch.cat([things_masks, stuff_masks], dim=0)
    else:
        labels = things_labels
        masks = things_masks

    masks = masks.long()
    return labels, masks

def preprocess_panoptic_occupancy_gt(gt_occ, num_classes, img_metas, max_objects=None):
    """Preprocess the ground truth for a image.
    Args:
        gt_occ (Tensor | None): Ground truth of semantic
            segmentation with the shape (1, x, y, z).
            255 means VOID. It's None when training instance segmentation.
        img_metas (dict): List of image meta information.
    Returns:
        tuple: a tuple containing the following targets.
            - labels (Tensor): Ground truth class indices for a
                image, with shape (n, ), n is the sum of number
                of stuff type and number of instance in a image.
            - masks (Tensor): Ground truth mask for a image, with
                shape (n, h, w). Contains stuff and things when training
                panoptic segmentation, and things only when training
                instance segmentation.
    """
    
    gt_occ = gt_occ.squeeze(0)
    label_ids = torch.unique(gt_occ)
    random.shuffle(label_ids)
    
    stuff_masks_list = []
    stuff_labels_list = []
    for label in label_ids:
        class_id, instance_id = label // 1e3, label % 1e3
        if class_id >= num_classes:
            continue
        
        stuff_mask = (gt_occ == label)
        stuff_labels_list.append(class_id)
        stuff_masks_list.append(stuff_mask)
        
    assert len(stuff_masks_list) > 0
    if max_objects is not None and len(stuff_masks_list) > max_objects:
        stuff_masks_list = stuff_masks_list[:max_objects]
        stuff_labels_list = stuff_labels_list[:max_objects]
    
    stuff_masks = torch.stack(stuff_masks_list, dim=0).long()
    stuff_labels = torch.stack(stuff_labels_list, dim=0).long()
    
    return stuff_labels, stuff_masks

def preprocess_occupancy_gt(gt_occ, num_classes, 
        with_binary_occupancy=False):
    """Preprocess the ground truth for a image.
    Args:
        gt_occ (Tensor | None): Ground truth of semantic
            segmentation with the shape (1, x, y, z).
            255 means VOID. It's None when training instance segmentation.
        img_metas (dict): List of image meta information.
    Returns:
        tuple: a tuple containing the following targets.
            - labels (Tensor): Ground truth class indices for a
                image, with shape (n, ), n is the sum of number
                of stuff type and number of instance in a image.
            - masks (Tensor): Ground truth mask for a image, with
                shape (n, h, w). Contains stuff and things when training
                panoptic segmentation, and things only when training
                instance segmentation.
    """
    
    gt_occ = gt_occ.squeeze(0)
    semantic_labels = torch.unique(
        gt_occ,
        sorted=False,
        return_inverse=False,
        return_counts=False)
    
    stuff_masks_list = []
    stuff_labels_list = []
    for label in semantic_labels:
        if label >= num_classes:
            continue
        
        stuff_mask = gt_occ == label
        stuff_masks_list.append(stuff_mask)
        stuff_labels_list.append(label)
    
    # define a class for binary occupancy
    if with_binary_occupancy:
        stuff_mask = gt_occ < 17
        stuff_masks_list.append(stuff_mask)
        stuff_labels_list.append(torch.tensor(num_classes - 1).type_as(stuff_labels_list[0]))
        
    binary_mask = gt_occ < 17
    
    assert len(stuff_masks_list) > 0
    stuff_masks = torch.stack(stuff_masks_list, dim=0).long()
    stuff_labels = torch.stack(stuff_labels_list, dim=0).long()
    
    return stuff_labels, stuff_masks, binary_mask

def preprocess_occupancy_gt_KITTI(gt_occ, num_classes,
        with_binary_occupancy=False):
    """Preprocess the ground truth for a image.
    Args:
        gt_occ (Tensor | None): Ground truth of semantic
            segmentation with the shape (1, x, y, z).
            255 means VOID. It's None when training instance segmentation.
        img_metas (dict): List of image meta information.
    Returns:
        tuple: a tuple containing the following targets.
            - labels (Tensor): Ground truth class indices for a
                image, with shape (n, ), n is the sum of number
                of stuff type and number of instance in a image.
            - masks (Tensor): Ground truth mask for a image, with
                shape (n, h, w). Contains stuff and things when training
                panoptic segmentation, and things only when training
                instance segmentation.
    """
    
    gt_occ = gt_occ.squeeze(0)
    semantic_labels = torch.unique(
        gt_occ,
        sorted=False,
        return_inverse=False,
        return_counts=False)

    stuff_masks_list = []
    stuff_labels_list = []
    for label in semantic_labels:
        if label >= num_classes:
            continue
        
        stuff_mask = gt_occ == label
        stuff_masks_list.append(stuff_mask)
        stuff_labels_list.append(label)
    
    # define a class for binary occupancy
    if with_binary_occupancy:
        stuff_mask = (gt_occ > 0) & (gt_occ < 255)
        stuff_masks_list.append(stuff_mask)
        stuff_labels_list.append(torch.tensor(num_classes - 1).type_as(stuff_labels_list[0]))
        
    binary_mask = (gt_occ > 0) & (gt_occ < 255)
    
    assert len(stuff_masks_list) > 0
    stuff_masks = torch.stack(stuff_masks_list, dim=0).long()
    stuff_labels = torch.stack(stuff_labels_list, dim=0).long()
    
    return stuff_labels, stuff_masks, binary_mask

