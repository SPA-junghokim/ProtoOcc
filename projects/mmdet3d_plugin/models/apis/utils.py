import pickle
import os
import torch
import numpy as np
import pycocotools.mask as mask_util
import mmcv
import torch.distributed as dist
import shutil
import os.path as osp
import tempfile
import torch.nn.functional as F

from PIL import Image
from mmcv.runner import get_dist_info
from projects.mmdet3d_plugin.utils.semkitti_io import get_inv_map

import pdb

def save_output_semantic_kitti(output_voxels, save_path, 
                    sequence_id, frame_id, raw_img=None, test_mapping=True):
    
    # output_voxels = torch.argmax(output_voxels, dim=0)
    output_voxels = output_voxels.cpu().numpy().reshape(-1)
    # print('truck counts = {}'.format((output_voxels == 4).sum()))
    
    # remap to lidarseg ID
    if test_mapping:
        inv_map = get_inv_map()
        output_voxels = inv_map[output_voxels].astype(np.uint16)
        
        save_folder = "{}/sequences/{}/predictions".format(save_path, sequence_id)
        save_file = os.path.join(save_folder, "{}.label".format(frame_id))
        os.makedirs(save_folder, exist_ok=True)

        with open(save_file, 'wb') as f:
            output_voxels.tofile(f)
            print('\n save to {}'.format(save_file))
    else:
        # for validation, generate both the occupancy & input image for visualization
        output_voxels = output_voxels.astype(np.uint8)
        out_dict = dict(
            output_voxel=output_voxels,
            raw_img=raw_img,
        )
        
        save_folder = "{}/sequences/{}/predictions".format(save_path, sequence_id)
        save_file = os.path.join(save_folder, "{}.pkl".format(frame_id))
        os.makedirs(save_folder, exist_ok=True)
        
        with open(save_file, "wb") as handle:
            pickle.dump(out_dict, handle)
            print("wrote to", save_file)


def save_occ_nusc(pred_f, img_metas, path, visible_mask=None, gt_occ=None, free_id=0, thres_low=0.4, thres_high=0.99):
    """
    visualization saving for paper:
    1. gt
    2. pred_f pred_c
    3. gt visible
    4. pred_f visible
    """
    pred_f = F.softmax(pred_f, dim=1)
    pred_f = pred_f[0].cpu().numpy()  # C W H D
    _, W, H, D = pred_f.shape
    coordinates_3D_f = np.stack(np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing='ij'), axis=-1).reshape(-1, 3) # (W*H*D, 3)
    pred_f = np.argmax(pred_f, axis=0) # (W, H, D)
    occ_pred_f_mask = (pred_f.reshape(-1))!=free_id
    pred_f_save = np.concatenate([coordinates_3D_f[occ_pred_f_mask], pred_f.reshape(-1)[occ_pred_f_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls

    scene_token = img_metas['scene_token']
    lidar_token = img_metas['lidar_token']
    save_path = osp.join(path, scene_token, lidar_token)
    if not osp.exists(save_path):
        os.makedirs(save_path)
    save_pred_f_path = osp.join(save_path, 'pred_f.npy')


    np.save(save_pred_f_path, pred_f_save)



# urgent need: visualize nuscenes
def save_output_nuscenes(img_inputs, output_voxels, output_points, 
        target_points, save_path, scene_token, sample_token, 
        img_filenames, timestamp, scene_name):
    
    rots, trans = img_inputs[1:3]
    num_img = rots.shape[1]
    cam2lidar = np.repeat(np.eye(4)[np.newaxis], repeats=num_img, axis=0)
    cam2lidar[:, :3, :3] = rots[0]
    cam2lidar[:, :3, -1] = trans[0]
    
    output_voxels_labels = output_voxels[0].cpu().numpy().astype(np.uint8)
    output_points_labels = output_points.cpu().numpy().astype(np.uint8)
    target_points = target_points.cpu().numpy().astype(np.float32)
    camera_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    cam_img_size = [480, 270]
    
    img_canvas = []
    for camera_name in camera_names:
        img = Image.open(img_filenames[camera_name])
        img = img.resize(cam_img_size, Image.BILINEAR)
        img_canvas.append(img)
    
    out_dict = dict(
        pred_voxels=output_voxels_labels,
        cam2lidar=cam2lidar,
        img_canvas=img_canvas,
    )
    
    if scene_name is not None:
        save_path = os.path.join(save_path, scene_name)
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, str(timestamp) + '.pkl')
        with open(filepath, "wb") as handle:
            pickle.dump(out_dict, handle)
            print("\nwrote to", filepath)
    
    else:
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, sample_token + '.pkl')
        with open(filepath, "wb") as handle:
            pickle.dump(out_dict, handle)
            print("\nwrote to", filepath)

def save_nuscenes_lidarseg_submission(output_points, save_path, img_metas):
    meta_file = os.path.join(save_path, 'test', 'submission.json')
    if not os.path.exists(meta_file):
        os.makedirs(os.path.join(save_path, 'test'), exist_ok=True)
        meta_info = dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
        )
        import json
        meta = dict(meta=meta_info)
        with open(meta_file, 'w') as f:
            json.dump(meta, f)
    
    save_path = os.path.join(save_path, 'lidarseg', 'test')
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, '{}_lidarseg.bin'.format(img_metas['lidar_token']))
    output_points = output_points.cpu().numpy().astype(np.uint8)
    output_points.tofile(save_file)
    print("\nwrote to", save_file)
    
def collect_results_cpu(result_part, size, tmpdir=None, type='list'):
    rank, world_size = get_dist_info()
    if tmpdir is None:
        MAX_LEN = 512
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    
    if rank != 0:
        return None
    part_list = []
    for i in range(world_size):
        part_file = osp.join(tmpdir, f'part_{i}.pkl')
        part_list.append(mmcv.load(part_file))
    
    if type == 'list':
        ordered_results = []
        for res in part_list:  
            ordered_results.extend(list(res))
        ordered_results = ordered_results[:size]
    
    else:
        raise NotImplementedError
    
    shutil.rmtree(tmpdir)
    
    return ordered_results