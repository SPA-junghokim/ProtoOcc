_base_ = [
    '../../../projects/configs/_base_/custom_nus-3d.py',
    '../../../projects/configs/_base_/default_runtime.py'
]

point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
camera_used = ['left']
class_names = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
    'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
    'pole', 'traffic-sign',
]
num_class = len(class_names)
occ_size = [256, 256, 32]
lss_downsample = [2, 2, 2]
voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]

data_config = {
    'input_size': (384, 1280),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
    'depth': [2.0, 58.0, 0.5],
}

grid_config_3dpool = {
    'x': [0, 51.2, voxel_x * lss_downsample[0]],
    'y': [-25.6, 25.6, voxel_y * lss_downsample[1]],
    'z': [-2, 4.4, voxel_z * lss_downsample[2]],
    'depth': [2.0, 58.0, 0.5],
}

learning_map = {
                1: 0,   5: 0,   7: 0,   8: 0,
                10: 0,  11: 0,  13: 0,  19: 0,
                20: 0,  0: 0,   29: 0,  31: 0,
                9: 1,   14: 2,  15: 3,  16: 3,
                17: 4,  18: 5,  21: 6,  2: 7,
                3: 7,   4: 7,   6: 7,   12: 8,
                22: 9,  23: 10, 24: 11, 25: 12,
                26: 13, 27: 14, 28: 15, 30: 16,
}

grid_size = [128, 128, 16]
numC_Trans = 80
depth_categories = 112

num_class = 20
voxel_out_channels = 80
pred_upsample=True
model = dict(
    type='ProtoOcc_SemanticKITTI',
    pc_range = point_cloud_range,
    grid_size = grid_size,
    img_bev_encoder_backbone=None, # for avoiding error during init BEVDet
    img_bev_encoder_neck=None, # for avoiding error during init BEVDet
    img_backbone=dict(
        type='CustomEfficientNet',
        arch='b7',
        drop_path_rate=0.2,
        frozen_stages=0,
        norm_eval=False,
        out_indices=(2, 3, 4, 5, 6),
        with_cp=True,
        init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint='ckpts/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth'),
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[48, 80, 224, 640, 2560],
        upsample_strides=[0.25, 0.5, 1, 2, 2],
        out_channels=[128, 128, 128, 128, 128]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShootVoxel_SemanticKITTI',
        numC_input=640,
        cam_channels=33,
        loss_depth_weight=1.0,
        grid_config=grid_config,
        data_config=data_config,
        numC_Trans=numC_Trans,
        ),
        
    dual_branch_encoder=dict(
        type='Dual_Branch_Encoder',
        z_size = [4, 8, 16],
        vox_feat1 = numC_Trans,
        vox_feat2 = voxel_out_channels//2,
        vox_feat3 = voxel_out_channels*4,
        vox_feat4 = voxel_out_channels*8,
        voxel_out_channels=voxel_out_channels,
        down_sample_for_3d_pooling=[numC_Trans*grid_size[2], numC_Trans*2],
        bev_encoder_backbone=dict(
            type='CustomBEVBackbone',
            stride=[2, 2, 2],
            numC_input=numC_Trans*2,
            num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
        bev_encoder_neck=dict(
            type='Custom_FPN_LSS',
            catconv_in_channels1=numC_Trans * 8 + numC_Trans * 4,
            catconv_in_channels2=numC_Trans * 2 + voxel_out_channels * 2,
            out_channels=voxel_out_channels,
            input_feature_index=(0, 1, 2),
            ),
    ),
    cnn3d_decoder=dict(
        type='cnn3d_decoder',
        in_dim=voxel_out_channels,
        out_dim=32,
        use_mask=True,
        num_classes=20,
        class_wise=False,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0
        ),
        loss_geo=True,
        loss_sem=True,
        loss_weight=10.,
        pred_upsample=pred_upsample,
    ),
    prototype_query_decoder=dict(
        type='Prototype_Query_Decoder_KITTI',
        with_cp = True,
        is_kitti = True,
        pred_upsample = pred_upsample,
        feat_channels=voxel_out_channels,
        out_channels=voxel_out_channels,
        num_queries=20,
        num_occupancy_classes=num_class,
        prototpye_EMA_weight = 0.01,
        RPL_Groups=2,
        RPL_label_noise_ratio=0.0,
        RPL_mask_noise_scale = 0.2,
        shift_noise = [10,10,3],
        mask_noise_scale = 0.1,
        mask_size=[128,128,16],
        query_self_attn_cfg=dict(
            type='Query_Transformer_RPL',
            decoder=dict(
                type='Query_TransformerDecoder_RPL',
                num_layers=1,
                transformerlayers=dict(
                    type='Query_Transformer_DecoderLayer_self_RPL',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=voxel_out_channels,
                            num_heads=8,
                            dropout=0.1),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=voxel_out_channels,
                        feedforward_channels=voxel_out_channels * 4,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    batch_first=True,
                    with_cp=True,
                    operation_order=['self_attn', 'norm',]),
            )),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_class + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544*4,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='MaskHungarianAssigner',
                cls_cost=dict(type='ClassificationCost', weight=2.0),
                mask_cost=dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dice_cost=dict(
                    type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
                sampler=dict(type='MaskPseudoSampler'),
            ),
        test_cfg=dict(
                semantic_on=True,
                panoptic_on=False,
                instance_on=False),
    ),
)

dataset_type = 'CustomSemanticKITTILssDataset'
data_root = 'data/SemanticKITTI'
ann_file = 'data/SemanticKITTI/labels'

bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0.5,)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_SemanticKitti', is_train=True,
            data_config=data_config, img_norm_cfg=img_norm_cfg),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='kitti'),
    dict(type='LoadSemKittiAnnotation', bda_aug_conf=bda_aug_conf, 
            is_train=True, point_cloud_range=point_cloud_range),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ'], 
         meta_keys=['pc_range', 'occ_size',]),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_SemanticKitti', is_train=False, 
            data_config=data_config, img_norm_cfg=img_norm_cfg),
    dict(type='LoadSemKittiAnnotation', bda_aug_conf=bda_aug_conf,
            is_train=False, point_cloud_range=point_cloud_range),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ'], 
         meta_keys=['pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img']),
]


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

test_config=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    split='test',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        split='train',
        camera_used=camera_used,
        occ_size=occ_size,
        pc_range=point_cloud_range,
    ),
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(type='AdamW', lr=4e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    step=[20, 25],
)
runner = dict(type='EpochBasedRunner', max_epochs=20)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# fp16 = dict(loss_scale='dynamic')
load_from = 'ckpts/occformer_kitti.pth'

evaluation = dict(interval=1, start=20, pipeline=test_pipeline)
checkpoint_config = dict(interval=5, max_keep_ckpts=20)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
