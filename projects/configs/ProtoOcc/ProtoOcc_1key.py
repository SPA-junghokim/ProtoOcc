_base_ = ['../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
          '../../../mmdetection3d/configs/_base_/default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-40.0, -40.0, -1, 40.0, 40.0, 5.4]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}

grid_config_3dpool = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
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

grid_size = [200, 200, 16]
numC_Trans = 80
depth_categories = 88
num_class = 18
voxel_out_channels = 48

model = dict(
    type='ProtoOcc',
    pc_range = point_cloud_range,
    grid_size = grid_size,
    img_bev_encoder_backbone=None, # for avoiding error during init BEVDet
    img_bev_encoder_neck=None, # for avoiding error during init BEVDet

    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0],
    ),
    depth_net=dict(
        type='CM_DepthNet',
        with_cp = True,
        in_channels=512,
        context_channels=numC_Trans,
        mid_channels=512,
        downsample=16,
        grid_config=grid_config,
        depth_channels=depth_categories,
        loss_depth_weight=3,
        use_dcn=False,
        aspp_mid_channels=96,
    ),
    img_view_transformer=dict(
        type='LSSViewTransformer_depthGT',
        grid_config=grid_config_3dpool,
        input_size=data_config['input_size'],
        in_channels=512,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=False,
        downsample=16,
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
        num_classes=18,
        class_wise=False,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0
        ),
        loss_weight=10.,
    ),
    prototype_query_decoder=dict(
        type='Prototype_Query_Decoder_nuScenes',
        with_cp = True, # for reducing GPU memory usage
        feat_channels=voxel_out_channels,
        out_channels=voxel_out_channels,
        num_queries=18,
        num_occupancy_classes=num_class,
        prototpye_EMA_weight = 0.01,
        RPL_Groups=3,
        RPL_label_noise_ratio=0.0,
        RPL_mask_noise_scale = 0.2,
        shift_noise = [10,10,3],
        mask_noise_scale = 0.1,
        mask_size=[200,200,16],
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
                        feedforward_channels=voxel_out_channels*4,
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
        loss_mask= dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=20.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=1.0),
        train_cfg=dict(
            num_points=12544*3,
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

dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile',ignore_nonvisible=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='LoadLidarsegFromFile',
        grid_config=grid_config,
        occupancy_root="./data/nuscenes/pc_panoptic/",
        learning_map=learning_map,
        label_from='panoptic',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera','sa_gt_depth', 'sa_gt_semantic'],
        meta_keys=['sample_idx','filename', ])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet',
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[29, ])
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# fp16 = dict(loss_scale='dynamic')
load_from = "ckpts/bevdet-r50-4d-depth-cbgs_depthnet_modify.pth"

evaluation = dict(interval=1, start=23, pipeline=test_pipeline)
checkpoint_config = dict(interval=3, max_keep_ckpts=24)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
