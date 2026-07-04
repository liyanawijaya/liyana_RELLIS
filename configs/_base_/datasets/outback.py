# dataset settings
dataset_type = 'OUTBACKDataset'
data_root = '../datasets/OUTBACK'
crop_size = (375, 600)
#crop_size = (600, 960)
train_pipeline = [
    #dict(type='LoadImageFromFile'), #my comments
    dict(type='LoadNpyAsImage'), #my code
    dict(type='LoadAnnotations'),
    #dict(type='LoadAuxAnnotations'),   # for the second annotations
    dict(type='LoadAuxAnnotations'),  # loads aux 1

    dict(
        type='LoadAuxAnnotations',
        aux_path_key='seg_map_path_aux2',
        aux_map_key='gt_seg_map_aux2'
    ),  # loads aux 2
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        #scale=(1920, 1080),
        ratio_range=(0.5, 2.0),
        #ratio_range=(1.0, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
    
]
test_pipeline = [
    #dict(type='LoadImageFromFile'), #my comments
    dict(type='LoadNpyAsImage'), #my code
    dict(type='LoadAnnotations'), #my comment
    #dict(type='LoadAuxAnnotations'),   #aux 2nd gt
    dict(type='LoadAuxAnnotations'),  # loads aux 1

    dict(
        type='LoadAuxAnnotations',
        aux_path_key='seg_map_path_aux2',
        aux_map_key='gt_seg_map_aux2'
    ),  # loads aux 2
    dict(type='Resize', scale=(1920, 1080), keep_ratio=False), #my comment
    #dict(type='Resize', scale=(600, 375), keep_ratio=False),
    #dict(type='LoadAnnotations'),
    #my code
    #dict(
    #    type='RandomResize',
    #    scale=(2048, 1024),
    #    ratio_range=(0.5, 2.0),
    #    keep_ratio=True),
    #dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    #dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    #my code
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform

    dict(type='PackSegInputs')
    #dict(type='PackSegInputs', keys=['img_detail', 'img_semantic', 'gt_semantic_seg'])                                                                                                                                                                                                                                           
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            #img_path='outback_DICTA_NW/rgb_final', seg_map_path='outback_DICTA_NW/masks_3_levels_for_rellis'),
            #img_path='outback_DICTA_NW/rgb_final', seg_map_path='outback_DICTA_NW/outback_masks_4_new_full'),
            #img_path='outback_DICTA_NW/rgbd_new', seg_map_path='outback_DICTA_NW/depth_outback_mask_3_levels_new'),
            #img_path='outback_DICTA_NW/rgbd_new', seg_map_path='outback_DICTA_NW/masks_3_levels_for_rellis'),
            ##img_path='outback_DICTA_NW/rgbd_new', seg_map_path='outback_DICTA_NW/outback_rgb_5_level_masks'),
           #img_path='outback_DICTA_NW/rgbd_new', seg_map_path='outback_thesis/masks_3_levels_outback_thesis'),
            img_path='npy', seg_map_path='label/train/masks_3_levels_com',
            seg_map_path_aux='label/train/masks_3_levels_blurred',  
            seg_map_path_aux2='label/train/masks_3_levels'),  
            #img_path='outback_DICTA_NW/rgb_final', seg_map_path='outback_thesis/masks_3_levels_outback_thesis'),
            ###img_path='outback_DICTA_NW/outback_depth_sky_white_latest', seg_map_path='outback_thesis/masks_3_levels_outback_thesis'), #depth original
            #img_path='outback_DICTA_NW/outback_depth_sky_white_new', seg_map_path='outback_DICTA_NW/masks_3_levels_for_rellis'),
        #ann_file='outback_DICTA_NW/train_rgb_final_3.txt',
        #ann_file='rellis/train_rellis_2.txt',
        #ann_file='outback_DICTA_NW/train_rgb_final_new.txt',
        #ann_file='outback_DICTA_NW/mixed_train.txt',
        #ann_file='outback_DICTA_NW/mixed_train.txt',
        ann_file='train_elevation.txt',
        ##ann_file='outback_DICTA_NW/train_rgb_final_3_levels.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            #img_path='rellis/00003_test_rgbd', seg_map_path='rellis/depth_masks_3_levels'),
            #img_path='rellis/RELLIS_DEPTH_TEST_new', seg_map_path='rellis/rellis_3_level_masks_test'),
            #img_path='outback_DICTA_NW/test_data_3/rgb_final', seg_map_path='outback_DICTA_NW/mask_3_levels'),
            ##img_path='outback_DICTA_NW/test_data_3/rgbd_stacked_test_1', seg_map_path='outback_DICTA_NW/test_data_3_mask/outback_rgb_5_levels_test'),
            #img_path='outback_DICTA_NW/rgbd_new', seg_map_path='outback_thesis/masks_3_levels_outback_thesis'),
            img_path='npy', seg_map_path='label/val/masks_3_levels',
            seg_map_path_aux='label/val/masks_3_levels',  
            seg_map_path_aux2='label/val/masks_3_levels'),
            #img_path='outback_DICTA_NW/test_data_3/rgb_final', seg_map_path='outback_thesis/test_masks_3_levels_thesis'),
            ###img_path='outback_DICTA_NW/test_data_3/depth_outback_4_env', seg_map_path='outback_thesis/test_masks_3_levels_thesis'),
            #img_path='outback_DICTA_NW/test_data_3/rgbd_stacked_test_1', seg_map_path='outback_DICTA_NW/test_data_3/test_mask_3_levels_grass'),
            #img_path='outback_DICTA_NW/test_data_3/depth_outback_4_env', seg_map_path='outback_DICTA_NW/mask_3_levels'), #depth original
            #img_path='outback_DICTA_NW/test_data_3/rgb_final', seg_map_path='outback_DICTA_NW/test_data_3_mask/rgb_masks_outback_4_levels'),
        #ann_file='rellis/rellis_test_set_3.txt',
        #ann_file='outback_DICTA_NW/train_rgb_final_ch_5.txt',
        ann_file='val.txt',
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='npy', seg_map_path='label/val/masks_3_levels'),

        ann_file='val.txt',
        pipeline=test_pipeline))

#val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
'''
val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU']),
    dict(type='CustomLossMetric', name='val_loss')  # Make sure this matches your registered metric
]

'''
val_evaluator = [
    dict(
        type='IoUMetric',
        iou_metrics=['mIoU'],
        collect_device='cpu',
        fp_tolerance=0,                # tolerance radius in pixels
        tolerant_classes=[6, 7, 8, 9, 10, 11, 12, 13]        # apply FP tolerance only for these classes
    ),

    #dict(
    #    type='CustomSegmentationMAP',
    #    num_classes=16,
    #    name='mAP'
    #)
    
    dict(
         type='CustomLossMetric',
        name='avg_val_loss'
    )
    
]
#'''
#my code
custom_hooks = [
    dict(type='AverageIterLossHook', interval_to_log=1000)
]

#my code
test_evaluator = val_evaluator
