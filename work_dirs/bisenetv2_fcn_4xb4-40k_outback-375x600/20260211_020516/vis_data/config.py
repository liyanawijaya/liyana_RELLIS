crop_size = (
    375,
    600,
)
custom_hooks = [
    dict(interval_to_log=1000, type='AverageIterLossHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmseg.models.custom.encoder_decoder_val_depth',
    ])
data_preprocessor = dict(
    depth_channel_idx=5,
    mean=[
        123.675,
        116.28,
        103.53,
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        375,
        600,
    ),
    std=[
        58.395,
        57.12,
        57.375,
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessorWithRawDepth')
data_root = '../GANav-offroad/data/'
dataset_type = 'OUTBACKDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1000, type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=[
        dict(
            align_corners=False,
            channels=1024,
            concat_input=False,
            in_channels=64,
            in_index=1,
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=5,
            num_convs=2,
            type='FCNHead'),
        dict(
            align_corners=False,
            channels=1024,
            concat_input=False,
            in_channels=32,
            in_index=2,
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=5,
            num_convs=2,
            type='FCNHead'),
        dict(
            align_corners=False,
            channels=1024,
            concat_input=False,
            in_channels=16,
            in_index=3,
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=5,
            num_convs=2,
            type='FCNHead'),
    ],
    backbone=dict(
        align_corners=False,
        bga_channels=64,
        detail_channels=(
            64,
            64,
            128,
        ),
        init_cfg=None,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        semantic_channels=(
            16,
            32,
            64,
            128,
        ),
        semantic_expansion_ratio=6,
        type='BiSeNetV2'),
    data_preprocessor=dict(
        depth_channel_idx=5,
        mean=[
            123.675,
            116.28,
            103.53,
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            375,
            600,
        ),
        std=[
            58.395,
            57.12,
            57.375,
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessorWithRawDepth'),
    decode_head=dict(
        align_corners=False,
        channels=1024,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=64,
        in_index=0,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=5,
        num_convs=1,
        type='FCNHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoderWithValLoss')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1500, start_factor=0.1, type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=10000,
        eta_min=0.0001,
        power=1.0,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='test.txt',
        data_prefix=dict(
            img_path='outback_DICTA_NW/test_data_3/rgb_val',
            seg_map_path='outback_DICTA_NW/test_data_3_mask/rgb_val'),
        data_root='../GANav-offroad/data/',
        pipeline=[
            dict(type='LoadNpyAsImage'),
            dict(keep_ratio=False, scale=(
                1920,
                1080,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='OUTBACKDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        collect_device='cpu',
        fp_tolerance=0,
        iou_metrics=[
            'mIoU',
        ],
        tolerant_classes=[
            0,
            3,
            4,
            5,
            6,
            8,
            9,
            10,
        ],
        type='IoUMetric'),
    dict(name='avg_val_loss', type='CustomLossMetric'),
]
test_pipeline = [
    dict(type='LoadNpyAsImage'),
    dict(keep_ratio=False, scale=(
        1920,
        1080,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=10000, type='IterBasedTrainLoop', val_interval=100)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='outback_DICTA_NW/train_complex_thesis.txt',
        data_prefix=dict(
            img_path='outback_DICTA_NW/rgbd_new',
            seg_map_path='outback_thesis/masks_3_levels_outback_thesis'),
        data_root='../GANav-offroad/data/',
        pipeline=[
            dict(type='LoadNpyAsImage'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    2048,
                    1024,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    375,
                    600,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='OUTBACKDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadNpyAsImage'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            2048,
            1024,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        375,
        600,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='outback_DICTA_NW/test_171.txt',
        data_prefix=dict(
            img_path='outback_DICTA_NW/test_data_3/rgbd_stacked_test_1',
            seg_map_path='outback_thesis/test_masks_3_levels_thesis'),
        data_root='../GANav-offroad/data/',
        pipeline=[
            dict(type='LoadNpyAsImage'),
            dict(keep_ratio=False, scale=(
                1920,
                1080,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='OUTBACKDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        collect_device='cpu',
        fp_tolerance=0,
        iou_metrics=[
            'mIoU',
        ],
        tolerant_classes=[
            0,
            3,
            4,
            5,
            6,
            8,
            9,
            10,
        ],
        type='IoUMetric'),
    dict(name='avg_val_loss', type='CustomLossMetric'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/bisenetv2_fcn_4xb4-40k_outback-375x600'
