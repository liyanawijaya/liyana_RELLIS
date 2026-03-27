crop_size = (
    600,
    600,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        600,
        600,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = '../GANav-offroad/data/rellis/'
dataset_type = 'RELLISNWDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=True, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'work_dirs/bisenetv2_fcn_4xb4-40k_rellis-1920x600/iter_100000.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=[
        dict(
            align_corners=False,
            channels=16,
            concat_input=False,
            in_channels=16,
            in_index=1,
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=19,
            num_convs=2,
            type='FCNHead'),
        dict(
            align_corners=False,
            channels=64,
            concat_input=False,
            in_channels=32,
            in_index=2,
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=19,
            num_convs=2,
            type='FCNHead'),
        dict(
            align_corners=False,
            channels=256,
            concat_input=False,
            in_channels=64,
            in_index=3,
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=19,
            num_convs=2,
            type='FCNHead'),
        dict(
            align_corners=False,
            channels=1024,
            concat_input=False,
            in_channels=128,
            in_index=4,
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=19,
            num_convs=2,
            type='FCNHead'),
    ],
    backbone=dict(
        align_corners=False,
        bga_channels=128,
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
            4,
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
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            600,
            600,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=1024,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=128,
        in_index=0,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=19,
        num_convs=1,
        type='FCNHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.05, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.05, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=100000, start_factor=0.1,
        type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=100000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='our_test.txt',
        data_prefix=dict(
            img_path='Rellis_fusion_full_NW', seg_map_path='annotation'),
        data_root='../GANav-offroad/data/rellis/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                1920,
                1200,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='RELLISNWDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        1920,
        1200,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=100000, type='IterBasedTrainLoop', val_interval=10000)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='our_train.txt',
        data_prefix=dict(
            img_path='Rellis_fusion_full_NW', seg_map_path='annotation'),
        data_root='../GANav-offroad/data/rellis/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1920,
                    1200,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    600,
                    600,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='RELLISNWDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            1920,
            1200,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        600,
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
        ann_file='our_val.txt',
        data_prefix=dict(
            img_path='Rellis_fusion_full_NW', seg_map_path='annotation'),
        data_root='../GANav-offroad/data/rellis/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                1920,
                1200,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='RELLISNWDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    save_dir='work_dirs/vis_preds',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/bisenetv2_fcn_4xb4-40k_rellis-1920x600'
