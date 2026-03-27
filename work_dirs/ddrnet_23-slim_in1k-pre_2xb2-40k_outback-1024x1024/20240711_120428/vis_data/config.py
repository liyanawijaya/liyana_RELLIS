checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/ddrnet/pretrain/ddrnet23s-in1kpre_3rdparty-1ccac5b1.pth'
class_weight = [
    0.0,
    1.0736,
    1.1913,
    0.9226,
    1.0146,
    1.2848,
    1.0349,
    1.0133,
    1.3928,
    1.1049,
    1.2267,
    0.9274,
    1.1085,
    1.9832,
    1.0217,
    1.5756,
    1.1232,
]
crop_size = (
    1024,
    1024,
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
        1024,
        1024,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = '../GANav-offroad/data/outback/'
dataset_type = 'OUTBACKDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=4000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
iters = 40000
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        align_corners=False,
        channels=32,
        in_channels=3,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/ddrnet/pretrain/ddrnet23s-in1kpre_3rdparty-1ccac5b1.pth',
            type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        ppm_channels=128,
        type='DDRNet'),
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
            1024,
            1024,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=64,
        dropout_ratio=0.0,
        in_channels=128,
        loss_decode=[
            dict(
                class_weight=[
                    0.0,
                    1.0736,
                    1.1913,
                    0.9226,
                    1.0146,
                    1.2848,
                    1.0349,
                    1.0133,
                    1.3928,
                    1.1049,
                    1.2267,
                    0.9274,
                    1.1085,
                    1.9832,
                    1.0217,
                    1.5756,
                    1.1232,
                ],
                loss_weight=1.0,
                min_kept=131072,
                thres=0.9,
                type='OhemCrossEntropy'),
            dict(
                class_weight=[
                    0.0,
                    1.0736,
                    1.1913,
                    0.9226,
                    1.0146,
                    1.2848,
                    1.0349,
                    1.0133,
                    1.3928,
                    1.1049,
                    1.2267,
                    0.9274,
                    1.1085,
                    1.9832,
                    1.0217,
                    1.5756,
                    1.1232,
                ],
                loss_weight=0.4,
                min_kept=131072,
                thres=0.9,
                type='OhemCrossEntropy'),
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=17,
        type='DDRHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=40000,
        eta_min=0,
        power=0.9,
        type='PolyLR'),
]
randomness = dict(seed=304)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='test.txt',
        data_prefix=dict(img_path='image', seg_map_path='annotation'),
        data_root='../GANav-offroad/data/outback/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                2048,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='OUTBACKDataset'),
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
        2048,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=40000, type='IterBasedTrainLoop', val_interval=40000)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='train.txt',
        data_prefix=dict(img_path='image', seg_map_path='annotation'),
        data_root='../GANav-offroad/data/outback/',
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
                    2048,
                    1024,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75,
                crop_size=(
                    1024,
                    1024,
                ),
                type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='OUTBACKDataset'),
    num_workers=2,
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
            2048,
            1024,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        1024,
        1024,
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
        ann_file='val.txt',
        data_prefix=dict(img_path='image', seg_map_path='annotation'),
        data_root='../GANav-offroad/data/outback/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                2048,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='OUTBACKDataset'),
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
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/ddrnet_23-slim_in1k-pre_2xb2-40k_outback-1024x1024'
