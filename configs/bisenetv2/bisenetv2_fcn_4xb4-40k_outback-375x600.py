_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/outback.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (375, 600)
#crop_size = (600, 960)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
"""
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=300000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=300000,
        end=400000,
        by_epoch=False,
    )
]
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
""" # my comments
# my code
"""
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4),
    type='OptimWrapper'
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        begin=0,
        end=1500,
        by_epoch=False
    ),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=100000,
        by_epoch=False
    )
]
"""
# my code
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4),
    type='OptimWrapper'
)
param_scheduler = [
  dict(type='LinearLR', start_factor=0.1, begin=0, end=1000, by_epoch=False),
  dict(type='PolyLR',  eta_min=1e-4, power=1.0, begin=1000, end=10000, by_epoch=False),
]
# train_cfg.max_iters = 20_000  (so LR is flat after 10k)

# base LR from optimizer = 0.01

#optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer) #my comment
#train_dataloader = dict(batch_size=2, num_workers=4) #my comment
train_dataloader = dict(batch_size=4, num_workers=4)# my code
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
