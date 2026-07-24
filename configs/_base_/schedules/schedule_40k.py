# optimizer
#optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005) # my comment
#my code

# my code
#optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None) #my comment
# learning policy
"""
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=300000,
        end=400000,
        by_epoch=False)
]
""" # my comment
"""
#my code
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
#
"""
"""
#my last code starts
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4),
    type='OptimWrapper'
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        begin=0,
        end=1500,           # (Optional) longer warmup
        by_epoch=False
    ),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=100000,         # Extended training duration
        by_epoch=False
    )
]
# my last code ends
"""
#new code
# max_iters = 10_000
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4),
    type='OptimWrapper'
)
param_scheduler = [
  dict(type='LinearLR', start_factor=0.1, begin=0, end=2000, by_epoch=False),
  dict(type='PolyLR',  eta_min=1e-4, power=1.0, begin=2000, end=1000, by_epoch=False),
]
# train_cfg.max_iters = 20_000  (so LR is flat after 10k)


#new cod end

#my code
# training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000,val_interval=500)
val_cfg = dict(type='ValLoop') #my comment


test_cfg = dict(type='TestLoop')
default_hooks = dict(
timer=dict(type='IterTimerHook'),
#logger=dict(type='LoggerHook', interval=4719, log_metric_by_epoch=False),
logger=dict(type='LoggerHook', interval=500, log_metric_by_epoch=False),
param_scheduler=dict(type='ParamSchedulerHook'),
checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000),
sampler_seed=dict(type='DistSamplerSeedHook'),
visualization=dict(type='SegVisualizationHook'))



                                             