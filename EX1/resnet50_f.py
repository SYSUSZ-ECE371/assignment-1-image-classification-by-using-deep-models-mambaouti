_base_ = [
    '/root/autodl-tmp/mmpretrain/configs/_base_/models/resnet50.py',           # 模型设置
    '/root/autodl-tmp/mmpretrain/configs/_base_/datasets/imagenet_bs32.py',    # 数据设置
    '/root/autodl-tmp/mmpretrain/configs/_base_/schedules/imagenet_bs256.py',  # 训练策略设置
    '/root/autodl-tmp/mmpretrain/configs/_base_/default_runtime.py',           # 运行设置
]

# 模型设置
model = dict(
    backbone=dict(
        # frozen_stages=2,
        
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(
        #type='ClsHead',
        num_classes=5,
        type='LinearClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,))
    
)

# 数据设置
data_preprocessor=dict(
    mean=[123.675,116.28,103.53],
    std=[58.395,57.12,57.375],
    to_rgb=True,
    num_classes=5,
)
data_type='ImageNet'
data_root='/root/autodl-tmp/work/data/flower_dataset'
classes=[c.strip() for c in open(f'{data_root}/classes.txt')]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=data_type,
        data_root=data_root,
        data_prefix=f'{data_root}/train',
        ann_file=f'{data_root}/train.txt',
        classes=classes,
        pipeline=[dict(type='LoadImageFromFile'),
                  dict(type='RandomResizedCrop',scale=224),
                  dict(type='RandomFlip',prob=0.5,direction='horizontal'),
                  dict(type='PackInputs')
                  ]
    )
)
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=data_type,
        data_root=data_root,
        data_prefix=f'{data_root}/val',
        ann_file=f'{data_root}/val.txt',
        classes=classes,
        pipeline=[dict(type='LoadImageFromFile'),
                  dict(type='RandomResizedCrop',scale=224),
                  dict(type='RandomFlip',prob=0.5,direction='horizontal'),
                  dict(type='PackInputs')
                  ]
    )
)

val_cfg=dict()
val_evaluator=dict(type='Accuracy',topk=(1,))
# test_dataloader = val_dataloader

# 训练策略设置
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1)
train_cfg=dict(
    by_epoch=True,
    max_epochs=25,
    val_interval=1,
)
#保存最好模型
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,              
        save_best='auto',
        rule='greater'   
    )
)