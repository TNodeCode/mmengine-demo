model=dict(
  type='SimpleCNN',
  blocks=[
    dict(type='ConvBlock', in_channels=3, out_channels=32, out_shape=64, kernel_size=3, pool_size=2),
    dict(type='ConvBlock', in_channels=32, out_channels=64, out_shape=32, kernel_size=3, pool_size=2),
    dict(type='ConvBlock', in_channels=64, out_channels=128, out_shape=16, kernel_size=3, pool_size=2),
    dict(type='ConvBlock', in_channels=128, out_channels=256, out_shape=8, kernel_size=3, pool_size=2),
  ],
  classifier=[
    dict(in_features=4096, out_features=2),
  ]
)

train_dataloader=dict(
  batch_size=8,
  num_workers=0,
  dataset=dict(
      type='ImageDataset',
      data_prefix='data/train',
      pipeline=[
          dict(type='Resize', size=(64, 64)),
          dict(type='ToImage'),
          dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ]
  ),
)

train_cfg = dict(
    type='EpochBasedTrainLoop',  # Specifies the type of training loop
    max_epochs=25,               # Number of epochs to train
    val_interval=1               # Interval for validation (if validation is used)
)

train_evaluator=dict(
    type='Accuracy',
)

val_dataloader=dict(
  batch_size=4,
  num_workers=0,
  dataset=dict(
      type='ImageDataset',
      data_prefix='data/val',
      pipeline=[
          dict(type='Resize', size=(64, 64)),
          dict(type='ToImage'),
          dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ]
  ),
)

val_evaluator=dict(
    type='Accuracy',
)

val_cfg=dict(
    type='ValLoop'
)

test_dataloader=dict(
  batch_size=4,
  num_workers=0,
  dataset=dict(
      type='ImageDataset',
      data_prefix='data/test',
      pipeline=[
          dict(type='Resize', size=(64, 64)),
          dict(type='ToImage'),
          dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ]
  ),
)

test_evaluator=dict(
    type='Accuracy',
)

test_cfg=dict(
    type='TestLoop'
)

optim_wrapper=dict(
  optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
  clip_grad=dict(max_norm=35, norm_type=2),
)

log_config=dict(
  interval=10,
  hooks=[
    dict(type='TextLoggerHook'),
    dict(type='FileLoggerHook', by_epoch=True, interval=1, out_dir='work_dirs')
  ]
)

hooks=[
  dict(type='CheckpointHook', interval=1)
]

work_dir = 'work_dir'
