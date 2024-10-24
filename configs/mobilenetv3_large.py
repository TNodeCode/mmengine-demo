model=dict(
  type='MobileNetV3Large',
  head=dict(
      input_size=1000,
      hidden_layers=[],
      num_classes=10,
      drop_p=0.2,
  ),
  fine_tuning=True,
)

train_dataloader=dict(
  batch_size=8,
  num_workers=0,
  dataset=dict(
      type='ImageDataset',
      data_prefix='data/mnist/train',
      pipeline=[
          dict(type='RandomRotation', degrees=30),
          dict(type='RandomResizedCrop', size=224, scale=(0.8, 1.2)),
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
      data_prefix='data/mnist/val',
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
      data_prefix='data/mnist/test',
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

work_dir = 'work_dirs/mobilenetv3_large'
