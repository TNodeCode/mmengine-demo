img_shape = 224

train_dataloader=dict(
  batch_size=32,
  num_workers=0,
  sampler=dict(
    type='DefaultSampler',
    shuffle=True
  ),
  dataset=dict(
      type='ImageDataset',
      data_prefix='data/train',
      pipeline=[
          dict(type='RandomResizedCrop', size=img_shape, scale=(0.8, 1.2)),
          dict(type='RandomRotation', degrees=15),
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
  batch_size=32,
  num_workers=0,
  sampler=dict(
    type='DefaultSampler',
    shuffle=True
  ),
  dataset=dict(
      type='ImageDataset',
      data_prefix='data/val',
      pipeline=[
          dict(type='Resize', size=(img_shape, img_shape)),
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
  batch_size=32,
  num_workers=0,
  sampler=dict(
    type='DefaultSampler',
    shuffle=True
  ),
  dataset=dict(
      type='ImageDataset',
      data_prefix='data/test',
      pipeline=[
          dict(type='Resize', size=(img_shape, img_shape)),
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