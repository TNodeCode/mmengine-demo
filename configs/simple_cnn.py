_base_ = [
    './_base_/datasets/image_dataset.py',
    './_base_/schedules/train_25_epochs.py',
]

_base_.train_dataloader.dataset.data_prefix="data/mnist/train"
_base_.train_dataloader.dataset.pipeline[0]=dict(type='Resize', size=(64, 64))
_base_.train_dataloader.batch_size=128
_base_.val_dataloader.dataset.data_prefix="data/mnist/val"
_base_.val_dataloader.dataset.pipeline[0]=dict(type='Resize', size=(64, 64))
_base_.val_dataloader.batch_size=128
_base_.test_dataloader.dataset.data_prefix="data/mnist/test"
_base_.test_dataloader.dataset.pipeline[0]=dict(type='Resize', size=(64, 64))
_base_.test_dataloader.batch_size=128

model=dict(
  type='SimpleCNN',
  blocks=[
    dict(type='ConvBlock', in_channels=3, out_channels=32, out_shape=64, kernel_size=3, pool_size=2),
    dict(type='ConvBlock', in_channels=32, out_channels=64, out_shape=32, kernel_size=3, pool_size=2),
    dict(type='ConvBlock', in_channels=64, out_channels=128, out_shape=16, kernel_size=3, pool_size=2),
    dict(type='ConvBlock', in_channels=128, out_channels=256, out_shape=8, kernel_size=3, pool_size=2),
  ],
  classifier=[
    dict(in_features=4096, out_features=10),
  ]
)

work_dir = 'work_dirs/simple_cnn'
