_base_ = [
    '../_base_/models/mobilenetv3_large.py',
    '../_base_/datasets/image_dataset.py',
    '../_base_/schedules/train_25_epochs.py',
]

_base_.model.head.num_classes=10
_base_.model.fine_tuning=True
_base_.train_dataloader.dataset.data_prefix="data/mnist/train"
_base_.train_dataloader.batch_size=128
_base_.val_dataloader.dataset.data_prefix="data/mnist/val"
_base_.val_dataloader.batch_size=128
_base_.test_dataloader.dataset.data_prefix="data/mnist/test"
_base_.test_dataloader.batch_size=128

work_dir = 'work_dirs/mobilenetv3_large/mnist'
