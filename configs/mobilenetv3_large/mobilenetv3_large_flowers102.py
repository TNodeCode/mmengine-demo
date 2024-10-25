_base_ = [
    '../_base_/models/mobilenetv3_large.py',
    '../_base_/datasets/image_dataset.py',
    '../_base_/schedules/train_25_epochs.py',
]

_base_.model.head.num_classes=102
_base_.model.fine_tuning=True
_base_.train_dataloader.dataset.data_prefix="data/flowers102/train"
_base_.train_dataloader.batch_size=32
_base_.val_dataloader.dataset.data_prefix="data/flowers102/val"
_base_.val_dataloader.batch_size=32
_base_.test_dataloader.dataset.data_prefix="data/flowers102/test"
_base_.test_dataloader.batch_size=32

work_dir = 'work_dirs/mobilenetv3_large/fowers102'
