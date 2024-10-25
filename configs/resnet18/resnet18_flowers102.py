_base_ = [
    '../_base_/models/resnet18.py',
    '../_base_/datasets/image_dataset.py',
    '../_base_/schedules/train_25_epochs.py',
]

_base_.model.head.num_classes=102
_base_.train_dataloader.dataset.data_prefix="data/flowers102/train"
_base_.val_dataloader.dataset.data_prefix="data/flowers102/val"
_base_.test_dataloader.dataset.data_prefix="data/flowers102/test"

work_dir = 'work_dirs/resnet18/flowers102'
