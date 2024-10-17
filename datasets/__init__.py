from mmengine.registry import METRICS
from .image_dataset import ImageDataset
from mmpretrain.evaluation.metrics import Accuracy

# Re-register MultiLabelMetric in the MMEngine registry
METRICS.register_module()(Accuracy)

