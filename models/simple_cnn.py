import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_block import ConvBlock
from mmengine.model import BaseModel
from mmengine.registry import MODELS


@MODELS.register_module()
class SimpleCNN(BaseModel):
    """
    see: https://mmengine.readthedocs.io/en/latest/tutorials/model.html
    """
    def __init__(self, blocks, classifier):
        super(SimpleCNN, self).__init__()
        self.blocks = nn.Sequential(*list(map(lambda conf: ConvBlock(**conf), blocks)))
        self.head = nn.Sequential(*list(map(lambda conf: nn.Linear(**conf), classifier)))
        self.prob = nn.Softmax(dim=1)

    def forward(self, *input, **kwargs):
        # Prepare arguments
        if 'mode' in kwargs.keys():
            imgs, labels, mode = kwargs['imgs'], kwargs['labels'], kwargs['mode']
            x = torch.stack(imgs)
        else:
            x, labels, mode = input[0], torch.tensor([-1]), None
        
        # Forward inputs
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        x = self.prob(x)
        
        # Build response
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, torch.tensor(labels))}
        elif mode == 'predict':
            return list(map(lambda v: {'gt_label': torch.tensor([v[0]]), 'pred_score': v[1]}, zip(labels, x)))
        else:
            return x
        