import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmengine.registry import MODELS


@MODELS.register_module()
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, out_shape, kernel_size=3, pool_size=2, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.norm = nn.LayerNorm([out_channels, out_shape, out_shape])
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.pool(x)
        return x