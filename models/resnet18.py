import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from .classification_head import ClassificationHead


@MODELS.register_module()
class ResNet18(BaseModel):
    """
    see: https://mmengine.readthedocs.io/en/latest/tutorials/model.html
    """
    def __init__(self, head: dict, fine_tuning: bool = False):
        '''
        Constructor for the model

        Parameters:
            head: a dictionary containing hyperparameters for the classification head
            fine_tuning: whether to train the backbone of the model or use the pretrained weights
        '''
        super(ResNet18, self).__init__()
        
        # Creating classification head module
        self.classification_head = ClassificationHead(
            input_size = head['input_size'], 
            hidden_layers = head['hidden_layers'], 
            output_size = head['num_classes'], 
            drop_p = head['drop_p']
        )

        # Using a Pretrained Network (You may use a different network here, bit we will use mobilenet)
        self.backbone_model = models.resnet18(weights='DEFAULT')

        # Freezing the parameters so we don't backprop through them, 
        # we will backprop through the classifier parameters only later
        for param in self.backbone_model.parameters():
            param.requires_grad = fine_tuning

        # Use the classification network as the classification layer in te pretrained network
        self.backbone_model.classifier = self.classification_head
        

    def forward(self, *input, **kwargs):
        # Prepare arguments
        if 'mode' in kwargs.keys():
            imgs, labels, mode = kwargs['imgs'], kwargs['labels'], kwargs['mode']
            x = torch.stack(imgs)
        else:
            x, labels, mode = input[0], torch.tensor([-1]), None
        
        # Forward inputs
        x = self.backbone_model(x)
        x = self.classification_head(x)
        
        # Build response
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, torch.tensor(labels))}
        elif mode == 'predict':
            return list(map(lambda v: {'gt_label': torch.tensor([v[0]]), 'pred_score': v[1]}, zip(labels, x)))
        else:
            return x
        