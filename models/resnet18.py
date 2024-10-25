import torch.nn as nn
from torchvision import models
from mmengine.registry import MODELS
from .classification_head import ClassificationHead
from .base_classification_model import BaseClassificationModel


@MODELS.register_module()
class ResNet18(BaseClassificationModel):
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
        classification_head = ClassificationHead(
            input_size = head['input_size'], 
            hidden_layers = head['hidden_layers'], 
            output_size = head['num_classes'], 
            drop_p = head['drop_p']
        )

        # Using a Pretrained Network (You may use a different network here, bit we will use mobilenet)
        self.model = models.resnet18(weights='DEFAULT')
        self.criterion = nn.NLLLoss()

        # Freezing the parameters so we don't backprop through them, 
        # we will backprop through the classifier parameters only later
        for param in self.model.parameters():
            param.requires_grad = fine_tuning

        # Use the classification network as the classification layer in te pretrained network
        self.model.fc = classification_head
        
        