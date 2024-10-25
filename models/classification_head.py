import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS


@MODELS.register_module()
class ClassificationHead(nn.Module):
    '''
    Defining a class for the classification head network.
    '''
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        '''
        super().__init__()

        if len(hidden_layers) > 0:
            # Add the first layer, input to a hidden layer.
            self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
            
            # Add a variable number of more hidden layers.
            layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
            self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
            
            # the last layer of the head maps the outputs from the previous layer to the numer of classes
            self.output = nn.Linear(hidden_layers[-1], output_size)
        else:
            self.hidden_layers = []
            self.output = nn.Linear(input_size, output_size)

        # classification function
        self.cls = nn.LogSoftmax(dim=1)
        
        # dropout helps the model to generalize better
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output log softmax.
            Arguments
            ---------
            self: all layers
            x: tensor vector
        '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout.   
        # fill the missing gaps     
        for h in self.hidden_layers:
            x = F.relu(h(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return self.cls(x)