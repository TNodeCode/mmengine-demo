import torch
import torch.nn.functional as F
from mmengine.model import BaseModel


class BaseClassificationModel(BaseModel):
    def forward(self, *input, **kwargs):
        # Prepare arguments
        if 'mode' in kwargs.keys():
            imgs, labels, mode = kwargs['imgs'], kwargs['labels'], kwargs['mode']
            x = torch.stack(imgs)
        else:
            x, labels, mode = input[0], torch.tensor([-1]), None
        
        # Forward inputs
        x = self.model(x)
        
        # Build response
        if mode == 'loss':
            return {'loss': self.criterion(x, torch.tensor(labels).to(x.device))}
        elif mode == 'predict':
            return list(map(lambda v: {'gt_label': torch.tensor([v[0]]), 'pred_score': torch.exp(v[1])}, zip(labels, x.detach().clone())))
        else:
            return x
