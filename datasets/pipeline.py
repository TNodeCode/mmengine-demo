
import torch
from torchvision.transforms import v2


def build_pipeline(pipeline):
  transforms = []

  for item in pipeline:
    if item['type'] == 'Resize':
      transforms.append(v2.Resize(size=item['size']))
    elif item['type'] == 'ToImage':
      transforms.append(v2.ToImage())
      transforms.append(v2.ToDtype(torch.float32, scale=True))
    elif item['type'] == 'Normalize':
      transforms.append(v2.Normalize(mean=item['mean'], std=item['std']))
    else:
      raise Exception(f"Transformation {item['type']} not known")

  return v2.Compose(transforms)