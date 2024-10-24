
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
    elif item['type'] == 'RandomRotation':
      transforms.append(v2.RandomRotation(degrees=item['degrees']))
    elif item['type'] == 'RandomResizedCrop':
      transforms.append(v2.RandomResizedCrop(size=item['size'], scale=item['scale']))
    elif item['type'] == 'RandomHorizontalFlip':
      transforms.append(v2.RandomHorizontalFlip())
    elif item['type'] == 'RandomVerticalFlip':
      transforms.append(v2.RandomVerticalFlip())
    else:
      raise Exception(f"Transformation {item['type']} not known")

  return v2.Compose(transforms)