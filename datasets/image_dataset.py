from mmengine.registry import DATASETS
from torch.utils.data import Dataset
from .pipeline import build_pipeline
from PIL import Image
import os

@DATASETS.register_module()
class ImageDataset(Dataset):
    def __init__(self, data_prefix, pipeline):
        self.data_prefix = data_prefix
        self.pipeline = build_pipeline(pipeline)
        self.data = []  # List of (image_path, label) pairs
        self._load_annotations()

    def _load_annotations(self):
        # Example assumes data is structured in folders by class
        for label, class_folder in enumerate(os.listdir(self.data_prefix)):
            class_folder_path = os.path.join(self.data_prefix, class_folder)
            for img_name in os.listdir(class_folder_path):
                self.data.append((os.path.join(class_folder_path, img_name), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')        
        img = self.pipeline(img)  # Apply transformations
        return {'imgs': img, 'labels': label}