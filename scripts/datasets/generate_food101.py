from torchvision.datasets import Food101
from generator import make_dataset
import torchvision.transforms as transforms

transform = transforms.ToTensor()


# Create dataset
make_dataset(
    base_dir='data/flowers102',
    n_classes=101,
    log_every=50,
    dataset_train=Food101(root='data', split='test', download=True, transform=transform),
    dataset_val=None,
    dataset_test=Food101(root='data', split='train', download=True, transform=transform),
)
