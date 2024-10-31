from torchvision.datasets import Flowers102
from generator import make_dataset
import torchvision.transforms as transforms

transform = transforms.ToTensor()

# Create dataset
make_dataset(
    base_dir='data/flowers102',
    n_classes=102,
    log_every=50,
    dataset_train=Flowers102(root='data', split='test', download=True, transform=transform),
    dataset_val=Flowers102(root='data', split='val', download=True, transform=transform),
    dataset_test=Flowers102(root='data', split='train', download=True, transform=transform),
)