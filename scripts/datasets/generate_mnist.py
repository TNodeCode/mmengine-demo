from torchvision.datasets import MNIST
from generator import make_dataset
import torchvision.transforms as transforms

transform = transforms.ToTensor()

# Create dataset
make_dataset(
    base_dir='data/mnist',
    n_classes=10,
    log_every=1000,
    dataset_train=MNIST(root='data', train=True, download=True, transform=transform),
    dataset_val=None,
    dataset_test=MNIST(root='data', train=False, download=True, transform=transform),
    cmap='gray',
)
