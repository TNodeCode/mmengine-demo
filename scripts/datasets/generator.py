import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split

def make_dataset(
        base_dir: str,
        dataset_class,
        n_classes,
        dataset_train,
        dataset_val,
        dataset_test,
        log_every=1000,
):
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    # Create the main directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Create subdirectories for each class (there are 102 classes)
    for i in range(n_classes):
        os.makedirs(os.path.join(train_dir, str(i)), exist_ok=True)
        os.makedirs(os.path.join(val_dir, str(i)), exist_ok=True)
        os.makedirs(os.path.join(test_dir, str(i)), exist_ok=True)

    num_train_samples = len(dataset_train)

    if dataset_val is None:
        # Split the dataset into training and validation sets (e.g., 80/20 split)
        num_train_samples, num_val_samples = int(0.8 * len(dataset_train)), int(0.2 * len(dataset_train))
        dataset_train, dataset_val = random_split(dataset_train, [num_train_samples, num_val_samples])

    num_val_samples = len(dataset_val)
    num_test_samples = len(dataset_test)

    print(f"There are {num_train_samples} training images, {num_val_samples} validation images and {num_test_samples} test images in the dataset.")

    # Function to save images
    def save_images(subset, directory, log_every=1000):
        for idx, (image, label) in enumerate(subset):
            if (idx - 1) % log_every == 0:
                print(f"progress [{idx} / {len(subset)}]")
            if image.shape[0] <= 3:
                image = image.permute(1,2,0)
            image_np = image.squeeze().numpy()  # Remove channel dimension
            plt.imsave(os.path.join(directory, str(label), f'{idx}.png'), image_np)

    # Save training images
    print("Generating training images ...")
    save_images(dataset_train, train_dir, log_every)

    # Save validation images
    print("Generating validation images ...")
    save_images(dataset_val, val_dir, log_every)

    # Save test images
    print("Generating test images ...")
    save_images(dataset_test, test_dir, log_every)

    print(f"Dataset saved to '{base_dir}' directory.")
