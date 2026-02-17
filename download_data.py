import torch
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download training set
train_dataset = datasets.KMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Download test set
test_dataset = datasets.KMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)