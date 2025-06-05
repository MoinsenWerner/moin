import os
from torchvision import datasets, transforms

def load_gtsrb(root: str = "./data"):
    """Load the GTSRB dataset with standard transformations."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_set = datasets.GTSRB(root, split="train", download=True, transform=transform)
    test_set = datasets.GTSRB(root, split="test", download=True, transform=transform)
    return train_set, test_set
