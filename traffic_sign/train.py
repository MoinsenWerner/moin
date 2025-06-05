import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import load_gtsrb
from .model import TrafficSignNet


def train_model(data_root: str, epochs: int, lr: float, batch_size: int, device: str, save_path: str):
    train_set, test_set = load_gtsrb(data_root)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # GTSRB has 43 classes
    model = TrafficSignNet(num_classes=43).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                pred = outputs.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        accuracy = 100.0 * correct / total
        print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader):.4f}, "
              f"Test Acc: {accuracy:.2f}%")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Traffic Sign Network")
    parser.add_argument("--data-root", default="./data", help="Dataset path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-path", default="./model.pth")
    args = parser.parse_args()
    train_model(**vars(args))


if __name__ == "__main__":
    main()
