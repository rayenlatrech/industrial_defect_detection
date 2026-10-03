import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from utils.dataset import CastingDataset
from models.cnn_classifier import SimpleCastingCNN


def get_device() -> torch.device:
    """
    Pick GPU if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        print("✅ Using GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        print("⚠️  GPU not available, using CPU.")
        return torch.device("cpu")


def create_dataloaders(
    train_root: str,
    batch_size: int = 32,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from the training folder.
    We split the 'train' folder into train and val subsets.

    Data augmentation is applied ONLY to the training subset.
    """
    # ---------- TRANSFORMS ----------

    # Train transforms: augmentation + normalization
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),           # a bit larger
        transforms.RandomResizedCrop(224,        # random crop to 224x224
                                     scale=(0.8, 1.0),
                                     ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Validation transforms: no augmentation, just deterministic resize + normalize
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Two views of the same image folder, each with its own transform.
    # (A single dataset shared by both subsets would end up with whichever
    # transform was assigned last, silently disabling augmentation.)
    train_base = CastingDataset(root_dir=train_root, transform=train_transform)
    val_base = CastingDataset(root_dir=train_root, transform=val_transform)

    total_len = len(train_base)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len

    # Same seeded permutation for both views, so train and val never overlap
    perm = torch.randperm(total_len, generator=torch.Generator().manual_seed(42)).tolist()
    train_dataset = Subset(train_base, perm[:train_len])
    val_dataset = Subset(val_base, perm[train_len:])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # increase on machines that support multiprocessing loaders
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Dataset size: {total_len} images")
    print(f"Train: {train_len}, Val: {val_len}")

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Train", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Val", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main():
    # === PATHS ===
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Official train folder of the casting dataset: data/raw/casting_data/train
    train_root = os.path.join(
        project_root,
        "data",
        "raw",
        "casting_data",
        "train"
    )

    # where to save model weights
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoints_dir, "simple_cnn_best.pth")

    device = get_device()

    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3

    train_loader, val_loader = create_dataloaders(
        train_root=train_root,
        batch_size=batch_size,
        val_split=0.2,
    )

    # Instantiate model, loss, optimizer
    model = SimpleCastingCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            print(f"✅ New best model saved with val_acc = {val_acc:.4f}")

    print(f"\nTraining finished. Best val_acc = {best_val_acc:.4f}")
    print(f"Best model checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
