import os
from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from utils.dataset import CastingDataset
from models.cnn_classifier import SimpleCastingCNN


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("✅ Using GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        print("⚠️  GPU not available, using CPU.")
        return torch.device("cpu")


def create_test_loader(
    test_root: str,
    batch_size: int = 32,
) -> DataLoader:
    """
    Create DataLoader for the test set.
    No augmentation here, just deterministic preprocessing.
    """
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = CastingDataset(root_dir=test_root, transform=test_transform)

    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Classes: {test_dataset.class_names}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return test_loader, test_dataset.class_names


def load_best_model(
    ckpt_path: str,
    device: torch.device,
) -> SimpleCastingCNN:
    """
    Load the best saved SimpleCastingCNN model from checkpoint.
    """
    model = SimpleCastingCNN(num_classes=2)
    model.to(device)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch')} with val_acc={checkpoint.get('val_acc'):.4f}")

    model.eval()
    return model


def run_test(
    model: SimpleCastingCNN,
    test_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
):
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
    )

    print("\n=== Test Results ===")
    print(f"Accuracy: {acc:.4f}\n")

    print("Confusion Matrix (rows = true, cols = pred):")
    print(cm)
    print("\nClassification Report:")
    print(report)


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Official test folder of the casting dataset: data/raw/casting_data/test
    test_root = os.path.join(
        project_root,
        "data",
        "raw",
        "casting_data",
        "test",
    )

    ckpt_path = os.path.join(
        project_root,
        "checkpoints",
        "simple_cnn_best.pth",
    )

    device = get_device()
    test_loader, class_names = create_test_loader(test_root=test_root, batch_size=32)
    model = load_best_model(ckpt_path=ckpt_path, device=device)

    run_test(model=model, test_loader=test_loader, device=device, class_names=class_names)


if __name__ == "__main__":
    main()
