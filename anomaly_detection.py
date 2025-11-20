import os
from typing import Tuple

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.dataset import CastingDataset
from models.cnn_classifier import SimpleCastingCNN


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("✅ Using GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        print("⚠️  GPU not available, using CPU.")
        return torch.device("cpu")


def create_dataloader(root_dir: str, batch_size: int = 64) -> Tuple[DataLoader, list]:
    """
    Create a DataLoader with deterministic transforms (no augmentation).
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    dataset = CastingDataset(root_dir=root_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Loaded dataset from {root_dir}")
    print(f"Size: {len(dataset)} | Classes: {dataset.class_names}")
    return loader, dataset.class_names


def load_model(ckpt_path: str, device: torch.device) -> SimpleCastingCNN:
    model = SimpleCastingCNN(num_classes=2)
    model.to(device)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Loaded model from {ckpt_path} (epoch {checkpoint.get('epoch')}, val_acc={checkpoint.get('val_acc'):.4f})")
    return model


def extract_features(
    model: SimpleCastingCNN,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run all images through the CNN feature extractor (forward_features) and
    return (X, y):
        X: (N, feature_dim)
        y: (N,) integer labels
    """
    model.eval()
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            feats = model.forward_features(inputs)  # (batch, feat_dim)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f"Extracted features: X.shape = {X.shape}, y.shape = {y.shape}")
    return X, y


def evaluate_scores(scores: np.ndarray, y_true: np.ndarray, name: str):
    """
    y_true: binary labels (0 = normal (OK), 1 = anomaly (DEFECT))
    scores: higher means more anomalous
    """
    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    print(f"\n=== {name} ===")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

    train_root = os.path.join(
        project_root,
        "data",
        "raw",
        "casting_data",
        "train",
    )
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

    # 1) Dataloaders
    train_loader, class_names = create_dataloader(train_root, batch_size=64)
    test_loader, _ = create_dataloader(test_root, batch_size=64)

    # We assume class_names like ['def_front', 'ok_front'] (sorted alphabetically)
    print("\nClass names:", class_names)
    try:
        ok_label = class_names.index("ok_front")
        def_label = class_names.index("def_front")
    except ValueError:
        raise RuntimeError(
            "Could not find 'ok_front' or 'def_front' in class_names. "
            f"Got: {class_names}"
        )

    # 2) Load model
    model = load_model(ckpt_path, device=device)

    # 3) Extract features
    X_train, y_train = extract_features(model, train_loader, device=device)
    X_test, y_test = extract_features(model, test_loader, device=device)

    # 4) Build normal (OK) train set for anomaly models
    X_train_ok = X_train[y_train == ok_label]
    print(f"\nUsing {X_train_ok.shape[0]} OK samples for anomaly model training.")

    # 5) Prepare test labels for anomaly detection metrics
    #    0 = normal (OK), 1 = anomaly (DEFECT)
    y_test_binary = (y_test == def_label).astype(int)

    # ---------------------------
    # kNN-based anomaly scoring
    # ---------------------------
    print("\nFitting kNN (NearestNeighbors) on OK features...")
    n_neighbors = 5
    nn_model = NearestNeighbors(n_neighbors=n_neighbors)
    nn_model.fit(X_train_ok)

    distances, _ = nn_model.kneighbors(X_test, n_neighbors=n_neighbors)
    # anomaly score = mean distance to k nearest OK neighbors
    knn_scores = distances.mean(axis=1)

    evaluate_scores(knn_scores, y_test_binary, name="kNN (mean distance)")

    # ---------------------------
    # Isolation Forest
    # ---------------------------
    print("\nFitting IsolationForest on OK features...")
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination="auto",   # estimates proportion of outliers
        random_state=42,
    )
    iso_forest.fit(X_train_ok)

    # decision_function: higher values for more normal points
    iso_raw = iso_forest.decision_function(X_test)
    iso_scores = -iso_raw  # invert so higher means more anomalous

    evaluate_scores(iso_scores, y_test_binary, name="IsolationForest")

    # ---------------------------
    # One-Class SVM
    # ---------------------------
    print("\nFitting One-Class SVM on OK features (this might take a bit)...")
    oc_svm = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.05,   # approx fraction of outliers allowed in training
    )
    oc_svm.fit(X_train_ok)

    svm_raw = oc_svm.decision_function(X_test)  # higher for normal
    svm_scores = -svm_raw

    evaluate_scores(svm_scores, y_test_binary, name="One-Class SVM")

    # Some basic score stats
    print("\nScore statistics on test set:")
    for name, scores in [
        ("kNN", knn_scores),
        ("IsolationForest", iso_scores),
        ("One-Class SVM", svm_scores),
    ]:
        normal_scores = scores[y_test_binary == 0]
        anomaly_scores = scores[y_test_binary == 1]
        print(f"\n{name}:")
        print(f"  Normal (OK)   - mean: {normal_scores.mean():.4f}, std: {normal_scores.std():.4f}")
        print(f"  Anomaly (DEF) - mean: {anomaly_scores.mean():.4f}, std: {anomaly_scores.std():.4f}")


if __name__ == "__main__":
    main()
