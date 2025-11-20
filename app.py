import os
from typing import Tuple

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.neighbors import NearestNeighbors

import streamlit as st

from utils.dataset import CastingDataset
from utils.grad_cam import GradCAM
from models.cnn_classifier import SimpleCastingCNN
from report_generator import (
    InspectionInput,
    generate_report_rule_based,
)


# -----------------------------
# Constants / normalization
# -----------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -----------------------------
# Helper functions
# -----------------------------


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_project_paths() -> Tuple[str, str, str]:
    """
    Returns:
        project_root, train_root, ckpt_path
    """
    project_root = os.path.dirname(os.path.abspath(__file__))

    train_root = os.path.join(
        project_root,
        "data",
        "raw",
        "casting_data",
        "train",
    )

    ckpt_path = os.path.join(
        project_root,
        "checkpoints",
        "simple_cnn_best.pth",
    )

    return project_root, train_root, ckpt_path


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor (3, H, W) back to a uint8 RGB image (H, W, 3).
    """
    img = tensor.clone().cpu().numpy()

    for c in range(3):
        img[c] = img[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]

    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)  # (3, H, W)
    img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
    return img


def overlay_heatmap_on_image(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a heatmap (H, W) on an RGB image (H, W, 3) using OpenCV.
    """
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    return overlay


# -----------------------------
# Model + kNN setup (cached)
# -----------------------------


@st.cache_resource
def load_model_and_knn(train_root: str, ckpt_path: str):
    """
    Load the trained CNN model and fit a kNN model on OK-part features.

    Returns:
        model: SimpleCastingCNN on the correct device
        device: torch.device
        class_names: list of class names
        knn_model: NearestNeighbors fitted on OK features
    """
    device = get_device()

    # 1) Load CNN
    model = SimpleCastingCNN(num_classes=2).to(device)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2) Build dataset & dataloader for training set
    transform = build_transform()
    train_dataset = CastingDataset(root_dir=train_root, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    class_names = train_dataset.class_names
    try:
        ok_label = class_names.index("ok_front")
    except ValueError:
        raise RuntimeError(
            "Could not find 'ok_front' in class_names. "
            f"Got: {class_names}"
        )

    # 3) Extract features for all train samples, then keep only OK ones
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            feats = model.forward_features(inputs)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    X_train = np.concatenate(all_feats, axis=0)
    y_train = np.concatenate(all_labels, axis=0)

    X_train_ok = X_train[y_train == ok_label]

    # 4) Fit kNN on OK features
    knn_model = NearestNeighbors(n_neighbors=5)
    knn_model.fit(X_train_ok)

    return model, device, class_names, knn_model


# -----------------------------
# Streamlit App
# -----------------------------


def main():
    st.set_page_config(
        page_title="Industrial Defect Detection",
        layout="wide",
    )

    st.title("🛠️ Industrial Casting Defect Detection")
    st.write(
        "Upload a casting image to run automated quality inspection.\n\n"
        "- A CNN model predicts whether the part is **OK** or **DEFECTIVE**.\n"
        "- A kNN anomaly detector estimates how far the part is from normal parts.\n"
        "- Grad-CAM highlights the most relevant image regions.\n"
        "- A rule-based report summarizes the inspection results."
    )

    project_root, train_root, ckpt_path = get_project_paths()

    with st.spinner("Loading model and fitting anomaly detector (kNN) on OK parts..."):
        model, device, class_names, knn_model = load_model_and_knn(train_root, ckpt_path)

    st.sidebar.header("Model Info")
    st.sidebar.write(f"Device: `{device}`")
    st.sidebar.write("Classes: " + ", ".join(class_names))
    st.sidebar.write("Anomaly model: kNN (mean distance to OK parts)")

    uploaded_file = st.file_uploader(
        "Upload casting image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )

    if uploaded_file is not None:
        # -------------------------
        # Show uploaded image
        # -------------------------
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.subheader("Uploaded Image")
        st.image(pil_img, caption=uploaded_file.name, use_column_width=True)

        # -------------------------
        # Preprocess image
        # -------------------------
        transform = build_transform()
        img_tensor = transform(pil_img)  # (3, H, W)
        input_batch = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

        # -------------------------
        # Classification
        # -------------------------
        with torch.no_grad():
            outputs = model(input_batch)
            probs = F.softmax(outputs, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            pred_prob = probs[0, pred_idx].item()

        pred_class_name = class_names[pred_idx]

        # -------------------------
        # Anomaly score via kNN
        # -------------------------
        with torch.no_grad():
            feat = model.forward_features(input_batch)  # (1, feat_dim)
        feat_np = feat.cpu().numpy()
        distances, _ = knn_model.kneighbors(feat_np, n_neighbors=5)
        anomaly_score = float(distances.mean())

        # -------------------------
        # Grad-CAM
        # -------------------------
        # last conv layer of SimpleCastingCNN is model.features[12]
        target_layer = model.features[12]
        grad_cam = GradCAM(model, target_layer)

        input_for_cam = img_tensor.unsqueeze(0).to(device)
        heatmap = grad_cam.generate(input_for_cam, target_class=pred_idx)
        heatmap_np = heatmap.cpu().numpy()

        img_vis = denormalize(img_tensor)          # original RGB (H, W, 3)
        overlay = overlay_heatmap_on_image(img_vis, heatmap_np, alpha=0.5)

        grad_cam.remove_hooks()

        # -------------------------
        # Layout: predictions + explainability + report
        # -------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Prediction")
            st.write(f"**Predicted class:** `{pred_class_name}`")
            st.write(f"**Confidence:** `{pred_prob:.3f}`")
            st.write(f"**Anomaly score (kNN):** `{anomaly_score:.3f}`")
            st.caption(
                "Anomaly score is the mean distance to the 5 nearest normal (OK) samples in feature space. "
                "Higher values indicate the part is less similar to typical OK parts."
            )

            st.subheader("Grad-CAM Heatmap")
            st.image(
                overlay,
                caption="Original image with Grad-CAM overlay (red = most important regions)",
                use_column_width=True,
            )

        with col2:
            st.subheader("Inspection Report")

            additional_comment = (
                "Grad-CAM highlights the areas in red as the most relevant regions for the model's decision. "
                "Please pay particular attention to these regions during manual inspection."
            )

            info = InspectionInput(
                image_id=uploaded_file.name,
                predicted_class=pred_class_name,
                predicted_prob=pred_prob,
                anomaly_score=anomaly_score,
                anomaly_model="kNN (mean distance to OK parts)",
                additional_comment=additional_comment,
            )

            report_text = generate_report_rule_based(info)
            st.text(report_text)

    else:
        st.info("👆 Upload an image to start the inspection.")


if __name__ == "__main__":
    main()
