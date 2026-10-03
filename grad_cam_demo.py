import os
from typing import List

import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import CastingDataset
from utils.grad_cam import GradCAM
from models.cnn_classifier import SimpleCastingCNN


# Same normalization stats used in training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("✅ Using GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        print("⚠️  GPU not available, using CPU.")
        return torch.device("cpu")


def create_test_dataset(test_root: str) -> CastingDataset:
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    dataset = CastingDataset(root_dir=test_root, transform=test_transform)
    print(f"Test dataset size: {len(dataset)}")
    print(f"Classes: {dataset.class_names}")
    return dataset


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


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor (3, H, W) back to a uint8 RGB image (H, W, 3).
    """
    # clone to avoid modifying in-place
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
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert to 0-255 uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply color map (JET)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay: alpha * heatmap + (1-alpha) * image
    overlay = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    return overlay


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

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

    output_dir = os.path.join(project_root, "grad_cam_outputs")
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()

    # 1) Dataset (no DataLoader needed, we'll index manually)
    test_dataset = create_test_dataset(test_root)

    # 2) Load model
    model = load_model(ckpt_path, device=device)

    # 3) Set up Grad-CAM on the last conv layer of the CNN
    # For SimpleCastingCNN, the last Conv2d is at index 12 in model.features
    target_layer = model.features[12]
    grad_cam = GradCAM(model, target_layer)

    # 4) Pick some indices to visualize
    # Here we just take the first 10 images from the test set
    num_to_visualize = min(10, len(test_dataset))
    indices = list(range(num_to_visualize))

    class_names = test_dataset.class_names

    for idx in indices:
        img_tensor, label = test_dataset[idx]  # img_tensor: (3, H, W)
        input_batch = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

        # Forward pass to get prediction
        with torch.no_grad():
            outputs = model(input_batch)
            probs = torch.softmax(outputs, dim=1)
            pred_class = probs.argmax(dim=1).item()
            pred_prob = probs[0, pred_class].item()

        true_class_name = class_names[label]
        pred_class_name = class_names[pred_class]

        # 5) Generate Grad-CAM heatmap for the predicted class
        # Note: GradCAM.generate does a forward+backward internally,
        # so we don't use no_grad here.
        input_batch_for_cam = img_tensor.unsqueeze(0).to(device)
        heatmap = grad_cam.generate(input_batch_for_cam, target_class=pred_class)
        heatmap_np = heatmap.cpu().numpy()

        # 6) Denormalize original image for visualization
        img_np = denormalize(img_tensor)  # (H, W, 3), uint8

        # 7) Overlay heatmap
        overlay = overlay_heatmap_on_image(img_np, heatmap_np, alpha=0.5)

        # 8) Save original + overlay side by side
        combined = np.concatenate([img_np, overlay], axis=1)

        out_fname = f"idx_{idx:04d}_true_{true_class_name}_pred_{pred_class_name}_prob_{pred_prob:.3f}.png"
        out_path = os.path.join(output_dir, out_fname)

        Image.fromarray(combined).save(out_path)

        print(f"Saved Grad-CAM visualization: {out_path}")

    grad_cam.remove_hooks()
    print("\n✅ Grad-CAM demo finished. Check the 'grad_cam_outputs' folder.")


if __name__ == "__main__":
    main()
