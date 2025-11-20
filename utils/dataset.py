import os
from typing import Tuple, Optional, Callable, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CastingDataset(Dataset):
    """
    PyTorch Dataset for the casting defect detection task.

    Expects a folder structure like:

    root/
        def_front/
            img1.png
            ...
        ok_front/
            imgA.png
            ...

    Each subfolder name becomes a class.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        :param root_dir: Path to directory containing class subfolders (def_front, ok_front)
        :param transform: Optional image transform
        :param target_transform: Optional label transform
        :param class_names: Optional explicit ordering of class names.
                            If None, will infer from subfolder names (sorted).
        """
        self.root_dir = root_dir

        # If class_names not provided, infer from folders
        if class_names is None:
            # Only folders
            class_names = [
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
            class_names.sort()

        self.class_names = class_names
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        self.samples = []  # list of (image_path, label_idx)

        for cls_name in self.class_names:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)

                if not os.path.isfile(fpath):
                    continue

                # Basic image extension filter
                ext = os.path.splitext(fname)[1].lower()
                if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                    continue

                label_idx = self.class_to_idx[cls_name]
                self.samples.append((fpath, label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load grayscale image and convert to RGB (3 channels) for CNNs/ViTs
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label
