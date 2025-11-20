import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCastingCNN(nn.Module):
    """
    A simple CNN for binary classification (OK vs DEFECTIVE) on casting images.

    Input: 3xHxW (RGB)
    Output: logits for 2 classes.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4, W/4

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/8, W/8

            # Block 4 (optional extra depth)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/16, W/16
        )

        # We'll adapt to input size 224x224:
        # After 4x MaxPool(2), spatial size becomes 224/(2^4) = 14
        # So flattened features size = 256 * 14 * 14

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # Now features size = 256 * 7 * 7 = 12544

        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(512, num_classes)
        )
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass up to the penultimate layer.

        Returns a feature vector for each image (batch_size, feature_dim).
        We'll use this for anomaly detection.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # self.classifier = [Linear, ReLU, Dropout, Linear]
        x = self.classifier[0](x)  # first Linear
        x = self.classifier[1](x)  # ReLU

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # flatten all dims except batch
        x = self.classifier(x)
        return x
