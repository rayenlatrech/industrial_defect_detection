import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Basic Grad-CAM implementation for CNN-based models.

    Usage:
        grad_cam = GradCAM(model, target_layer)
        heatmap = grad_cam.generate(input_tensor, target_class)

    - model: your trained model (e.g., SimpleCastingCNN)
    - target_layer: the convolutional layer to inspect (e.g. model.features[12])
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self._forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # output: feature maps from target conv layer
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0]: gradients w.r.t. activations
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._forward_hook.remove()
        self._backward_hook.remove()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> torch.Tensor:
        """
        Generate Grad-CAM heatmap for given input and target class.

        :param input_tensor: Tensor of shape (1, 3, H, W)
        :param target_class: index of target class (int).
                             If None, uses the predicted class.
        :return: heatmap tensor of shape (H, W), values in [0, 1]
        """
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(input_tensor)  # shape (1, num_classes)

        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        # Scalar value for the target class
        target_score = outputs[0, target_class]

        # Backward to get gradients at target layer
        target_score.backward()

        # gradients: (1, C, H', W')
        # activations: (1, C, H', W')
        gradients = self.gradients  # d(score)/d(feature_maps)
        activations = self.activations

        # Global average pooling over spatial dimensions to get weights per channel
        # shape: (C,)
        alpha = gradients.mean(dim=(2, 3))[0]

        # Weighted sum of activations
        # activations[0]: (C, H', W')
        weighted = (alpha[:, None, None] * activations[0]).sum(dim=0)  # (H', W')

        # Apply ReLU
        heatmap = F.relu(weighted)

        # Normalize to [0, 1]
        heatmap -= heatmap.min()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        # Return heatmap as 2D tensor (H', W')
        return heatmap
