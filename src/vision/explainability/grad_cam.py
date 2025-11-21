"""
Grad-CAM and Grad-CAM++ Implementation for Vision Transformers.

This module provides explainability for defect detection models,
enabling visualization of which regions the model focuses on.

Key Features:
1. Grad-CAM: Gradient-weighted Class Activation Mapping
2. Grad-CAM++: Improved gradient weighting for better localization
3. SwinGradCAM: Specialized implementation for Swin Transformers
4. Attention Rollout: Aggregate attention across transformer layers

References:
- Grad-CAM: https://arxiv.org/abs/1610.02391
- Grad-CAM++: https://arxiv.org/abs/1710.11063
- Transformer Explainability: https://arxiv.org/abs/2005.00928
"""

from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CAMResult:
    """Result of CAM computation."""

    heatmap: np.ndarray  # Shape: (H, W), values 0-1
    prediction: int  # Predicted class index
    confidence: float  # Prediction confidence
    class_scores: Dict[int, float]  # All class scores
    target_class: int  # Class used for CAM


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Computes attention heatmaps by combining gradients and activations
    from a target layer. Works with CNN and Vision Transformer models.

    Usage:
        grad_cam = GradCAM(model, target_layer)
        heatmap = grad_cam(input_tensor, target_class=1)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        use_cuda: bool = True,
    ):
        """
        Initialize Grad-CAM.

        Args:
            model: The neural network model
            target_layer: Layer to compute CAM from
            use_cuda: Whether to use CUDA if available
        """
        self.model = model
        self.target_layer = target_layer
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

        # Storage for activations and gradients
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._register_hooks()

        logger.info(f"GradCAM initialized for layer: {type(target_layer).__name__}")

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""

        def forward_hook(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                output = output[0]
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Handle different gradient types
            if isinstance(grad_output, tuple):
                grad_output = grad_output[0]
            self.gradients = grad_output.detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(
            backward_hook
        )

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> CAMResult:
        """
        Compute Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (B, C, H, W) or (C, H, W)
            target_class: Class index to compute CAM for. If None, uses predicted class.

        Returns:
            CAMResult with heatmap and prediction info
        """
        # Ensure batch dimension
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)

        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Get prediction
        probs = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()

        # Class scores
        class_scores = {i: probs[0, i].item() for i in range(output.shape[1])}

        # Use predicted class if not specified
        if target_class is None:
            target_class = predicted_class

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute CAM
        heatmap = self._compute_cam()

        return CAMResult(
            heatmap=heatmap,
            prediction=predicted_class,
            confidence=confidence,
            class_scores=class_scores,
            target_class=target_class,
        )

    def _compute_cam(self) -> np.ndarray:
        """Compute the CAM from stored activations and gradients."""
        if self.activations is None or self.gradients is None:
            raise RuntimeError("No activations/gradients stored. Run forward pass first.")

        # Global average pooling of gradients to get weights
        # Shape: (B, C, H, W) -> (B, C)
        if self.gradients.dim() == 4:
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        elif self.gradients.dim() == 3:
            # Transformer output: (B, N, C) -> pool over sequence
            weights = self.gradients.mean(dim=1, keepdim=True)
        else:
            weights = self.gradients.mean(dim=-1, keepdim=True)

        # Weighted combination of activations
        if self.activations.dim() == 4:
            # CNN: (B, C, H, W)
            cam = (weights * self.activations).sum(dim=1)
        elif self.activations.dim() == 3:
            # Transformer: (B, N, C) - reshape to spatial
            cam = (weights * self.activations).sum(dim=-1)
            # Reshape to 2D if possible
            num_tokens = cam.shape[1]
            side = int(np.sqrt(num_tokens))
            if side * side == num_tokens:
                cam = cam.view(cam.shape[0], side, side)
            else:
                # Handle non-square or with CLS token
                side = int(np.sqrt(num_tokens - 1))  # Exclude CLS token
                if side * side == num_tokens - 1:
                    cam = cam[:, 1:].view(cam.shape[0], side, side)
                else:
                    # Fallback: reshape to closest square
                    cam = cam[:, :side*side].view(cam.shape[0], side, side)
        else:
            cam = self.activations

        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def __del__(self):
        """Clean up hooks."""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ with improved gradient weighting.

    Uses second-order gradients for better localization,
    especially for multiple instances of the same class.
    """

    def _compute_cam(self) -> np.ndarray:
        """Compute Grad-CAM++ heatmap with improved weighting."""
        if self.activations is None or self.gradients is None:
            raise RuntimeError("No activations/gradients stored.")

        # For Grad-CAM++, we need second-order gradient information
        # Simplified implementation using gradient magnitudes

        if self.gradients.dim() == 4:
            # CNN case
            grad_2 = self.gradients ** 2
            grad_3 = grad_2 * self.gradients

            # Alpha weights
            sum_activations = self.activations.sum(dim=(2, 3), keepdim=True)
            alpha_num = grad_2
            alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
            alpha = alpha_num / alpha_denom

            # Weighted gradients
            weights = (alpha * F.relu(self.gradients)).sum(dim=(2, 3), keepdim=True)

            # Compute CAM
            cam = (weights * self.activations).sum(dim=1)

        else:
            # Fallback to regular Grad-CAM for transformers
            return super()._compute_cam()

        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


class SwinGradCAM:
    """
    Specialized Grad-CAM for Swin Transformer models.

    Handles the hierarchical structure of Swin Transformers
    and provides attention-aware explanations.

    Features:
    1. Layer-wise CAM extraction from each Swin stage
    2. Attention rollout across transformer layers
    3. Patch-level to pixel-level upsampling
    """

    def __init__(
        self,
        model: nn.Module,
        use_cuda: bool = True,
    ):
        """
        Initialize SwinGradCAM.

        Args:
            model: Swin Transformer model (from timm or custom)
            use_cuda: Whether to use CUDA
        """
        self.model = model
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

        # Find the target layers in Swin backbone
        self.target_layers = self._find_swin_layers()

        # Storage
        self.layer_activations: Dict[str, torch.Tensor] = {}
        self.layer_gradients: Dict[str, torch.Tensor] = {}

        # Register hooks on all target layers
        self._register_hooks()

        logger.info(f"SwinGradCAM initialized with {len(self.target_layers)} target layers")

    def _find_swin_layers(self) -> List[Tuple[str, nn.Module]]:
        """Find suitable layers in Swin Transformer for CAM."""
        layers = []

        # Look for backbone
        backbone = getattr(self.model, 'backbone', self.model)

        # Swin has layers structured as: layers[i].blocks[j]
        # We target the norm layers after each stage
        if hasattr(backbone, 'layers'):
            for i, layer in enumerate(backbone.layers):
                # Each layer has blocks with attention
                if hasattr(layer, 'blocks') and len(layer.blocks) > 0:
                    # Use the last block's norm layer
                    last_block = layer.blocks[-1]
                    if hasattr(last_block, 'norm2'):
                        layers.append((f"layer_{i}_norm", last_block.norm2))
                    elif hasattr(last_block, 'norm1'):
                        layers.append((f"layer_{i}_norm", last_block.norm1))

        # Also add the final norm if exists
        if hasattr(backbone, 'norm'):
            layers.append(("final_norm", backbone.norm))

        # Fallback: if no Swin structure found, try to find any suitable layer
        if not layers:
            for name, module in backbone.named_modules():
                if isinstance(module, nn.LayerNorm) and 'norm' in name.lower():
                    layers.append((name, module))
                    if len(layers) >= 4:
                        break

        return layers

    def _register_hooks(self) -> None:
        """Register hooks on target layers."""
        self.handles = []

        for name, layer in self.target_layers:
            def make_forward_hook(layer_name):
                def forward_hook(module, input, output):
                    self.layer_activations[layer_name] = output.detach()
                return forward_hook

            def make_backward_hook(layer_name):
                def backward_hook(module, grad_input, grad_output):
                    if isinstance(grad_output, tuple):
                        self.layer_gradients[layer_name] = grad_output[0].detach()
                    else:
                        self.layer_gradients[layer_name] = grad_output.detach()
                return backward_hook

            fh = layer.register_forward_hook(make_forward_hook(name))
            bh = layer.register_full_backward_hook(make_backward_hook(name))
            self.handles.extend([fh, bh])

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        layer_weights: Optional[List[float]] = None,
    ) -> CAMResult:
        """
        Compute multi-layer Grad-CAM for Swin Transformer.

        Args:
            input_tensor: Input image tensor
            target_class: Target class for CAM
            layer_weights: Weights for combining layer CAMs (default: equal)

        Returns:
            CAMResult with combined heatmap
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)

        # Clear previous activations
        self.layer_activations.clear()
        self.layer_gradients.clear()

        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Get predictions
        probs = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()
        class_scores = {i: probs[0, i].item() for i in range(output.shape[1])}

        if target_class is None:
            target_class = predicted_class

        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # Compute CAM for each layer and combine
        cams = []
        input_size = input_tensor.shape[2:]  # (H, W)

        for name, _ in self.target_layers:
            if name in self.layer_activations and name in self.layer_gradients:
                cam = self._compute_layer_cam(
                    self.layer_activations[name],
                    self.layer_gradients[name],
                    input_size,
                )
                cams.append(cam)

        # Combine CAMs with weights
        if not cams:
            # Fallback: return zeros
            heatmap = np.zeros((input_size[0], input_size[1]))
        else:
            if layer_weights is None:
                # Equal weights, but emphasize later layers
                layer_weights = [i + 1 for i in range(len(cams))]

            # Normalize weights
            total = sum(layer_weights[:len(cams)])
            layer_weights = [w / total for w in layer_weights[:len(cams)]]

            # Weighted combination
            heatmap = np.zeros_like(cams[0])
            for cam, weight in zip(cams, layer_weights):
                heatmap += weight * cam

            # Normalize final heatmap
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return CAMResult(
            heatmap=heatmap,
            prediction=predicted_class,
            confidence=confidence,
            class_scores=class_scores,
            target_class=target_class,
        )

    def _compute_layer_cam(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Compute CAM for a single layer.

        Args:
            activations: Layer activations
            gradients: Layer gradients
            target_size: Output size (H, W)

        Returns:
            Normalized CAM array
        """
        # Handle transformer output: (B, N, C)
        if activations.dim() == 3:
            # Global average of gradients
            weights = gradients.mean(dim=1, keepdim=True)  # (B, 1, C)

            # Weighted sum
            cam = (weights * activations).sum(dim=-1)  # (B, N)

            # Remove batch dim
            cam = cam[0]  # (N,)

            # Reshape to 2D
            num_tokens = cam.shape[0]
            side = int(np.sqrt(num_tokens))

            if side * side != num_tokens:
                # Has CLS token, remove it
                if (side + 1) * (side + 1) == num_tokens + 1:
                    cam = cam[1:]  # Remove CLS
                    side = int(np.sqrt(cam.shape[0]))
                else:
                    # Just take what we can
                    side = int(np.sqrt(num_tokens))
                    cam = cam[:side*side]

            cam = cam.view(side, side)

        elif activations.dim() == 4:
            # CNN style: (B, C, H, W)
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1)
            cam = cam[0]
        else:
            return np.zeros(target_size)

        # Apply ReLU
        cam = F.relu(cam)

        # Upsample to target size
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        cam = F.interpolate(
            cam,
            size=target_size,
            mode='bilinear',
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def get_attention_rollout(
        self,
        input_tensor: torch.Tensor,
        head_fusion: str = "mean",
    ) -> np.ndarray:
        """
        Compute attention rollout across transformer layers.

        This provides an alternative explanation based on attention flow.

        Args:
            input_tensor: Input image tensor
            head_fusion: How to combine attention heads ("mean", "max", "min")

        Returns:
            Attention rollout heatmap
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device)

        # Get attention weights from each layer
        attentions = []
        backbone = getattr(self.model, 'backbone', self.model)

        # Hook to capture attention
        def attention_hook(module, input, output):
            if hasattr(module, 'attn_drop'):
                # This is an attention module
                attentions.append(output.detach())

        # Register temporary hooks
        hooks = []
        for name, module in backbone.named_modules():
            if 'attn' in name.lower() and hasattr(module, 'forward'):
                h = module.register_forward_hook(attention_hook)
                hooks.append(h)

        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)

        # Remove hooks
        for h in hooks:
            h.remove()

        if not attentions:
            logger.warning("No attention weights captured")
            return np.zeros((224, 224))

        # Rollout: multiply attention matrices
        # Start with identity (equal attention to all tokens)
        result = torch.eye(attentions[0].shape[-1]).to(self.device)

        for attention in attentions:
            # Fuse heads
            if head_fusion == "mean":
                attention_fused = attention.mean(dim=1)
            elif head_fusion == "max":
                attention_fused = attention.max(dim=1)[0]
            else:
                attention_fused = attention.min(dim=1)[0]

            # Add residual connection and renormalize
            attention_fused = attention_fused + torch.eye(attention_fused.shape[-1]).to(self.device)
            attention_fused = attention_fused / attention_fused.sum(dim=-1, keepdim=True)

            # Multiply with previous result
            result = result @ attention_fused

        # Extract CLS token attention to all patches
        rollout = result[0, 0, 1:].cpu().numpy()  # Skip CLS token

        # Reshape to 2D
        num_patches = rollout.shape[0]
        side = int(np.sqrt(num_patches))
        if side * side == num_patches:
            rollout = rollout.reshape(side, side)
        else:
            # Pad or truncate
            rollout = rollout[:side*side].reshape(side, side)

        # Normalize
        if rollout.max() > 0:
            rollout = (rollout - rollout.min()) / (rollout.max() - rollout.min() + 1e-8)

        return rollout

    def __del__(self):
        """Clean up hooks."""
        if hasattr(self, 'handles'):
            for h in self.handles:
                h.remove()


def overlay_heatmap(
    image: Union[np.ndarray, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay CAM heatmap on original image.

    Args:
        image: Original image (H, W, 3) or PIL Image
        heatmap: CAM heatmap (H, W)
        alpha: Blending factor
        colormap: Matplotlib colormap name

    Returns:
        Blended image as numpy array
    """
    import matplotlib.pyplot as plt

    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure image is in 0-255 range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # Resize heatmap to match image
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap_resized = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (image.shape[1], image.shape[0]),
                Image.BILINEAR
            )
        ) / 255.0
    else:
        heatmap_resized = heatmap

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # Remove alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend
    blended = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)

    return blended
