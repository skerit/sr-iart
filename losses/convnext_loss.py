"""ConvNeXt Perceptual Loss wrapper for BasicSR training."""

import torch
import torch.nn as nn
import sys
import os

# Add ConvNeXt perceptual loss to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'convnext_perceptual_loss'))
from convnext_perceptual_loss import ConvNextPerceptualLoss, ConvNextType

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ConvNextLoss(nn.Module):
    """ConvNeXt Perceptual Loss wrapper for BasicSR.
    
    More stable than VGG perceptual loss, using modern ConvNeXt architecture.
    
    Args:
        loss_weight (float): Weight for this loss. Default: 1.0
        model_type (str): ConvNeXt model size: 'tiny', 'small', 'base', 'large'. Default: 'tiny'
        feature_layers (list): Which layers to use for features. Default: [0, 2, 4, 6]
        use_gram (bool): Whether to use Gram matrix (for style). Default: False
        layer_weight_decay (float): Decay factor for layer weights. Default: 0.9
        reduction (str): Reduction mode: 'mean' | 'sum'. Default: 'mean'
        input_range (tuple): Expected input range. Default: (0, 1) for normalized tensors
    """
    
    def __init__(self, 
                 loss_weight=1.0, 
                 model_type='tiny',
                 feature_layers=None,
                 use_gram=False,
                 layer_weight_decay=0.9,
                 reduction='mean',
                 input_range=(0, 1)):
        super(ConvNextLoss, self).__init__()
        
        self.loss_weight = loss_weight
        self.reduction = reduction
        
        # Default feature layers based on model type
        if feature_layers is None:
            # Use early-to-mid layers for better stability
            # ConvNeXt has ~15 blocks total
            feature_layers = [0, 2, 4, 6]  # Conservative default
        
        # Map string to ConvNextType enum
        model_type_map = {
            'tiny': ConvNextType.TINY,
            'small': ConvNextType.SMALL, 
            'base': ConvNextType.BASE,
            'large': ConvNextType.LARGE
        }
        
        if model_type not in model_type_map:
            raise ValueError(f"model_type must be one of {list(model_type_map.keys())}")
        
        # Initialize ConvNeXt perceptual loss
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.convnext_loss = ConvNextPerceptualLoss(
            device=device,
            model_type=model_type_map[model_type],
            feature_layers=feature_layers,
            feature_weights=None,  # Let it auto-calculate with decay
            use_gram=use_gram,
            input_range=input_range,
            layer_weight_decay=layer_weight_decay
        )
        
        # Move to same device
        self.convnext_loss = self.convnext_loss.to(device)
    
    def forward(self, pred, target, **kwargs):
        """
        Calculate ConvNeXt perceptual loss.
        
        Args:
            pred (Tensor): Predicted tensor with shape (B, C, H, W) or (B, T, C, H, W)
            target (Tensor): Target tensor with same shape as pred
            
        Returns:
            Tensor: Calculated loss
        """
        # Handle video tensors (B, T, C, H, W)
        if pred.dim() == 5:
            b, t, c, h, w = pred.shape
            pred = pred.view(b * t, c, h, w)
            target = target.view(b * t, c, h, w)
        
        # Check for very dark patches that might cause issues
        pred_mean = pred.mean()
        target_mean = target.mean()
        
        # Skip if patches are too dark (prevents instability)
        if pred_mean < 0.01 or target_mean < 0.01:
            # Return small constant loss for dark patches
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Calculate ConvNeXt perceptual loss
        loss = self.convnext_loss(pred, target)
        
        # Apply loss weight
        weighted_loss = self.loss_weight * loss
        
        # Apply reduction if needed (ConvNeXt loss already returns scalar)
        if self.reduction == 'mean':
            return weighted_loss
        elif self.reduction == 'sum':
            return weighted_loss
        else:
            return weighted_loss