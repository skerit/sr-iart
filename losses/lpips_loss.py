"""LPIPS (Learned Perceptual Image Patch Similarity) loss wrapper for BasicSR training."""

import torch
import torch.nn as nn
import sys
import os

# Add LPIPS to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'PerceptualSimilarity'))
import lpips

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class LPIPSLoss(nn.Module):
    """LPIPS (Learned Perceptual Image Patch Similarity) loss wrapper for BasicSR.
    
    Human-calibrated perceptual loss that matches human perception better than raw VGG/ConvNeXt.
    
    Args:
        loss_weight (float): Weight for this loss. Default: 1.0
        net_type (str): Network type: 'alex', 'vgg', or 'squeeze'. Default: 'alex'
                       - 'alex': Fastest, best performance as metric (recommended)
                       - 'vgg': Closest to traditional perceptual loss for optimization
                       - 'squeeze': Smallest model
        use_gpu (bool): Whether to use GPU. Default: True
        spatial (bool): Return spatial loss map. Default: False (returns scalar)
        version (str): LPIPS version. Default: '0.1' (latest)
        normalize (bool): Whether input is in [0,1] range (True) or [-1,1] (False). Default: True
        reduction (str): Reduction mode: 'mean' | 'sum'. Default: 'mean'
    """
    
    def __init__(self, 
                 loss_weight=1.0, 
                 net_type='alex',
                 use_gpu=True,
                 spatial=False,
                 version='0.1',
                 normalize=True,
                 reduction='mean'):
        super(LPIPSLoss, self).__init__()
        
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.normalize = normalize  # Our tensors are [0,1], LPIPS expects [-1,1]
        
        # Validate network type
        if net_type not in ['alex', 'vgg', 'squeeze']:
            raise ValueError(f"net_type must be one of ['alex', 'vgg', 'squeeze'], got {net_type}")
        
        # Initialize LPIPS model
        self.lpips_model = lpips.LPIPS(
            net=net_type,
            version=version,
            spatial=spatial,
            verbose=True
        )
        
        # Move to GPU if requested
        if use_gpu and torch.cuda.is_available():
            self.lpips_model = self.lpips_model.cuda()
        
        # Set to eval mode (we're not training LPIPS itself)
        self.lpips_model.eval()
        
        # Freeze LPIPS parameters
        for param in self.lpips_model.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target, **kwargs):
        """
        Calculate LPIPS loss.
        
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
            batch_size = b * t
        else:
            batch_size = pred.shape[0]
        
        # Check for very dark patches that might cause issues
        pred_mean = pred.mean()
        target_mean = target.mean()
        
        # Skip if patches are too dark (prevents instability)
        if pred_mean < 0.01 or target_mean < 0.01:
            # Return small constant loss for dark patches
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # LPIPS expects RGB images, handle grayscale if needed
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        # Normalize from [0,1] to [-1,1] if needed
        if self.normalize:
            pred = 2.0 * pred - 1.0
            target = 2.0 * target - 1.0
        
        # Calculate LPIPS distance
        # Note: LPIPS returns a distance (lower = more similar)
        # For training, we want to minimize this distance
        with torch.no_grad():  # LPIPS model is frozen
            lpips_distance = self.lpips_model(pred, target)
        
        # LPIPS returns shape [N, 1, 1, 1], squeeze to get [N]
        lpips_distance = lpips_distance.squeeze()
        
        # Apply reduction
        if self.reduction == 'mean':
            loss = lpips_distance.mean()
        elif self.reduction == 'sum':
            loss = lpips_distance.sum()
        else:
            loss = lpips_distance
        
        # Apply loss weight
        weighted_loss = self.loss_weight * loss
        
        # Ensure gradients can flow (even though LPIPS is frozen, we need gradients for the generator)
        # Create a tensor that requires grad
        weighted_loss = weighted_loss * torch.ones(1, device=pred.device, requires_grad=True)
        
        return weighted_loss[0]