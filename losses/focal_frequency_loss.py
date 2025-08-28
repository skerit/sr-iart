"""Focal Frequency Loss integration for IART - CORRECTED VERSION.

Based on: Focal Frequency Loss for Image Reconstruction and Synthesis (ICCV 2021)
https://arxiv.org/pdf/2012.12821.pdf

This loss adaptively focuses on hard-to-synthesize frequency components,
perfect for DVD→Blu-ray restoration where we need to fix color first,
then add details.
"""

import torch
import torch.nn as nn
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss - adaptively weights frequency components.
    
    Unlike standard FFT loss, this dynamically adjusts weights based on
    reconstruction difficulty, focusing more on frequencies that are hard
    to synthesize.
    
    Args:
        loss_weight (float): Overall weight for the loss. Default: 1.0
        alpha (float): Focusing factor. Higher = more focus on hard frequencies. Default: 1.0
        patch_factor (int): Divide image into patches for local frequency analysis. Default: 1
        ave_spectrum (bool): Use minibatch average spectrum. Default: False
        log_matrix (bool): Apply log to spectrum weights for stability. Default: False
        batch_matrix (bool): Calculate weights using batch statistics. Default: False
    """
    
    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1,
                 ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
    
    def tensor2freq(self, x):
        """Convert tensor to frequency domain with optional patching."""
        # Handle 5D video tensors
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            
        batch, channels, height, width = x.shape
        
        # Patch-based processing
        if self.patch_factor > 1:
            # Ensure divisibility
            assert height % self.patch_factor == 0 and width % self.patch_factor == 0, \
                f'Image size ({height}, {width}) must be divisible by patch_factor {self.patch_factor}'
            
            patch_h = height // self.patch_factor
            patch_w = width // self.patch_factor
            
            # Extract patches
            patches = []
            for i in range(self.patch_factor):
                for j in range(self.patch_factor):
                    patch = x[:, :, 
                             i * patch_h:(i + 1) * patch_h,
                             j * patch_w:(j + 1) * patch_w]
                    patches.append(patch)
            
            # Stack patches: (B, num_patches, C, H_patch, W_patch)
            x = torch.stack(patches, dim=1)
        else:
            # Add patch dimension for consistency
            x = x.unsqueeze(1)
        
        # Apply 2D FFT (real-to-complex)
        freq = torch.fft.fft2(x, norm='ortho')
        
        # Convert complex to real representation: [..., 2] for (real, imag)
        freq = torch.stack([freq.real, freq.imag], dim=-1)
        
        return freq
    
    def loss_formulation(self, pred_freq, target_freq, matrix=None):
        """Calculate the loss with adaptive weighting.
        
        This follows the original implementation's structure exactly.
        """
        # Calculate or use provided weight matrix
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            # Calculate frequency-wise error magnitude
            freq_error = (pred_freq - target_freq) ** 2
            freq_error_magnitude = torch.sqrt(freq_error[..., 0] + freq_error[..., 1])
            
            # Apply focusing factor (alpha)
            weight_matrix = freq_error_magnitude ** self.alpha
            
            # Optional: Apply log for numerical stability
            if self.log_matrix:
                weight_matrix = torch.log(weight_matrix + 1.0)
            
            # Normalize weights
            if self.batch_matrix:
                # Normalize across entire batch
                weight_matrix = weight_matrix / weight_matrix.max()
            else:
                # Normalize per sample - CRITICAL FIX HERE
                # Original uses hierarchical max: max over last two dims (H, W)
                # Shape is (batch, patches, channels, height, width)
                # We need to keep batch, patches, channels and add None for H, W
                weight_matrix = weight_matrix / weight_matrix.max(-1).values.max(-1).values[:, :, :, None, None]
            
            # Handle NaN and clamp
            weight_matrix[torch.isnan(weight_matrix)] = 0.0
            weight_matrix = torch.clamp(weight_matrix, min=0.0, max=1.0)
            weight_matrix = weight_matrix.clone().detach()
        
        # Assertion check for weight validity
        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            f'but got Min: {weight_matrix.min().item():.10f} Max: {weight_matrix.max().item():.10f}')
        
        # Calculate frequency distance (squared Euclidean)
        freq_distance = (pred_freq - target_freq) ** 2
        freq_distance = freq_distance[..., 0] + freq_distance[..., 1]
        
        # Apply adaptive weights (Hadamard product)
        weighted_loss = weight_matrix * freq_distance
        
        return torch.mean(weighted_loss)
    
    def forward(self, pred, target, matrix=None):
        """Calculate focal frequency loss.
        
        Args:
            pred: Predicted tensor (B, C, H, W) or (B, T, C, H, W)
            target: Target tensor, same shape as pred
            matrix: Pre-computed weight matrix (optional)
        
        Returns:
            Focal frequency loss value
        """
        # Convert to frequency domain
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        
        # Optional: Use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = pred_freq.mean(dim=0, keepdim=True)
            target_freq = target_freq.mean(dim=0, keepdim=True)
        
        # Calculate loss using original formulation
        loss = self.loss_formulation(pred_freq, target_freq, matrix)
        
        return loss * self.loss_weight