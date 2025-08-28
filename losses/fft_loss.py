"""FFT Loss for preserving high-frequency details in video super-resolution.

Recommended by Gemini Pro for stability and sharpness preservation.
"""

import torch
import torch.nn as nn
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class FFTLoss(nn.Module):
    """Frequency domain loss using Fast Fourier Transform.
    
    This loss encourages the model to preserve high-frequency details
    like edges, textures, and film grain without the instability of
    perceptual losses.
    
    Args:
        loss_weight: Weight for the loss
        reduction: Reduction method ('mean' or 'sum')
        highpass_cutoff: If > 0, applies high-pass filter keeping only frequencies 
                        above this fraction (0-1) of the spectrum. 
                        E.g., 0.3 keeps only frequencies > 30% of max frequency.
                        Set to 0 to disable (use full spectrum).
        highpass_type: Type of high-pass filter:
                      'hard': Binary mask (sharp cutoff)
                      'gaussian': Smooth Gaussian transition
                      'linear': Linear ramp transition
    """
    
    def __init__(self, loss_weight=1.0, reduction='mean', 
                 highpass_cutoff=0.0, highpass_type='gaussian'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.highpass_cutoff = highpass_cutoff
        self.highpass_type = highpass_type
        self.is_lowpass = False  # Can be set externally to invert the mask
    
    def create_highpass_mask(self, shape, device):
        """Create high-pass filter mask for frequency domain.
        
        Args:
            shape: Shape of the FFT output (H, W)
            device: Device to create the mask on
        
        Returns:
            High-pass filter mask
        """
        h, w = shape[-2:]
        
        # Create coordinate grids for distance calculation
        # For rfft2, the width is reduced
        cy, cx = h // 2, 0  # Center for rfft2 is at (h//2, 0)
        
        # Create meshgrid
        y = torch.arange(h, device=device).float()
        x = torch.arange(w, device=device).float()
        
        # Shift y coordinates to center
        y = torch.minimum(y, h - y)
        
        # Create 2D grid
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Calculate normalized distance from DC component (low frequencies)
        # Normalize by the maximum possible distance
        max_dist = (h // 2)
        dist = torch.sqrt((yy / h) ** 2 + (xx / w) ** 2)
        dist = dist / dist.max()
        
        if self.highpass_type == 'hard':
            # Binary mask
            mask = (dist > self.highpass_cutoff).float()
        elif self.highpass_type == 'gaussian':
            # Smooth Gaussian transition
            # Sigmoid function for smooth transition
            steepness = 10.0  # Controls transition sharpness
            mask = torch.sigmoid(steepness * (dist - self.highpass_cutoff))
        elif self.highpass_type == 'linear':
            # Linear ramp
            transition_width = 0.1
            mask = torch.clamp((dist - self.highpass_cutoff) / transition_width, 0, 1)
        else:
            mask = torch.ones_like(dist)
        
        return mask
    
    def forward(self, pred, target):
        """Calculate FFT loss.
        
        Args:
            pred: Predicted tensor (B, C, H, W) or (B, T, C, H, W) for video
            target: Ground truth tensor, same shape as pred
        
        Returns:
            FFT loss value
        """
        # Handle video tensors (B, T, C, H, W) by reshaping to (B*T, C, H, W)
        if pred.dim() == 5:
            b, t, c, h, w = pred.shape
            pred = pred.view(b * t, c, h, w)
            target = target.view(b * t, c, h, w)
        
        # Convert to float32 for FFT computation (required for non-power-of-2 sizes in mixed precision)
        # Store original dtype to convert back
        orig_dtype = pred.dtype
        pred_float = pred.float()
        target_float = target.float()
        
        # Apply 2D FFT
        pred_fft = torch.fft.rfft2(pred_float, norm='ortho')
        target_fft = torch.fft.rfft2(target_float, norm='ortho')
        
        # Create high-pass mask if needed
        if self.highpass_cutoff > 0:
            mask = self.create_highpass_mask(pred_fft.shape, pred_fft.device)
            
            # Invert mask for low-pass filtering if requested
            if self.is_lowpass:
                mask = 1.0 - mask
            
            # Expand mask to match batch and channel dimensions
            while mask.dim() < pred_fft.dim():
                mask = mask.unsqueeze(0)
        else:
            mask = 1.0
        
        # Calculate loss on both magnitude and phase
        # Magnitude captures frequency content
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        mag_loss = torch.abs(pred_mag - target_mag) * mask
        
        # Phase captures structure
        # Use real and imaginary parts for more stable gradients
        real_loss = torch.abs(pred_fft.real - target_fft.real) * mask
        imag_loss = torch.abs(pred_fft.imag - target_fft.imag) * mask
        
        # Combine losses
        loss = mag_loss + 0.5 * (real_loss + imag_loss)
        
        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        # Convert back to original dtype if needed
        loss = loss.type(orig_dtype)
        
        return loss * self.loss_weight