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
    """
    
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    
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
        
        # Calculate loss on both magnitude and phase
        # Magnitude captures frequency content
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        mag_loss = torch.abs(pred_mag - target_mag)
        
        # Phase captures structure
        # Use real and imaginary parts for more stable gradients
        real_loss = torch.abs(pred_fft.real - target_fft.real)
        imag_loss = torch.abs(pred_fft.imag - target_fft.imag)
        
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