"""Patch-based discriminator for sharpness detection without aggressive downsampling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class PatchSharpnessDiscriminator(nn.Module):
    """Discriminator that evaluates sharpness on local patches without excessive downsampling.
    
    Key improvements:
    - Minimal downsampling to preserve high-frequency information
    - Returns spatial quality map for localized feedback
    - Can be used patch-wise or with spatial pooling
    """
    
    def __init__(self, in_channels=6, base_channels=64, num_layers=4):
        super().__init__()
        
        layers = []
        current_channels = in_channels
        out_channels = base_channels
        
        for i in range(num_layers):
            # Only downsample in first 2 layers to preserve detail
            # This gives us 4x downsampling total instead of 16x
            if i < 2:
                stride = 2
                kernel_size = 4
                padding = 1
            else:
                # No more downsampling - use stride 1 to preserve resolution
                stride = 1
                kernel_size = 3
                padding = 1
            
            # Apply spectral normalization for training stability
            layers.append(
                spectral_norm(nn.Conv2d(current_channels, out_channels, kernel_size, stride, padding, bias=True))
            )
            
            # BatchNorm except first layer
            if i > 0:
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            current_channels = out_channels
            # Increase channels but not as aggressively
            out_channels = min(out_channels * 2, 256)
        
        self.features = nn.Sequential(*layers)
        
        # Final 1x1 conv for per-pixel quality score
        # Apply spectral normalization to final layer as well
        self.final = spectral_norm(nn.Conv2d(current_channels, 1, kernel_size=1, stride=1, padding=0))
        
        self._initialize_weights()
    
    def forward(self, generated, ground_truth, return_spatial=False):
        """Compare generated and GT images.
        
        Args:
            generated: Generated/upscaled image [B, 3, H, W]
            ground_truth: Ground truth image [B, 3, H, W]
            return_spatial: If True, return spatial quality map. If False, return averaged score.
            
        Returns:
            If return_spatial:
                Quality map [B, 1, H//4, W//4] - spatial quality scores
            Else:
                Quality score [B, 1] - averaged quality score
        """
        # Concatenate inputs
        combined = torch.cat([generated, ground_truth], dim=1)  # [B, 6, H, W]
        
        # Extract features (only 4x downsampled, not 16x)
        features = self.features(combined)
        
        # Get quality map
        quality_map = self.final(features)  # [B, 1, H//4, W//4]
        
        if return_spatial:
            # Return spatial map for localized loss
            return quality_map
        else:
            # Global average pooling for single score
            return quality_map.mean(dim=[2, 3])  # [B, 1]
    
    def _initialize_weights(self):
        """Initialize weights for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize final layer for reasonable initial predictions
        nn.init.normal_(self.final.weight, mean=0, std=0.02)
        nn.init.constant_(self.final.bias, 0)


class MultiScalePatchDiscriminator(nn.Module):
    """Multi-scale patch discriminator that evaluates at different resolutions.
    
    Combines judgments from multiple scales to capture both fine details and overall structure.
    """
    
    def __init__(self, base_channels=64):
        super().__init__()
        
        # Three discriminators at different scales
        self.scale1 = PatchSharpnessDiscriminator(6, base_channels, num_layers=4)  # Full resolution
        self.scale2 = PatchSharpnessDiscriminator(6, base_channels, num_layers=3)  # Half resolution
        self.scale3 = PatchSharpnessDiscriminator(6, base_channels, num_layers=2)  # Quarter resolution
        
        # No learnable weights - use simple averaging for robustness
        # This ensures all scales contribute equally
    
    def forward(self, generated, ground_truth):
        """Evaluate at multiple scales.
        
        Args:
            generated: Generated image [B, 3, H, W]
            ground_truth: Ground truth image [B, 3, H, W]
            
        Returns:
            Combined quality score [B, 1]
        """
        scores = []
        
        # Full resolution
        score1 = self.scale1(generated, ground_truth, return_spatial=False)
        scores.append(score1)
        
        # Half resolution
        gen_half = F.interpolate(generated, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt_half = F.interpolate(ground_truth, scale_factor=0.5, mode='bilinear', align_corners=False)
        score2 = self.scale2(gen_half, gt_half, return_spatial=False)
        scores.append(score2)
        
        # Quarter resolution
        gen_quarter = F.interpolate(generated, scale_factor=0.25, mode='bilinear', align_corners=False)
        gt_quarter = F.interpolate(ground_truth, scale_factor=0.25, mode='bilinear', align_corners=False)
        score3 = self.scale3(gen_quarter, gt_quarter, return_spatial=False)
        scores.append(score3)
        
        # Simple averaging is more robust than learnable weights
        # Prevents mode collapse where one scale dominates
        combined = torch.stack(scores, dim=0).mean(dim=0)
        
        return combined


class LocalPatchDiscriminator(nn.Module):
    """Fully convolutional discriminator for efficient patch-based evaluation.
    
    This is the most detail-preserving approach, using a fully convolutional
    architecture to evaluate all patches in a single forward pass.
    """
    
    def __init__(self, receptive_field=33, in_channels=6, base_channels=32):
        super().__init__()
        
        # Fully convolutional network - no downsampling to preserve all details
        layers = []
        current_channels = in_channels
        
        # Build network to achieve desired receptive field
        # Each 3x3 conv adds 2 to receptive field, need ~11 layers for 33x33 RF
        num_layers = (receptive_field - 1) // 2
        
        for i in range(min(num_layers, 11)):  # Cap at 11 layers
            out_channels = base_channels * (2 ** min(i // 3, 2))  # Increase channels every 3 layers
            
            # Apply spectral normalization for training stability
            layers.append(
                spectral_norm(nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True))
            )
            
            if i > 0 and i % 2 == 0:  # BatchNorm every 2 layers
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Final 1x1 conv for per-location quality score
        # Each spatial location in output represents quality of receptive field at that location
        self.final = spectral_norm(nn.Conv2d(current_channels, 1, kernel_size=1))
        
        self.receptive_field = receptive_field
        self._initialize_weights()
    
    def forward(self, generated, ground_truth, return_spatial=False):
        """Evaluate quality across entire image in single forward pass.
        
        Args:
            generated: Generated image [B, 3, H, W]
            ground_truth: Ground truth image [B, 3, H, W]
            return_spatial: If True, return spatial quality map
            
        Returns:
            If return_spatial:
                Quality map [B, 1, H, W] - quality score at each location
            Else:
                Average quality score [B, 1]
        """
        # Concatenate inputs
        combined = torch.cat([generated, ground_truth], dim=1)  # [B, 6, H, W]
        
        # Single forward pass through fully convolutional network
        features = self.features(combined)  # [B, C, H, W]
        quality_map = self.final(features)  # [B, 1, H, W]
        
        if return_spatial:
            # Return full spatial quality map
            return quality_map
        else:
            # Global average pooling for single score
            return quality_map.mean(dim=[2, 3])  # [B, 1]
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)