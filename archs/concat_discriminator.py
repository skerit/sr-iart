"""Concatenated Discriminator for VHS→Bluray Super-Resolution Quality Assessment"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ConcatDiscriminator(nn.Module):
    """Discriminator that compares generated and ground truth images directly.
    
    Takes concatenated (generated, GT) pairs and outputs quality score.
    Lower scores = better match (more similar to GT quality).
    """
    
    def __init__(self, in_channels=6, base_channels=128, num_layers=5):
        super().__init__()
        
        # High-resolution branch (minimal downsampling for fine detail)
        # Only downsample once to 64x64 to preserve sharpness information
        self.high_res_branch = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels, base_channels, 3, stride=1, padding=1, bias=True)),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels, base_channels//2, 3, stride=1, padding=1, bias=True)),
            nn.BatchNorm2d(base_channels//2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Main branch (original architecture)
        layers = []
        current_channels = in_channels
        out_channels = base_channels
        
        for i in range(num_layers):
            # Only downsample in first 2 layers to preserve detail (4x total downsampling)
            # This preserves high-frequency information needed for sharpness detection
            stride = 2 if i < 2 else 1
            
            # Add bias back for better expressiveness
            # Apply spectral normalization for training stability
            layers.append(
                spectral_norm(nn.Conv2d(current_channels, out_channels, 4, stride, 1, bias=True))
            )
            
            # Skip BatchNorm on first layer (following standard practice)
            if i > 0:
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            current_channels = out_channels
            out_channels = min(out_channels * 2, 512)  # Cap at 512 channels
        
        self.features = nn.Sequential(*layers)
        
        # Combine high-res and main branch outputs
        # High-res branch outputs base_channels//2 channels at 64x64
        # Main branch outputs 512 channels at 32x32
        # We'll process each separately then combine scores
        
        # Final layers for each branch
        self.high_res_final = spectral_norm(nn.Conv2d(base_channels//2, 1, 3, 1, 1))
        self.main_final = spectral_norm(nn.Conv2d(current_channels, 1, 3, 1, 1))
        
        # Initialize weights properly
        self._initialize_weights()
        
    def forward(self, generated, ground_truth):
        """Compare generated image with ground truth.
        
        Args:
            generated: Generated/upscaled image [B, 3, H, W]
            ground_truth: Ground truth image [B, 3, H, W]
            
        Returns:
            Quality score [B, 1] - 0 = perfect match, 1 = very different
        """
        # Concatenate along channel dimension
        combined = torch.cat([generated, ground_truth], dim=1)  # [B, 6, H, W]
        
        # Process through both branches
        # High-res branch: 128x128 -> 64x64, focuses on fine details
        high_res_features = self.high_res_branch(combined)
        high_res_score_map = self.high_res_final(high_res_features)  # [B, 1, 64, 64]
        high_res_score = high_res_score_map.mean(dim=[2, 3])  # [B, 1]
        
        # Main branch: 128x128 -> 32x32, captures global patterns
        main_features = self.features(combined)
        main_score_map = self.main_final(main_features)  # [B, 1, 32, 32]
        main_score = main_score_map.mean(dim=[2, 3])  # [B, 1]
        
        # Combine scores with weighted average
        # High-res branch is weighted more (0.6) for sharpness detection
        # Main branch (0.4) for overall quality
        score = 0.6 * high_res_score + 0.4 * main_score
        
        # Apply sigmoid to convert logits to 0-1 probability
        # 0 = perfect match, 1 = very different
        score = torch.sigmoid(score)
        
        return score
    
    def forward_with_sigmoid(self, generated, ground_truth):
        """Forward pass with sigmoid for inference/testing.
        
        Note: Sigmoid is already applied in forward(), so this just calls forward().
        Kept for backward compatibility.
        """
        return self.forward(generated, ground_truth)
    
    def _initialize_weights(self):
        """Proper weight initialization for better training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use Kaiming initialization which works better with ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for final layers to break symmetry
        # Initialize to output different values for different inputs
        nn.init.normal_(self.high_res_final.weight, mean=0, std=0.1)
        nn.init.constant_(self.high_res_final.bias, 0.0)  # Start neutral
        
        nn.init.normal_(self.main_final.weight, mean=0, std=0.1)
        nn.init.constant_(self.main_final.bias, 0.0)  # Start neutral
    
    def get_feature_maps(self, generated, ground_truth):
        """Get intermediate feature maps for loss calculation.
        
        Useful for feature matching loss (like perceptual loss).
        """
        combined = torch.cat([generated, ground_truth], dim=1)
        
        feature_maps = []
        x = combined
        for module in self.features:
            x = module(x)
            if isinstance(module, nn.Conv2d):
                feature_maps.append(x)
        
        return feature_maps


class ConcatDiscriminatorLoss(nn.Module):
    """Loss function wrapper for using pre-trained discriminator."""
    
    def __init__(self, discriminator_path=None, loss_weight=1.0, 
                 target_score=0.0, device='cuda'):
        super().__init__()
        
        self.loss_weight = loss_weight
        self.target_score = target_score
        self.device = device
        
        # Initialize discriminator
        self.discriminator = ConcatDiscriminator().to(device)
        
        # Load pre-trained weights if provided
        if discriminator_path:
            checkpoint = torch.load(discriminator_path, map_location=device)
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            print(f"Loaded discriminator from {discriminator_path}")
        
        # Freeze discriminator (we're using it as a loss, not training it)
        self.discriminator.eval()
        for param in self.discriminator.parameters():
            param.requires_grad = False
    
    def forward(self, generated, ground_truth):
        """Calculate discriminator-based loss.
        
        Args:
            generated: Model output [B, 3, H, W]
            ground_truth: Target image [B, 3, H, W]
            
        Returns:
            Loss value (scalar)
        """
        with torch.no_grad():
            # Get scores from discriminator (already in 0-1 range with sigmoid)
            predicted_scores = self.discriminator(generated, ground_truth)
        
        # Use MSE loss for direct regression to target score
        # Generator should produce outputs that discriminator scores as 0.0 (perfect)
        target = torch.full_like(predicted_scores, self.target_score)
        loss = F.mse_loss(predicted_scores, target)
        
        return self.loss_weight * loss