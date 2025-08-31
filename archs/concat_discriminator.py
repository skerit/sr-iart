"""Concatenated Discriminator for VHS→Bluray Super-Resolution Quality Assessment"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatDiscriminator(nn.Module):
    """Discriminator that compares generated and ground truth images directly.
    
    Takes concatenated (generated, GT) pairs and outputs quality score.
    Lower scores = better match (more similar to GT quality).
    """
    
    def __init__(self, in_channels=6, base_channels=192, num_layers=7):
        super().__init__()
        
        layers = []
        current_channels = in_channels
        out_channels = base_channels
        
        for i in range(num_layers):
            # Use stride 2 for downsampling (except last layer)
            stride = 2 if i < num_layers - 1 else 1
            
            layers.append(
                nn.Conv2d(current_channels, out_channels, 4, stride, 1, bias=False)
            )
            
            # Skip BatchNorm on first layer (following standard practice)
            if i > 0:
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            current_channels = out_channels
            out_channels = min(out_channels * 2, 512)  # Cap at 512 channels
        
        self.features = nn.Sequential(*layers)
        
        # Final layer outputs a single channel (quality map)
        # Use 3x3 kernel with padding 1 for better compatibility with various depths
        self.final = nn.Conv2d(current_channels, 1, 3, 1, 1)
        
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
        
        # Extract features
        features = self.features(combined)
        
        # Get quality map
        quality_map = self.final(features)  # [B, 1, H', W']
        
        # Global average pooling to get single score per image
        score = torch.sigmoid(quality_map.mean(dim=[2, 3]))  # [B, 1]
        
        return score
    
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
            score = self.discriminator(generated, ground_truth)
        
        # We want the score to be close to target_score (usually 0)
        loss = F.mse_loss(score, torch.full_like(score, self.target_score))
        
        return self.loss_weight * loss