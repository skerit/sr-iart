"""
Feature Matching Loss for GAN training
Based on CIPLAB NTIRE 2020 implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.losses.loss_util import weighted_loss
from basicsr.utils.registry import LOSS_REGISTRY


def huber_loss(input, target, delta=0.01):
    """Huber loss as used in CIPLAB
    
    Args:
        input: Input tensor
        target: Target tensor  
        delta: Threshold for quadratic/linear transition
    """
    abs_error = torch.abs(input - target)
    quadratic = torch.clamp(abs_error, max=delta)
    
    # Same as tf.maximum(abs_error - delta, 0)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic.pow(2) / delta + linear
    
    return loss.mean()


@LOSS_REGISTRY.register()
class FeatureMatchingLoss(nn.Module):
    """Feature Matching Loss for discriminator intermediate features
    
    Compares intermediate discriminator features between real and fake images.
    Used in CIPLAB NTIRE 2020 winning solution.
    
    Args:
        loss_weight: Weight for the loss. Default: 1.0
        loss_type: Type of loss to use ('l1', 'l2', 'huber'). Default: 'huber'
        delta: Delta parameter for Huber loss. Default: 0.01
    """
    
    def __init__(self, loss_weight=1.0, loss_type='huber', delta=0.01):
        super(FeatureMatchingLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.delta = delta
        
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'huber':
            self.loss_fn = lambda x, y: huber_loss(x, y, delta)
        else:
            raise ValueError(f'Unsupported loss type: {loss_type}')
    
    def forward(self, fake_features, real_features):
        """Calculate feature matching loss
        
        Args:
            fake_features: Tuple of (encoder_features, decoder_features) from fake images
                          Each is a list of feature tensors
            real_features: Tuple of (encoder_features, decoder_features) from real images
                          Each is a list of feature tensors
                          
        Returns:
            Weighted feature matching loss
        """
        fake_enc, fake_dec = fake_features
        real_enc, real_dec = real_features
        
        losses = []
        
        # Compare encoder features
        for fake_feat, real_feat in zip(fake_enc, real_enc):
            # Detach real features to avoid backprop through discriminator
            losses.append(self.loss_fn(fake_feat, real_feat.detach()))
        
        # Compare decoder features  
        for fake_feat, real_feat in zip(fake_dec, real_dec):
            # Detach real features to avoid backprop through discriminator
            losses.append(self.loss_fn(fake_feat, real_feat.detach()))
        
        # Average all feature losses
        loss = torch.stack(losses).mean()
        
        return loss * self.loss_weight


@LOSS_REGISTRY.register()
class CIPLABFeatureMatchingLoss(nn.Module):
    """CIPLAB-specific Feature Matching Loss implementation
    
    Exactly matches the CIPLAB NTIRE 2020 implementation.
    Uses Huber loss with delta=0.01 for all feature comparisons.
    
    Args:
        loss_weight: Weight for the loss. Default: 1.0 (as in CIPLAB)
    """
    
    def __init__(self, loss_weight=1.0):
        super(CIPLABFeatureMatchingLoss, self).__init__()
        self.loss_weight = loss_weight
    
    def forward(self, fake_e_feats, fake_d_feats, real_e_feats, real_d_feats):
        """Calculate CIPLAB-style feature matching loss
        
        Args:
            fake_e_feats: List of encoder features from fake images
            fake_d_feats: List of decoder features from fake images
            real_e_feats: List of encoder features from real images  
            real_d_feats: List of decoder features from real images
                          
        Returns:
            Weighted feature matching loss
        """
        loss_FMs = []
        
        # Match each layer's features using Huber loss
        for f in range(len(fake_e_feats)):
            loss_FMs.append(huber_loss(fake_e_feats[f], real_e_feats[f].detach(), delta=0.01))
            loss_FMs.append(huber_loss(fake_d_feats[f], real_d_feats[f].detach(), delta=0.01))
        
        # Average all feature losses and apply weight
        loss_FM = torch.mean(torch.stack(loss_FMs)) * self.loss_weight
        
        return loss_FM