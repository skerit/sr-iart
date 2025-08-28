"""
Feature Distance Loss (FDL) for IART
Robust to misalignments by comparing features in frequency domain
Based on: https://github.com/eezkni/FDL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv
from basicsr.utils.registry import LOSS_REGISTRY


class VGGFeatureExtractor(nn.Module):
    """VGG19 feature extractor for FDL loss."""
    
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        
        vgg_pretrained_features = tv.vgg19(pretrained=pretrained).features
        
        # Create stages for different feature levels
        self.stage1 = nn.Sequential()
        self.stage2 = nn.Sequential()
        self.stage3 = nn.Sequential()
        self.stage4 = nn.Sequential()
        self.stage5 = nn.Sequential()
        
        # VGG19 layer divisions
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        
        # Channel dimensions for each stage
        self.chns = [64, 128, 256, 512, 512]
    
    def forward(self, x):
        # Normalize input
        h = (x - self.mean) / self.std
        
        # Extract features at different levels
        h = self.stage1(h)
        h_relu1_2 = h
        
        h = self.stage2(h)
        h_relu2_2 = h
        
        h = self.stage3(h)
        h_relu3_3 = h
        
        h = self.stage4(h)
        h_relu4_3 = h
        
        h = self.stage5(h)
        h_relu5_3 = h
        
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]


class ResNetFeatureExtractor(nn.Module):
    """ResNet101 feature extractor for FDL loss."""
    
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        
        model = tv.resnet101(pretrained=pretrained)
        model.eval()
        
        self.stage1 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu
        )
        self.stage2 = nn.Sequential(
            model.maxpool,
            model.layer1,
        )
        self.stage3 = nn.Sequential(
            model.layer2,
        )
        self.stage4 = nn.Sequential(
            model.layer3,
        )
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        
        # Channel dimensions for each stage
        self.chns = [64, 256, 512, 1024]
    
    def forward(self, x):
        # Normalize input
        h = (x - self.mean) / self.std
        
        # Extract features at different levels
        h = self.stage1(h)
        h_stage1 = h
        
        h = self.stage2(h)
        h_stage2 = h
        
        h = self.stage3(h)
        h_stage3 = h
        
        h = self.stage4(h)
        h_stage4 = h
        
        return [h_stage1, h_stage2, h_stage3, h_stage4]


@LOSS_REGISTRY.register()
class FDLLoss(nn.Module):
    """Feature Distance Loss - robust to misalignment through frequency domain comparison.
    
    Args:
        loss_weight (float): Weight of the loss. Default: 1.0
        model (str): Feature extractor model ('VGG' or 'ResNet'). Default: 'VGG'
        patch_size (int): Size of patches for SWD. Default: 5
        stride (int): Stride for patch extraction. Default: 1
        num_proj (int): Number of projections for SWD. Default: 256
        phase_weight (float): Weight for phase component. Default: 1.0
        reduction (str): Reduction method ('mean' or 'sum'). Default: 'mean'
    """
    
    def __init__(self,
                 loss_weight=1.0,
                 model='VGG',
                 patch_size=5,
                 stride=1,
                 num_proj=256,
                 phase_weight=1.0,
                 reduction='mean'):
        super().__init__()
        
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.phase_weight = phase_weight
        self.stride = stride
        
        # Initialize feature extractor
        if model == 'VGG':
            self.feature_extractor = VGGFeatureExtractor()
        elif model == 'ResNet':
            self.feature_extractor = ResNetFeatureExtractor()
        else:
            raise ValueError(f"Unsupported model: {model}. Choose 'VGG' or 'ResNet'")
        
        # Pre-generate random projections for each layer
        for i in range(len(self.feature_extractor.chns)):
            rand = torch.randn(num_proj, self.feature_extractor.chns[i], patch_size, patch_size)
            # Normalize projections
            rand = rand / rand.view(rand.shape[0], -1).norm(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            self.register_buffer(f"rand_{i}", rand)
    
    def compute_swd(self, x, y, idx):
        """Compute Sliced Wasserstein Distance between feature maps.
        
        Args:
            x, y: Feature tensors of shape (N, C, H, W)
            idx: Layer index for selecting appropriate random projections
        """
        rand = getattr(self, f"rand_{idx}")
        
        # Project features using random filters
        proj_x = F.conv2d(x, rand, stride=self.stride)
        proj_x = proj_x.reshape(proj_x.shape[0], proj_x.shape[1], -1)
        
        proj_y = F.conv2d(y, rand, stride=self.stride)
        proj_y = proj_y.reshape(proj_y.shape[0], proj_y.shape[1], -1)
        
        # Sort projections (key for Wasserstein distance)
        proj_x, _ = torch.sort(proj_x, dim=-1)
        proj_y, _ = torch.sort(proj_y, dim=-1)
        
        # Compute mean absolute difference
        distance = torch.abs(proj_x - proj_y).mean([1, 2])
        
        return distance
    
    def forward(self, pred, target):
        """Calculate FDL loss.
        
        Args:
            pred: Predicted tensor (B, C, H, W) or (B, T, C, H, W) for video
            target: Target tensor, same shape as pred
        
        Returns:
            FDL loss value
        """
        # Handle video tensors
        if pred.dim() == 5:  # (B, T, C, H, W)
            b, t, c, h, w = pred.shape
            pred = pred.view(b * t, c, h, w)
            target = target.view(b * t, c, h, w)
        
        # Extract features
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        loss = 0.0
        
        # Process each feature level
        for i, (feat_pred, feat_target) in enumerate(zip(pred_features, target_features)):
            # Transform to frequency domain
            fft_pred = torch.fft.fftn(feat_pred, dim=(-2, -1))
            fft_target = torch.fft.fftn(feat_target, dim=(-2, -1))
            
            # Separate amplitude and phase
            pred_amp = torch.abs(fft_pred)
            pred_phase = torch.angle(fft_pred)
            target_amp = torch.abs(fft_target)
            target_phase = torch.angle(fft_target)
            
            # Compute SWD for amplitude and phase
            amp_distance = self.compute_swd(pred_amp, target_amp, i)
            phase_distance = self.compute_swd(pred_phase, target_phase, i)
            
            # Combine amplitude and phase distances
            layer_loss = amp_distance + self.phase_weight * phase_distance
            
            if self.reduction == 'mean':
                layer_loss = layer_loss.mean()
            elif self.reduction == 'sum':
                layer_loss = layer_loss.sum()
            
            loss += layer_loss
        
        return loss * self.loss_weight