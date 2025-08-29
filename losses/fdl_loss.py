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
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast


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
    
    @autocast(enabled=False)  # Force float32 for pretrained network stability
    def forward(self, x):
        # Ensure float32 precision
        x = x.float()
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
    
    @autocast(enabled=False)  # Force float32 for pretrained network stability
    def forward(self, x):
        # Ensure float32 precision
        x = x.float()
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
                 num_proj=256,  # Full projections - memory safe with chunking
                 phase_weight=1.0,
                 reduction='mean',
                 chunk_size=16,  # Process projections in chunks
                 scale_factor=1.0):  # Configurable scale factor
        super().__init__()
        
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.phase_weight = phase_weight
        self.stride = stride
        self.num_proj = num_proj
        self.chunk_size = min(chunk_size, num_proj)  # Ensure chunk size doesn't exceed num_proj
        self.scale_factor = scale_factor
        
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
            # Normalize projections with epsilon for numerical stability
            rand_norm = rand.view(rand.shape[0], -1).norm(dim=1, keepdim=True)
            rand = rand.view(rand.shape[0], -1) / (rand_norm + 1e-8)
            rand = rand.view(num_proj, self.feature_extractor.chns[i], patch_size, patch_size)
            self.register_buffer(f"rand_{i}", rand)
    
    def compute_swd(self, x, y, idx):
        """Compute Sliced Wasserstein Distance between feature maps.
        Memory-efficient version that processes projections in chunks.
        
        Args:
            x, y: Feature tensors of shape (N, C, H, W)
            idx: Layer index for selecting appropriate random projections
        """
        rand = getattr(self, f"rand_{idx}")
        
        B = x.shape[0]
        distance = torch.zeros(B, device=x.device)
        
        # Process projections in chunks to reduce memory usage
        for i in range(0, self.num_proj, self.chunk_size):
            end_idx = min(i + self.chunk_size, self.num_proj)
            rand_chunk = rand[i:end_idx]
            
            # Project features using random filters for this chunk
            proj_x = F.conv2d(x, rand_chunk, stride=self.stride)
            proj_x = proj_x.reshape(B, end_idx - i, -1)
            
            proj_y = F.conv2d(y, rand_chunk, stride=self.stride)
            proj_y = proj_y.reshape(B, end_idx - i, -1)
            
            # Sort projections (key for Wasserstein distance)
            # Use in-place operations to save memory
            proj_x = proj_x.sort(dim=-1)[0]
            proj_y = proj_y.sort(dim=-1)[0]
            
            # Compute mean absolute difference for this chunk
            chunk_distance = torch.abs(proj_x - proj_y).mean([1, 2])
            distance += chunk_distance * (end_idx - i) / self.num_proj
            
            # Explicitly delete intermediate tensors to free memory
            del proj_x, proj_y, chunk_distance
        
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
        has_valid_layer = False  # Track if we have at least one valid layer
        
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
            
            # Check for NaN and skip if found
            if torch.isnan(layer_loss).any():
                print(f"Warning: NaN detected in FDL loss at layer {i}, skipping this layer")
                continue
                
            loss += layer_loss
            has_valid_layer = True
        
        # Check if any valid layers were processed
        if not has_valid_layer:
            print("Warning: All FDL layers had NaN, returning 0")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Final NaN check
        if torch.isnan(loss).any():
            print("Warning: NaN in final FDL loss, returning 0")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Apply loss weight and scale factor
        # Note: Following original FDL, we sum across layers (no averaging)
        # Use scale_factor to control magnitude relative to pixel loss
        return loss * self.loss_weight * self.scale_factor