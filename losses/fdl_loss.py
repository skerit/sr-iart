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
    
    def __init__(self, requires_grad=False, pretrained=True, use_input_norm=True, range_norm=False):
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
        
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        
        if self.use_input_norm:
            # ImageNet normalization - for images in [0, 1] range
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        
        # Channel dimensions for each stage
        self.chns = [64, 128, 256, 512, 512]
    
    @autocast(enabled=False)  # Force float32 for pretrained network stability
    def forward(self, x):
        # Ensure float32 precision
        x = x.float()
        
        # Optional: Convert from [-1, 1] to [0, 1] if needed
        if self.range_norm:
            x = (x + 1) / 2
        
        # Apply ImageNet normalization if enabled
        if self.use_input_norm:
            # This expects x to be roughly in [0, 1] range
            # But doesn't enforce it with clamping
            x = (x - self.mean) / self.std
        
        # Extract features at different levels
        h = self.stage1(x)
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
    """ResNet101 feature extractor for FDL loss - Fixed version."""
    
    def __init__(self, requires_grad=False, pretrained=True, use_input_norm=True, range_norm=False):
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
        
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        
        if self.use_input_norm:
            # ImageNet normalization
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        
        # Channel dimensions for each stage
        self.chns = [64, 256, 512, 1024]
    
    @autocast(enabled=False)  # Force float32 for pretrained network stability
    def forward(self, x):
        # Ensure float32 precision
        x = x.float()
        
        # Optional: Convert from [-1, 1] to [0, 1] if needed
        if self.range_norm:
            x = (x + 1) / 2
        
        # Apply ImageNet normalization if enabled
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        
        # Extract features at different levels
        h = self.stage1(x)
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
    """Feature Distance Loss.
    
    Args:
        loss_weight (float): Weight of the loss. Default: 1.0
        model (str): Feature extractor model ('VGG' or 'ResNet'). Default: 'VGG'
        patch_size (int): Size of patches for SWD. Default: 5
        stride (int): Stride for patch extraction. Default: 1
        num_proj (int): Number of projections for SWD. Default: 256
        phase_weight (float): Weight for phase component. Default: 1.0
        reduction (str): Reduction method ('mean' or 'sum'). Default: 'mean'
        use_input_norm (bool): Apply ImageNet normalization. Default: True
        range_norm (bool): Convert from [-1,1] to [0,1]. Default: False
        skip_nan_layers (bool): Skip layers with NaN loss. Default: True
        skip_layers (list): List of layer indices to always skip. Default: None
                            Example: [0, 1] to skip first two layers
    """
    
    def __init__(self,
                 loss_weight=1.0,
                 model='VGG',
                 patch_size=5,
                 stride=1,
                 num_proj=256,
                 phase_weight=1.0,
                 reduction='mean',
                 chunk_size=16,
                 scale_factor=1.0,
                 frame_chunk_size=4,
                 use_input_norm=True,
                 range_norm=False,
                 skip_nan_layers=True,
                 skip_layers=None):
        super().__init__()
        
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.phase_weight = phase_weight
        self.stride = stride
        self.num_proj = num_proj
        self.chunk_size = min(chunk_size, num_proj)
        self.scale_factor = scale_factor
        self.frame_chunk_size = frame_chunk_size
        self.skip_nan_layers = skip_nan_layers
        self.skip_layers = skip_layers if skip_layers is not None else []
        
        # Initialize feature extractor with proper normalization settings
        if model == 'VGG':
            self.feature_extractor = VGGFeatureExtractor(
                use_input_norm=use_input_norm,
                range_norm=range_norm
            )
        elif model == 'ResNet':
            self.feature_extractor = ResNetFeatureExtractor(
                use_input_norm=use_input_norm,
                range_norm=range_norm
            )
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
        # Handle video tensors - process in smaller chunks to save memory
        if pred.dim() == 5:  # (B, T, C, H, W)
            b, t, c, h, w = pred.shape
            total_loss = 0.0
            num_chunks = 0
            
            # Process frames in chunks and compute loss per chunk
            for i in range(0, t, self.frame_chunk_size):
                end_idx = min(i + self.frame_chunk_size, t)
                pred_chunk = pred[:, i:end_idx].reshape(-1, c, h, w)
                target_chunk = target[:, i:end_idx].reshape(-1, c, h, w)
                
                # Compute FDL loss for this chunk
                chunk_loss = self._compute_fdl_loss(pred_chunk, target_chunk)
                total_loss += chunk_loss
                num_chunks += 1
            
            # Average loss across chunks
            return (total_loss / num_chunks) * self.loss_weight * self.scale_factor
        else:
            # Extract features normally for images
            loss = self._compute_fdl_loss(pred, target)
            return loss * self.loss_weight * self.scale_factor
    
    @autocast(enabled=False)  # Force float32 for numerical stability in FFT/SWD
    def _compute_fdl_loss(self, pred, target):
        """Compute FDL loss for a batch of frames/images."""
        # Ensure inputs are float32 for numerical stability
        pred = pred.float()
        target = target.float()
        
        # Extract features
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        loss = 0.0
        has_valid_layer = False
        skipped_layers = []
        
        # Process each feature level
        for i, (feat_pred, feat_target) in enumerate(zip(pred_features, target_features)):
            # Skip layers based on configuration
            if i in self.skip_layers:
                skipped_layers.append(i)
                continue
            
            # Check for degenerate features (all zeros or very small)
            if feat_pred.abs().mean() < 1e-8 or feat_target.abs().mean() < 1e-8:
                if self.skip_nan_layers:
                    skipped_layers.append(i)
                    continue
            
            # Transform to frequency domain with stability check
            try:
                fft_pred = torch.fft.fftn(feat_pred, dim=(-2, -1))
                fft_target = torch.fft.fftn(feat_target, dim=(-2, -1))
                
                # Add small epsilon to prevent numerical issues in angle computation
                eps = 1e-10
                fft_pred_stable = fft_pred + eps
                fft_target_stable = fft_target + eps
                
                # Separate amplitude and phase
                pred_amp = torch.abs(fft_pred_stable)
                pred_phase = torch.angle(fft_pred_stable)
                target_amp = torch.abs(fft_target_stable)
                target_phase = torch.angle(fft_target_stable)
                
                # Compute SWD for amplitude and phase
                amp_distance = self.compute_swd(pred_amp, target_amp, i)
                phase_distance = self.compute_swd(pred_phase, target_phase, i)
                
                # Combine amplitude and phase distances
                layer_loss = amp_distance + self.phase_weight * phase_distance
                
                if self.reduction == 'mean':
                    layer_loss = layer_loss.mean()
                elif self.reduction == 'sum':
                    layer_loss = layer_loss.sum()
                
                # Check for NaN after computation
                if torch.isnan(layer_loss).any() or torch.isinf(layer_loss).any():
                    if self.skip_nan_layers:
                        skipped_layers.append(i)
                        continue
                    else:
                        # Fallback to zero contribution from this layer
                        layer_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                
                loss += layer_loss
                has_valid_layer = True
                
            except Exception as e:
                # Handle any FFT or computation errors
                if self.skip_nan_layers:
                    skipped_layers.append(i)
                    continue
                else:
                    raise
        
        # Log skipped layers if any (differentiate between configured and automatic skips)
        if skipped_layers:
            configured_skips = [i for i in skipped_layers if i in self.skip_layers]
            auto_skips = [i for i in skipped_layers if i not in self.skip_layers]
            
            if auto_skips and configured_skips:
                print(f"FDL: Skipped layers {configured_skips} (configured) and {auto_skips} (numerical issues)")
            elif auto_skips:
                print(f"FDL: Skipped layers {auto_skips} due to numerical issues")
            # Don't log configured skips unless there are also auto skips
        
        # Check if any valid layers were processed
        if not has_valid_layer:
            print("Warning: All FDL layers had issues, returning small loss")
            # Return small non-zero loss to maintain gradient flow
            return torch.tensor(0.001, device=pred.device, dtype=pred.dtype, requires_grad=True)
        
        # Return raw loss (weight and scale factor applied in forward())
        return loss