import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
import cv2
import numpy as np
from collections import OrderedDict
from copy import deepcopy
import torchvision
from torch.amp import autocast
from torch.nn.parallel import DataParallel, DistributedDataParallel

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.models.video_recurrent_model import VideoRecurrentModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class RecurrentMixPrecisionRTModel(VideoRecurrentModel):
    """VRT Model adopted in the original VRT. Mix precision is adopted.

    Paper: A Video Restoration Transformer
    """

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        self.current_data = None  # Store current batch data for logging
        
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.net_g.to(self.device)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # Conditionally initialize discriminator if specified
        if opt.get('network_d'):
            self.use_discriminator = True
            self.net_d = build_network(opt['network_d'])
            self.net_d = self.net_d.to(self.device)
            self.print_network(self.net_d)
            
            # Load pretrained discriminator if available
            load_path = self.opt['path'].get('pretrain_network_d', None)
            if load_path:
                self._load_discriminator_network(load_path)
        else:
            self.use_discriminator = False
            
        if self.is_train:
            self.init_training_settings()
            self.fix_flow_iter = opt['train'].get('fix_flow')
            
            # Initialize FFT loss if configured
            train_opt = self.opt['train']
            if train_opt.get('fft_opt'):
                from basicsr.utils import get_root_logger
                logger = get_root_logger()
                
                # Import our custom FFT loss
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from losses.fft_loss import FFTLoss
                
                fft_opt = train_opt['fft_opt']
                self.cri_fft = FFTLoss(
                    loss_weight=fft_opt.get('loss_weight', 1.0),
                    reduction=fft_opt.get('reduction', 'mean'),
                    highpass_cutoff=fft_opt.get('highpass_cutoff', 0.0),
                    highpass_type=fft_opt.get('highpass_type', 'gaussian')
                ).to(self.device)
                logger.info(f"FFT Loss initialized with weight {fft_opt.get('loss_weight', 1.0)}, "
                           f"highpass_cutoff {fft_opt.get('highpass_cutoff', 0.0)}")
            else:
                self.cri_fft = None
            
            # Initialize High-frequency FFT loss if configured (separate from regular FFT)
            if train_opt.get('fft_high_opt'):
                from basicsr.utils import get_root_logger
                logger = get_root_logger()
                
                # Import our custom FFT loss
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from losses.fft_loss import FFTLoss
                
                fft_high_opt = train_opt['fft_high_opt']
                self.cri_fft_high = FFTLoss(
                    loss_weight=fft_high_opt.get('loss_weight', 1.0),
                    reduction=fft_high_opt.get('reduction', 'mean'),
                    highpass_cutoff=fft_high_opt.get('highpass_cutoff', 0.3),  # Default to 30% cutoff
                    highpass_type=fft_high_opt.get('highpass_type', 'gaussian')
                ).to(self.device)
                logger.info(f"High-freq FFT Loss initialized with weight {fft_high_opt.get('loss_weight', 1.0)}, "
                           f"highpass_cutoff {fft_high_opt.get('highpass_cutoff', 0.3)}")
            else:
                self.cri_fft_high = None
            
            # Initialize Low-frequency FFT loss if configured (for color/brightness matching)
            if train_opt.get('fft_low_opt'):
                from basicsr.utils import get_root_logger
                logger = get_root_logger()
                
                # Import our custom FFT loss
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from losses.fft_loss import FFTLoss
                
                fft_low_opt = train_opt['fft_low_opt']
                # For low-pass, we invert the mask logic
                # We'll need to add a lowpass mode to FFTLoss
                self.cri_fft_low = FFTLoss(
                    loss_weight=fft_low_opt.get('loss_weight', 1.0),
                    reduction=fft_low_opt.get('reduction', 'mean'),
                    highpass_cutoff=fft_low_opt.get('lowpass_cutoff', 0.3),  # Use as lowpass
                    highpass_type=fft_low_opt.get('filter_type', 'gaussian')
                ).to(self.device)
                # Mark this as a low-pass filter
                self.cri_fft_low.is_lowpass = True
                logger.info(f"Low-freq FFT Loss initialized with weight {fft_low_opt.get('loss_weight', 1.0)}, "
                           f"lowpass_cutoff {fft_low_opt.get('lowpass_cutoff', 0.3)}")
            else:
                self.cri_fft_low = None
            
            # Initialize ConvNeXt loss if configured
            if train_opt.get('convnext_opt'):
                from basicsr.utils import get_root_logger
                logger = get_root_logger()
                
                # Import our custom ConvNeXt loss
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from losses.convnext_loss import ConvNextLoss
                
                convnext_opt = train_opt['convnext_opt']
                self.cri_convnext = ConvNextLoss(
                    loss_weight=convnext_opt.get('loss_weight', 0.01),
                    model_type=convnext_opt.get('model_type', 'tiny'),
                    feature_layers=convnext_opt.get('feature_layers', None),
                    use_gram=convnext_opt.get('use_gram', False),
                    layer_weight_decay=convnext_opt.get('layer_weight_decay', 0.9),
                    reduction=convnext_opt.get('reduction', 'mean'),
                    input_range=convnext_opt.get('input_range', (0, 1))
                ).to(self.device)
                logger.info(f"ConvNeXt Loss initialized with weight {convnext_opt.get('loss_weight', 0.01)} "
                           f"using {convnext_opt.get('model_type', 'tiny')} model")
            else:
                self.cri_convnext = None
            
            # Initialize LPIPS loss if configured
            if train_opt.get('lpips_opt'):
                from basicsr.utils import get_root_logger
                logger = get_root_logger()
                
                # Import our custom LPIPS loss
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from losses.lpips_loss import LPIPSLoss
                
                lpips_opt = train_opt['lpips_opt']
                self.cri_lpips = LPIPSLoss(
                    loss_weight=lpips_opt.get('loss_weight', 1.0),
                    net_type=lpips_opt.get('net_type', 'alex'),
                    use_gpu=True,
                    spatial=lpips_opt.get('spatial', False),
                    version=lpips_opt.get('version', '0.1'),
                    normalize=lpips_opt.get('normalize', True),
                    reduction=lpips_opt.get('reduction', 'mean')
                ).to(self.device)
                # Store the start iteration for LPIPS
                self.lpips_start_iter = lpips_opt.get('start_iter', 0)
                logger.info(f"LPIPS Loss initialized with weight {lpips_opt.get('loss_weight', 1.0)} "
                           f"using {lpips_opt.get('net_type', 'alex')} network, "
                           f"starting at iteration {self.lpips_start_iter}")
            else:
                self.cri_lpips = None
                self.lpips_start_iter = float('inf')  # Never start if not configured
            
            # Initialize Focal Frequency Loss if configured
            if train_opt.get('focal_freq_opt'):
                from basicsr.utils import get_root_logger
                logger = get_root_logger()
                
                # Import our custom Focal Frequency Loss
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from losses.focal_frequency_loss import FocalFrequencyLoss
                
                focal_opt = train_opt['focal_freq_opt']
                self.cri_focal = FocalFrequencyLoss(
                    loss_weight=focal_opt.get('loss_weight', 1.0),
                    alpha=focal_opt.get('alpha', 1.0),
                    patch_factor=focal_opt.get('patch_factor', 1),
                    ave_spectrum=focal_opt.get('ave_spectrum', False),
                    log_matrix=focal_opt.get('log_matrix', False),
                    batch_matrix=focal_opt.get('batch_matrix', False)
                ).to(self.device)
                logger.info(f"Focal Frequency Loss initialized with weight {focal_opt.get('loss_weight', 1.0)}, "
                           f"alpha {focal_opt.get('alpha', 1.0)}")
            else:
                self.cri_focal = None
            
            # Initialize FDL (Feature Distance Loss) if configured
            if train_opt.get('fdl_opt'):
                from basicsr.utils import get_root_logger
                logger = get_root_logger()
                
                # Import our custom FDL Loss
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from losses.fdl_loss import FDLLoss
                
                fdl_opt = train_opt['fdl_opt']
                self.cri_fdl = FDLLoss(
                    loss_weight=fdl_opt.get('loss_weight', 1.0),
                    model=fdl_opt.get('model', 'VGG'),
                    patch_size=fdl_opt.get('patch_size', 5),
                    stride=fdl_opt.get('stride', 1),
                    num_proj=fdl_opt.get('num_proj', 256),
                    phase_weight=fdl_opt.get('phase_weight', 1.0),
                    reduction=fdl_opt.get('reduction', 'mean')
                ).to(self.device)
                logger.info(f"FDL Loss initialized with {fdl_opt.get('model', 'VGG')} backbone, "
                           f"weight {fdl_opt.get('loss_weight', 1.0)}, "
                           f"phase_weight {fdl_opt.get('phase_weight', 1.0)}")
            else:
                self.cri_fdl = None
                
            # Initialize discriminator-related losses if discriminator is enabled
            if self.use_discriminator:
                from basicsr.utils import get_root_logger
                logger = get_root_logger()
                
                # Initialize safe defaults for all GAN-related parameters
                self.fm_start_iter = float('inf')  # Disabled by default
                self.consistency_start_iter = float('inf')  # Disabled by default
                
                # GAN loss
                if train_opt.get('gan_opt'):
                    self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
                    logger.info(f"GAN Loss initialized with type {train_opt['gan_opt'].get('gan_type', 'vanilla')}")
                    
                # Feature Matching loss (if specified)
                if train_opt.get('feature_matching_opt'):
                    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from losses.feature_matching_loss import CIPLABFeatureMatchingLoss
                    
                    fm_opt = train_opt['feature_matching_opt']
                    self.cri_fm = CIPLABFeatureMatchingLoss(
                        loss_weight=fm_opt.get('loss_weight', 1.0)
                    ).to(self.device)
                    self.fm_weight = fm_opt.get('loss_weight', 1.0)
                    self.fm_start_iter = fm_opt.get('start_iter', 0)
                    logger.info(f"CIPLAB Feature Matching Loss initialized with weight {self.fm_weight}")
                    
                # Consistency/CutMix loss (if specified)  
                if train_opt.get('consistency_opt'):
                    self.cri_consistency = nn.MSELoss(reduction='mean').to(self.device)
                    self.consistency_weight = train_opt['consistency_opt'].get('loss_weight', 1.0)
                    self.cutmix_prob_max = train_opt['consistency_opt'].get('cutmix_prob_max', 0.5)
                    self.cutmix_prob_scale = train_opt['consistency_opt'].get('cutmix_prob_scale', 100000)
                    self.consistency_start_iter = train_opt['consistency_opt'].get('start_iter', 75000)
                    logger.info(f"CutMix Consistency Loss initialized with weight {self.consistency_weight}")
                    
                # D update frequency settings
                self.net_d_iters = train_opt.get('net_d_iters', 1)
                self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
                self.perceptual_start_iter = train_opt.get('perceptual_start_iter', 75000)
                logger.info(f"Discriminator training starts at iteration {self.perceptual_start_iter}")
    
    def _load_discriminator_network(self, load_path):
        """Helper method to load discriminator network with flexible checkpoint format handling."""
        logger = get_root_logger()
        param_key = self.opt['path'].get('param_key_d', 'params')
        strict = self.opt['path'].get('strict_load_d', True)
        
        # First, load the checkpoint to check its structure
        checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
        
        # Check if it's a direct state_dict or has the expected structure
        if isinstance(checkpoint, dict):
            # Check if it has the param_key
            if param_key in checkpoint:
                # Standard format, use normal loading
                self.load_network(self.net_d, load_path, strict, param_key)
            elif 'params' in checkpoint and param_key != 'params':
                # Has 'params' but we're looking for a different key
                logger.warning(f"Expected key '{param_key}' not found, but 'params' exists. Using 'params' instead.")
                self.load_network(self.net_d, load_path, strict, 'params')
            else:
                # It's likely a direct state_dict, load it directly
                logger.info(f"Loading discriminator as direct state_dict from {load_path}")
                net = self.get_bare_model(self.net_d)
                # Remove 'module.' prefix if present
                for k, v in deepcopy(checkpoint).items():
                    if k.startswith('module.'):
                        checkpoint[k[7:]] = v
                        checkpoint.pop(k)
                net.load_state_dict(checkpoint, strict=strict)
        else:
            # Not a dict, might be an OrderedDict or direct tensor storage
            logger.info(f"Loading discriminator from non-dict checkpoint at {load_path}")
            net = self.get_bare_model(self.net_d)
            net.load_state_dict(checkpoint, strict=strict)
    
    def feed_data(self, data):
        """Override feed_data to store current batch info for logging."""
        self.current_data = data  # Store the full data dict
        super(RecurrentMixPrecisionRTModel, self).feed_data(data)

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        """Override to add AdamW support."""
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer
    
    # add use_static_graph
    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', False)
            net = DistributedDataParallel(
                net, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
            use_static_graph = self.opt.get('use_static_graph', False)
            if use_static_graph:
                logger = get_root_logger()
                logger.info(
                    f'Using static graph. Make sure that "unused parameters" will not change during training loop.')
                net._set_static_graph()
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                # add 'deform'
                if 'spynet' in name or 'deform' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_g_config = train_opt['optim_g'].copy()
        optim_type = optim_g_config.pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **optim_g_config)

        # # adopt mix precision
        # use_apex_amp = self.opt.get('apex_amp', False)
        # if use_apex_amp:
        #     self.net_g, self.optimizer_g = apex_amp_initialize(
        #         self.net_g, self.optimizer_g, init_args=dict(opt_level='O1'))
        #     logger = get_root_logger()
        #     logger.info(f'Using apex mix precision to accelerate.')

        # adopt DDP
        self.net_g = self.model_to_device(self.net_g)
        self.optimizers.append(self.optimizer_g)
        
        # Setup discriminator optimizer if using GAN training
        if self.use_discriminator:
            train_opt = self.opt['train']
            
            # Discriminator optimizer
            optim_d_config = train_opt['optim_d'].copy()
            optim_type_d = optim_d_config.pop('type')
            self.optimizer_d = self.get_optimizer(
                optim_type_d, 
                self.net_d.parameters(), 
                **optim_d_config
            )
            
            # Apply DDP to discriminator
            self.net_d = self.model_to_device(self.net_d)
            self.optimizers.append(self.optimizer_d)
            
            logger = get_root_logger()
            logger.info(f"Discriminator optimizer initialized with {optim_type_d}")

    def optimize_parameters(self, scaler, current_iter):
        # Store current iteration for use in optimization
        self.current_iter = current_iter

        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'deform' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        # Gradient accumulation settings
        accumulation_steps = self.opt['train'].get('gradient_accumulation_steps', 1)
        
        # Only zero gradients at the beginning of accumulation cycle
        # current_iter starts at 1, so we need to adjust for modulo
        if (current_iter - 1) % accumulation_steps == 0:
            self.optimizer_g.zero_grad()

        # Disable autocast for float32 training (more stable but uses 2x memory)
        use_mixed_precision = self.opt['train'].get('half_precision', True)
        
        with autocast('cuda', enabled=use_mixed_precision):
            self.output = self.net_g(self.lq)
            
            # Check for NaN immediately after model forward pass
            if torch.isnan(self.output).any():
                logger = get_root_logger()
                logger.error(f"CRITICAL: Model forward pass produced NaN! Model weights may be corrupted.")
                # Check if model parameters have NaN (only check first few to avoid OOM)
                param_checked = 0
                for name, param in self.net_g.named_parameters():
                    if param_checked > 5:  # Only check first 5 parameters
                        break
                    if torch.isnan(param).any():
                        logger.error(f"  NaN found in parameter: {name}")
                        break
                    param_checked += 1
                # Clean up to prevent OOM
                del self.output
                torch.cuda.empty_cache()
                # Don't zero_grad() here - preserve accumulated gradients
                return
            
            # Clamp output to valid range to prevent pure black
            self.output = torch.clamp(self.output, min=0.0, max=1.0)
            
        # ========== DISCRIMINATOR UPDATE (if enabled and past pretraining) ==========
        if self.use_discriminator:
            use_perceptual = current_iter >= self.perceptual_start_iter
            
            # Only update D every net_d_iters and after net_d_init_iters
            if use_perceptual and current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters:
                self.optimize_discriminator(scaler, current_iter)
        else:
            use_perceptual = False
            
        # ========== GENERATOR UPDATE ==========
        # Continue with generator optimization
        with autocast('cuda', enabled=use_mixed_precision):
            l_total = 0
            loss_dict = OrderedDict()
            # Check if model is outputting black/invalid/NaN frames
            if torch.isnan(self.output).any():
                logger = get_root_logger()
                logger.warning(f"Model outputting NaN after clamp! Skipping iteration.")
                # Clean up and skip
                del self.output
                torch.cuda.empty_cache()
                # Don't zero_grad() here - preserve accumulated gradients
                return
            
            output_mean = self.output.mean().item()
            if output_mean < 0.001:
                logger = get_root_logger()
                logger.warning(f"Model outputting near-black frames! Mean: {output_mean:.6f}")
                # Clean up and skip
                del self.output
                torch.cuda.empty_cache()
                # Don't zero_grad() here - preserve accumulated gradients
                return
            
            skip_because_too_dark = False
            check_too_dark_patches = False
            
            if check_too_dark_patches:
                output_mean = output_4d.mean()
                gt_mean = gt_4d.mean()
                if output_mean > 0.02 and gt_mean > 0.02:
                    skip_because_too_dark = False
                else:
                    skip_because_too_dark = True
                    logger = get_root_logger()
                    logger.info(f"Skipped loss for dark patch (output mean: {output_mean:.4f}, gt mean: {gt_mean:.4f})")
            
            allow_patch = not skip_because_too_dark

            # pixel loss
            if self.cri_pix:
                l_pix = self.cri_pix(self.output, self.gt)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix
            # perceptual loss
            if self.cri_perceptual:
                # Reshape 5D video tensors to 4D for perceptual loss
                # self.output shape: [B, T, C, H, W] -> [B*T, C, H, W]
                b, t, c, h, w = self.output.shape
                output_4d = self.output.view(b * t, c, h, w)
                gt_4d = self.gt.view(b * t, c, h, w)
                
                if allow_patch:
                    l_percep, l_style = self.cri_perceptual(output_4d, gt_4d)
                else:
                    l_percep = torch.tensor(0.0, device=self.device)
                    l_style = torch.tensor(0.0, device=self.device)
                if l_percep is not None:
                    l_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_total += l_style
                    loss_dict['l_style'] = l_style
            
            # FFT loss for preserving high-frequency details
            if hasattr(self, 'cri_fft') and self.cri_fft:
                l_fft = self.cri_fft(self.output, self.gt)
                l_total += l_fft
                loss_dict['l_fft'] = l_fft
            
            # High-frequency FFT loss (for sharpness and details)
            if hasattr(self, 'cri_fft_high') and self.cri_fft_high:
                l_fft_high = self.cri_fft_high(self.output, self.gt)
                l_total += l_fft_high
                loss_dict['l_fft_high'] = l_fft_high
            
            # Low-frequency FFT loss (for color and brightness matching)
            if hasattr(self, 'cri_fft_low') and self.cri_fft_low:
                l_fft_low = self.cri_fft_low(self.output, self.gt)
                l_total += l_fft_low
                loss_dict['l_fft_low'] = l_fft_low
            
            # ConvNeXt perceptual loss for better stability than VGG
            if hasattr(self, 'cri_convnext') and self.cri_convnext:
                # Reshape 5D video tensors to 4D for ConvNeXt loss
                if self.output.dim() == 5:
                    b, t, c, h, w = self.output.shape
                    output_4d = self.output.view(b * t, c, h, w)
                    gt_4d = self.gt.view(b * t, c, h, w)
                else:
                    output_4d = self.output
                    gt_4d = self.gt
                
                if allow_patch:
                    l_convnext = self.cri_convnext(output_4d, gt_4d)
                else:
                    l_convnext = torch.tensor(0.0, device=self.device)
                
                l_total += l_convnext
                loss_dict['l_convnext'] = l_convnext
            
            # Focal Frequency Loss - adaptively focuses on hard frequencies
            if hasattr(self, 'cri_focal') and self.cri_focal:
                l_focal = self.cri_focal(self.output, self.gt)
                l_total += l_focal
                loss_dict['l_focal'] = l_focal
            
            # FDL (Feature Distance Loss) - robust to misalignment
            if hasattr(self, 'cri_fdl') and self.cri_fdl:
                l_fdl = self.cri_fdl(self.output, self.gt)
                l_total += l_fdl
                loss_dict['l_fdl'] = l_fdl
            
            # LPIPS perceptual loss - human-calibrated
            if hasattr(self, 'cri_lpips') and self.cri_lpips and current_iter >= self.lpips_start_iter:
                # Reshape 5D video tensors to 4D for LPIPS loss
                if self.output.dim() == 5:
                    b, t, c, h, w = self.output.shape
                    output_4d = self.output.view(b * t, c, h, w)
                    gt_4d = self.gt.view(b * t, c, h, w)
                else:
                    output_4d = self.output
                    gt_4d = self.gt
                
                if allow_patch:
                    l_lpips = self.cri_lpips(output_4d, gt_4d)
                else:
                    l_lpips = torch.tensor(0.0, device=self.device)
                
                l_total += l_lpips
                loss_dict['l_lpips'] = l_lpips
            
            # ========== GAN-RELATED LOSSES (if discriminator enabled and past pretraining) ==========
            if self.use_discriminator and use_perceptual:
                # Freeze D, unfreeze G for generator training (only if not already in correct state)
                if not hasattr(self, '_g_training_mode') or not self._g_training_mode:
                    for p in self.net_d.parameters():
                        p.requires_grad = False
                    for p in self.net_g.parameters():
                        p.requires_grad = True
                    self._g_training_mode = True
                    self._d_training_mode = False
                
                # Reshape video tensors if needed
                if self.output.dim() == 5:
                    b, t, c, h, w = self.output.shape
                    output_d = self.output.view(b * t, c, h, w)
                    gt_d = self.gt.view(b * t, c, h, w)
                else:
                    output_d = self.output
                    gt_d = self.gt
                    
                # Get discriminator predictions for fake samples
                if hasattr(self.net_d, 'forward_with_features'):
                    # CIPLAB discriminator returns e_out, d_out, encoder_feats, decoder_feats
                    fake_pred_e, fake_pred_d, fake_enc_feats, fake_dec_feats = self.net_d.forward_with_features(output_d)
                    
                    # Feature Matching loss (if configured)
                    if hasattr(self, 'cri_fm') and current_iter >= self.fm_start_iter:
                        # Get real features (use stored from D update if available, else compute)
                        if hasattr(self, 'real_enc_feats') and hasattr(self, 'real_dec_feats'):
                            real_enc_feats = self.real_enc_feats
                            real_dec_feats = self.real_dec_feats
                        else:
                            with torch.no_grad():
                                _, _, real_enc_feats, real_dec_feats = self.net_d.forward_with_features(gt_d)
                        
                        # Use CIPLAB Feature Matching loss
                        l_fm = self.cri_fm(fake_enc_feats, fake_dec_feats, 
                                          real_enc_feats, real_dec_feats)
                        l_total += l_fm  # Loss weight already applied in cri_fm
                        loss_dict['l_fm'] = l_fm
                        
                        # Clean up stored features to prevent memory accumulation
                        if hasattr(self, 'real_enc_feats'):
                            del self.real_enc_feats
                        if hasattr(self, 'real_dec_feats'):
                            del self.real_dec_feats
                else:
                    fake_pred_e, fake_pred_d = self.net_d(output_d)
                    
                # Adversarial loss for generator (CIPLAB-style)
                if hasattr(self, 'cri_gan'):
                    # Apply GAN loss to encoder and decoder outputs separately
                    l_g_gan_e = self.cri_gan(fake_pred_e, True, is_disc=False)
                    l_g_gan_d = self.cri_gan(fake_pred_d, True, is_disc=False)
                    l_g_gan = (l_g_gan_e + l_g_gan_d) / 2  # Average the losses (not the outputs!)
                    
                    # Apply GAN loss weight from config
                    gan_weight = self.opt['train'].get('gan_opt', {}).get('loss_weight', 0.001)
                    l_total += l_g_gan * gan_weight
                    loss_dict['l_g_gan'] = l_g_gan
                    loss_dict['out_g_fake'] = (torch.mean(fake_pred_e.detach()) + torch.mean(fake_pred_d.detach())) / 2
            
            # Scale the loss by accumulation steps to maintain effective learning rate
            l_total = l_total / accumulation_steps
            
            # Backward pass (accumulates gradients)
            scaler.scale(l_total).backward()
            
            # Only perform optimizer step after accumulation_steps iterations
            if (self.current_iter) % accumulation_steps == 0:
                # Always unscale before gradient operations
                scaler.unscale_(self.optimizer_g)
                
                # Gradient clipping and norm calculation
                if self.opt['train'].get('use_grad_clip', False):
                    max_norm = self.opt['train'].get('grad_clip_norm', 0.5)
                    # clip_grad_norm_ returns the norm before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=max_norm)
                else:
                    # Just compute norm for logging without clipping
                    total_norm = 0.0
                    for p in self.net_g.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    grad_norm = total_norm ** 0.5
                
                # Log gradient norm and learning rate periodically
                if self.current_iter % 50 == 0:
                    logger = get_root_logger()
                    current_lr = self.optimizer_g.param_groups[0]['lr']
                    logger.info(f"Iter {self.current_iter}: grad_norm={grad_norm:.4f}, lr={current_lr:.2e}, "
                               f"gradient_accumulation={accumulation_steps}")
                
                # Skip update if gradients are bad
                if torch.isnan(grad_norm) or grad_norm > 1000:  # Increased threshold
                    logger = get_root_logger()
                    logger.warning(f"Skipping update - bad gradients detected! Norm: {grad_norm:.2f}")
                    self.optimizer_g.zero_grad()
                    scaler.update()  # Still need to update scaler
                    
                    # Track bad updates
                    if not hasattr(self, 'bad_update_count'):
                        self.bad_update_count = 0
                    self.bad_update_count += 1
                    
                    # Reset optimizer if too many bad updates
                    if self.bad_update_count >= 3:
                        logger.warning("Too many bad updates - resetting optimizer state!")
                        self.optimizer_g.state.clear()
                        self.bad_update_count = 0
                    return
                
                # Reset bad update counter on successful update
                if hasattr(self, 'bad_update_count'):
                    self.bad_update_count = 0
                
                # Periodic optimizer reset to prevent long-term accumulation
                # Reset every 5000 iterations (adjust as needed)
                if self.current_iter > 0 and self.current_iter % 5000 == 0:
                    logger = get_root_logger()
                    logger.info(f"Periodic optimizer reset at iteration {self.current_iter}")
                    self.optimizer_g.state.clear()
                
                # Actually update weights (gradient clipping already done above)
                scaler.step(self.optimizer_g)
                # Update scaler after step (tracks gradient health for next iteration)
                scaler.update()
            else:
                # Log that we're accumulating gradients
                if self.current_iter % 100 == 0:
                    logger = get_root_logger()
                    logger.debug(f"Iter {self.current_iter}: Accumulating gradients ({self.current_iter % accumulation_steps}/{accumulation_steps})")
                # During accumulation, we don't call scaler.update() because
                # no inf checks were performed (no unscale or step happened)

            # l_total.backward()
            # self.optimizer_g.step()

            # Merge discriminator losses if they exist
            if hasattr(self, 'log_dict'):
                # Discriminator already ran and created log_dict
                self.log_dict.update(self.reduce_loss_dict(loss_dict))
            else:
                # Normal case - create log_dict from generator losses
                self.log_dict = self.reduce_loss_dict(loss_dict)
            
            # Log detailed losses every 50 iterations to debug learning
            if self.current_iter % 50 == 0:
                logger = get_root_logger()
                loss_str = ' | '.join([f'{k}={v:.4f}' for k, v in self.log_dict.items()])
                logger.info(f"Losses at iter {self.current_iter}: {loss_str}")
            
            # Log detailed sample information when NaN is detected or losses are unusually high
            if hasattr(self, 'current_data'):
                # Get threshold from config, default to 0.1
                high_loss_threshold = self.opt['train'].get('high_loss_threshold', 0.1)
                
                # Check for NaN or very high losses
                has_nan = any(torch.isnan(v) if torch.is_tensor(v) else math.isnan(v) 
                             for v in loss_dict.values())
                has_high_loss = any((v > high_loss_threshold if torch.is_tensor(v) else v > high_loss_threshold) 
                                   for v in loss_dict.values())
                
                if has_nan or has_high_loss:
                    logger = get_root_logger()
                    data = self.current_data
                    key = data.get('key', 'unknown')
                    coords = data.get('crop_coords', {})
                    interval = data.get('interval', 1)
                    
                    # Extract actual values from tensors
                    if torch.is_tensor(key):
                        key = key.item() if key.numel() == 1 else key.tolist()
                    if torch.is_tensor(interval):
                        interval = interval.item() if interval.numel() == 1 else interval.tolist()
                    
                    # Extract crop coordinates
                    gt_top = coords.get('gt_top', 0)
                    gt_left = coords.get('gt_left', 0)
                    gt_size = coords.get('gt_size', 0)
                    if torch.is_tensor(gt_top):
                        gt_top = gt_top.item() if gt_top.numel() == 1 else gt_top.tolist()
                    if torch.is_tensor(gt_left):
                        gt_left = gt_left.item() if gt_left.numel() == 1 else gt_left.tolist()
                    if torch.is_tensor(gt_size):
                        gt_size = gt_size.item() if gt_size.numel() == 1 else gt_size.tolist()
                    
                    # Format losses for logging
                    loss_str = ', '.join([f"{k}: {v:.4f}" if not (torch.isnan(v) if torch.is_tensor(v) else math.isnan(v)) 
                                        else f"{k}: NaN" for k, v in loss_dict.items()])
                    
                    severity = "NaN DETECTED" if has_nan else "HIGH LOSS"
                    logger.warning(f"[{severity}] Sample: {key} | Crop: (top={gt_top}, left={gt_left}, "
                              f"size={gt_size}) | Interval: {interval} | Losses: {loss_str}")
                    
                    # Save problematic images for debugging
                    if has_nan or (has_high_loss and self.opt['train'].get('save_high_loss_images', False)):
                        debug_dir = os.path.join(self.opt['path']['visualization'], 'debug_patches')
                        os.makedirs(debug_dir, exist_ok=True)
                        
                        # Clean key for filename
                        clean_key = str(key).replace('/', '_').replace('[', '').replace(']', '').replace('\'', '')
                        base_filename = f"iter_{current_iter:06d}_{clean_key}_top{gt_top}_left{gt_left}"
                        
                        # Save the images (handle video tensors)
                        if self.lq.dim() == 5:  # Video tensor (B, T, C, H, W)
                            # Save middle frame with original naming for compatibility
                            mid_frame = self.lq.shape[1] // 2
                            lq_save = self.lq[:, mid_frame, :, :, :]
                            gt_save = self.gt[:, mid_frame, :, :, :]
                            output_save = self.output[:, mid_frame, :, :, :]
                            
                            # Save middle frame with original filenames
                            torchvision.utils.save_image(
                                lq_save, 
                                os.path.join(debug_dir, f"{base_filename}_lq.png"),
                                normalize=False
                            )
                            torchvision.utils.save_image(
                                gt_save,
                                os.path.join(debug_dir, f"{base_filename}_gt.png"),
                                normalize=False
                            )
                            torchvision.utils.save_image(
                                output_save,
                                os.path.join(debug_dir, f"{base_filename}_output.png"),
                                normalize=False
                            )
                            
                            # Additionally, save ALL frames for debugging temporal alignment
                            frames_dir = os.path.join(debug_dir, f"{base_filename}_frames")
                            os.makedirs(frames_dir, exist_ok=True)
                            
                            num_frames = self.lq.shape[1]
                            for frame_idx in range(num_frames):
                                # Save each frame individually
                                torchvision.utils.save_image(
                                    self.lq[:, frame_idx, :, :, :],
                                    os.path.join(frames_dir, f"frame_{frame_idx:02d}_lq.png"),
                                    normalize=False
                                )
                                torchvision.utils.save_image(
                                    self.gt[:, frame_idx, :, :, :],
                                    os.path.join(frames_dir, f"frame_{frame_idx:02d}_gt.png"),
                                    normalize=False
                                )
                                torchvision.utils.save_image(
                                    self.output[:, frame_idx, :, :, :],
                                    os.path.join(frames_dir, f"frame_{frame_idx:02d}_output.png"),
                                    normalize=False
                                )
                            
                            # Create a grid showing all frames for quick visual inspection
                            # Concatenate all output frames horizontally
                            output_frames = []
                            for i in range(num_frames):
                                output_frames.append(self.output[:, i, :, :, :])
                            output_grid = torch.cat(output_frames, dim=3)  # Concatenate along width
                            torchvision.utils.save_image(
                                output_grid,
                                os.path.join(frames_dir, "all_output_frames.png"),
                                normalize=False
                            )
                            
                            # Create comparison grid for each frame
                            for i in range(num_frames):
                                # Upscale LQ to match GT/output size for comparison
                                lq_frame = self.lq[:, i, :, :, :]
                                output_frame = self.output[:, i, :, :, :]
                                gt_frame = self.gt[:, i, :, :, :]
                                
                                # Upscale LQ using bilinear interpolation to match GT size
                                if lq_frame.shape[-2:] != gt_frame.shape[-2:]:
                                    import torch.nn.functional as F
                                    lq_frame_upscaled = F.interpolate(
                                        lq_frame, 
                                        size=gt_frame.shape[-2:], 
                                        mode='bilinear', 
                                        align_corners=False
                                    )
                                else:
                                    lq_frame_upscaled = lq_frame
                                
                                frame_comparison = torch.cat([
                                    lq_frame_upscaled,
                                    output_frame,
                                    gt_frame
                                ], dim=3)  # LQ | Output | GT horizontally
                                torchvision.utils.save_image(
                                    frame_comparison,
                                    os.path.join(frames_dir, f"frame_{i:02d}_comparison.png"),
                                    normalize=False
                                )
                                
                        else:
                            lq_save = self.lq
                            gt_save = self.gt
                            output_save = self.output
                            
                            # Save regular images
                            torchvision.utils.save_image(
                                lq_save, 
                                os.path.join(debug_dir, f"{base_filename}_lq.png"),
                                normalize=False
                            )
                            torchvision.utils.save_image(
                                gt_save,
                                os.path.join(debug_dir, f"{base_filename}_gt.png"),
                                normalize=False
                            )
                            torchvision.utils.save_image(
                                output_save,
                                os.path.join(debug_dir, f"{base_filename}_output.png"),
                                normalize=False
                            )
                        
                        logger.info(f"Saved debug images to {debug_dir}/{base_filename}_*.png")

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Custom validation with image saving support.
        
        Override BasicSR's validation to allow saving images during training.
        """
        import os
        from collections import Counter
        from os import path as osp
        import numpy as np
        import torch
        import torchvision
        import cv2
        from tqdm import tqdm
        from basicsr.metrics import calculate_metric
        from basicsr.utils import tensor2img, imwrite
        from basicsr.utils.dist_util import get_dist_info
        
        # Set model to eval mode and disable gradient computation
        self.net_g.eval()
        
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        
        # Save images whenever save_img is true
        save_img_this_iter = save_img
        
        # Check if metrics are available
        with_metrics = self.opt['val'].get('metrics') is not None
        
        # Initialize metric results
        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {}
                num_frame_each_folder = Counter(dataset.data_info['folder'])
                for folder, num_frame in num_frame_each_folder.items():
                    self.metric_results[folder] = torch.zeros(
                        num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
            # Initialize best metric results
            self._initialize_best_metric_results(dataset_name)
            
            # Zero metric results
            for _, tensor in self.metric_results.items():
                tensor.zero_()
        
        # Get rank info for distributed training
        rank, world_size = get_dist_info()
        
        # Process validation data
        metric_data = dict()
        num_folders = len(dataset)
        
        if rank == 0:
            pbar = tqdm(total=num_folders, unit='folder', desc='Validation')
        
        # Limit validation to first 5 clips for speed during testing
        max_folders = min(5, num_folders) if current_iter <= 10 else num_folders
        
        with torch.no_grad():  # Disable gradient computation for validation
            for idx in range(max_folders):
                val_data = dataset[idx]
                folder = val_data['folder']
                
                # Add batch dimension
                val_data['lq'].unsqueeze_(0)
                val_data['gt'].unsqueeze_(0)
                self.feed_data(val_data)
                val_data['lq'].squeeze_(0)
                val_data['gt'].squeeze_(0)
                
                # Test (forward pass)
                self.test()
                visuals = self.get_current_visuals()
                
                # Process output - handle video tensors properly
                # For video tensors, extract a specific frame for metrics
                if len(visuals['result'].shape) == 5:  # [B, T, C, H, W]
                    # Use middle frame for consistency with grid display
                    mid_frame = visuals['result'].shape[1] // 2
                    result_img = tensor2img([visuals['result'][0, mid_frame]])  # uint8, bgr
                    gt_img = tensor2img([visuals['gt'][0, mid_frame]])  # uint8, bgr
                    lq_img = tensor2img([visuals['lq'][0, mid_frame]])  # uint8, bgr
                else:
                    result_img = tensor2img([visuals['result']])  # uint8, bgr
                    gt_img = tensor2img([visuals['gt']])  # uint8, bgr
                    lq_img = tensor2img([visuals['lq']])  # uint8, bgr
                
                # Calculate metrics
                if with_metrics:
                    for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                        metric_type = opt_['type']
                        metric_result = calculate_metric(
                            {'img': result_img, 'img2': gt_img}, opt_)
                        metric_data[f'{folder}_{metric_type}'] = metric_result
                        
                        # Store in metric_results
                        if hasattr(self, 'metric_results'):
                            self.metric_results[folder][0, metric_idx] += metric_result
                
                # Save images
                if save_img_this_iter:
                    if self.opt['val'].get('grids', False):
                        # For video tensors, extract middle frame for comparison
                        if len(visuals['result'].shape) == 5:  # [B, T, C, H, W]
                            mid_frame = visuals['result'].shape[1] // 2
                            result_frame = tensor2img([visuals['result'][0, mid_frame]])
                            gt_frame = tensor2img([visuals['gt'][0, mid_frame]])
                            lq_frame = tensor2img([visuals['lq'][0, mid_frame]])
                        else:
                            result_frame = result_img
                            gt_frame = gt_img
                            lq_frame = lq_img
                        
                        # Create comparison grid: LQ | Output | GT
                        # All images are already uint8 BGR numpy arrays
                        h, w = gt_frame.shape[:2]
                        lq_resized = cv2.resize(lq_frame, (w, h), interpolation=cv2.INTER_CUBIC)
                        
                        # Create grid
                        grid = np.concatenate([lq_resized, result_frame, gt_frame], axis=1)
                        
                        # Save grid
                        save_folder = osp.join(self.opt['path']['visualization'], 
                                             f'val_images_iter_{current_iter:06d}')
                        os.makedirs(save_folder, exist_ok=True)
                        save_path = osp.join(save_folder, f'{folder}_grid.png')
                        imwrite(grid, save_path)
                        
                        # Also save the generated frame separately
                        output_path = osp.join(save_folder, f'{folder}_output.png')
                        imwrite(result_frame, output_path)
                        
                        # Log to W&B if available
                        try:
                            import wandb
                            if wandb.run is not None:
                                # Convert BGR to RGB for W&B
                                grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
                                result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                                
                                # Log comparison grid
                                wandb.log({
                                    f'val/{folder}_grid': wandb.Image(grid_rgb, 
                                        caption=f'{folder} - LQ | Output | GT'),
                                    f'val/{folder}_output': wandb.Image(result_rgb,
                                        caption=f'{folder} - Output')
                                }, commit=False)
                        except ImportError:
                            pass  # W&B not installed
                    else:
                        # Save individual images
                        save_folder = osp.join(self.opt['path']['visualization'], 
                                             f'val_images_iter_{current_iter:06d}')
                        os.makedirs(save_folder, exist_ok=True)
                        
                        # Save LQ, output, GT
                        imwrite(lq_img, osp.join(save_folder, f'{folder}_lq.png'))
                        imwrite(result_img, osp.join(save_folder, f'{folder}_output.png'))
                        imwrite(gt_img, osp.join(save_folder, f'{folder}_gt.png'))
                        
                        # Log to W&B if available
                        try:
                            import wandb
                            if wandb.run is not None:
                                # Convert BGR to RGB for W&B
                                lq_rgb = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)
                                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                                gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                                
                                # Log individual images
                                wandb.log({
                                    f'val/{folder}_lq': wandb.Image(lq_rgb, caption=f'{folder} - LQ'),
                                    f'val/{folder}_output': wandb.Image(result_rgb, caption=f'{folder} - Output'),
                                    f'val/{folder}_gt': wandb.Image(gt_rgb, caption=f'{folder} - GT')
                                }, commit=False)
                        except ImportError:
                            pass  # W&B not installed
                
                # Clean up visuals and GPU memory after each validation clip
                del visuals
                if hasattr(self, 'lq'):
                    del self.lq
                if hasattr(self, 'gt'):
                    del self.gt
                if hasattr(self, 'output'):
                    del self.output
                torch.cuda.empty_cache()
                
                if rank == 0:
                    pbar.update(1)
                    pbar.set_postfix(**{k: f'{v:.3f}' for k, v in metric_data.items() 
                                       if folder in k})
        
        if rank == 0:
            pbar.close()
        
        # Calculate average metrics
        if with_metrics:
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            
            # Log average metrics
            avg_metrics = {}
            for metric_type in self.opt['val']['metrics'].keys():
                values = [v for k, v in metric_data.items() if metric_type in k]
                if values:
                    avg_metrics[metric_type] = sum(values) / len(values)
            
            logger = get_root_logger()
            log_str = f'Validation {dataset_name} - Iter {current_iter}: '
            for metric, value in avg_metrics.items():
                log_str += f'{metric}: {value:.4f}  '
            logger.info(log_str)
            
            if save_img_this_iter:
                logger.info(f'Validation images saved to: {save_folder}')
        
        # Set model back to train mode
        self.net_g.train()
        if self.use_discriminator:
            self.net_d.train()
        
        # Final cleanup of GPU memory after validation
        if hasattr(self, 'lq'):
            del self.lq
        if hasattr(self, 'gt'):
            del self.gt
        if hasattr(self, 'output'):
            del self.output
        torch.cuda.empty_cache()
    
    # ========== DISCRIMINATOR SUPPORT METHODS ==========
    
    def optimize_discriminator(self, scaler, current_iter):
        """Separate discriminator optimization step with gradient accumulation support"""
        # Freeze G, unfreeze D (only if not already in correct state)
        if not hasattr(self, '_d_training_mode') or not self._d_training_mode:
            for p in self.net_g.parameters():
                p.requires_grad = False
            for p in self.net_d.parameters():
                p.requires_grad = True
            self._d_training_mode = True
            self._g_training_mode = False
        
        # Get gradient accumulation steps for discriminator (can be different from generator)
        d_accumulation_steps = self.opt['train'].get('d_gradient_accumulation_steps', 
                                                     self.opt['train'].get('gradient_accumulation_steps', 1))
        
        # Only zero grad at the start of accumulation cycle
        if not hasattr(self, '_d_accumulation_counter'):
            self._d_accumulation_counter = 0
        
        if self._d_accumulation_counter == 0:
            self.optimizer_d.zero_grad()
        
        loss_dict = OrderedDict()
        
        with autocast('cuda', enabled=self.opt['train'].get('half_precision', True)):
            # Reshape video tensors if needed
            if self.gt.dim() == 5:
                b, t, c, h, w = self.gt.shape
                gt_d = self.gt.view(b * t, c, h, w)
                output_d = self.output.view(b * t, c, h, w)
            else:
                gt_d = self.gt
                output_d = self.output
            
            # Real samples
            if hasattr(self.net_d, 'forward_with_features'):
                real_pred_e, real_pred_d, self.real_enc_feats, self.real_dec_feats = self.net_d.forward_with_features(gt_d)
            else:
                real_pred_e, real_pred_d = self.net_d(gt_d)
            
            # CIPLAB-style: Apply GAN loss to encoder and decoder outputs separately
            l_d_real_e = self.cri_gan(real_pred_e, True, is_disc=True)
            l_d_real_d = self.cri_gan(real_pred_d, True, is_disc=True)
            l_d_real = l_d_real_e + l_d_real_d  # Sum for discriminator (not average)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = (torch.mean(real_pred_e.detach()) + torch.mean(real_pred_d.detach())) / 2
            
            # Fake samples (detached to prevent backprop to G)
            if hasattr(self.net_d, 'forward_with_features'):
                fake_pred_e, fake_pred_d, fake_enc_feats, fake_dec_feats = self.net_d.forward_with_features(output_d.detach())
            else:
                fake_pred_e, fake_pred_d = self.net_d(output_d.detach())
                
            # CIPLAB-style: Apply GAN loss to encoder and decoder outputs separately
            l_d_fake_e = self.cri_gan(fake_pred_e, False, is_disc=True)
            l_d_fake_d = self.cri_gan(fake_pred_d, False, is_disc=True)
            l_d_fake = l_d_fake_e + l_d_fake_d  # Sum for discriminator (not average)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = (torch.mean(fake_pred_e.detach()) + torch.mean(fake_pred_d.detach())) / 2
            
            # CutMix consistency loss (CIPLAB-style)
            l_d_consistency = 0
            if hasattr(self, 'cri_consistency') and current_iter >= self.consistency_start_iter:
                l_d_consistency = self.compute_cutmix_consistency(gt_d, output_d.detach(), current_iter)
                if l_d_consistency > 0:
                    loss_dict['l_d_consistency'] = l_d_consistency
            
            # Scale loss by accumulation steps to maintain effective learning rate
            l_d_total = (l_d_real + l_d_fake + l_d_consistency) / d_accumulation_steps
        
        # Accumulate gradients
        scaler.scale(l_d_total).backward()
        
        # Increment accumulation counter
        self._d_accumulation_counter += 1
        
        # Only update weights after accumulating enough gradients
        if self._d_accumulation_counter >= d_accumulation_steps:
            scaler.unscale_(self.optimizer_d)
            
            # Gradient clipping for D
            if self.opt['train'].get('use_grad_clip', False):
                max_norm = self.opt['train'].get('grad_clip_norm', 0.1)
                torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), max_norm=max_norm)
            
            scaler.step(self.optimizer_d)
            scaler.update()
            
            # Reset accumulation counter
            self._d_accumulation_counter = 0
        
        # Update or create log dict
        if hasattr(self, 'log_dict'):
            self.log_dict.update(self.reduce_loss_dict(loss_dict))
        else:
            self.log_dict = self.reduce_loss_dict(loss_dict)
    
    def compute_feature_matching_loss(self, fake_features, real_features):
        """Compute Feature Matching loss across all feature layers"""
        loss = 0
        num_features = len(fake_features)
        for fake_feat, real_feat in zip(fake_features, real_features):
            loss += self.cri_fm(fake_feat, real_feat.detach())
        return loss / num_features if num_features > 0 else 0
    
    def rand_bbox(self, size, lam):
        """Generate random bounding box for CutMix"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def compute_cutmix_consistency(self, real_batch, fake_batch, current_iter):
        """CIPLAB-style CutMix consistency loss for discriminator regularization"""
        # Probability increases from 0 to cutmix_prob_max over cutmix_prob_scale iterations
        p_mix = min(current_iter / self.cutmix_prob_scale, self.cutmix_prob_max)
        
        if torch.rand(1).item() <= p_mix:
            # Create mixed batch
            batch_mixed = fake_batch.clone()
            lam = torch.rand(1).item()  # Real/fake ratio
            
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(batch_mixed.size(), lam)
            batch_mixed[:, :, bbx1:bbx2, bby1:bby2] = real_batch[:, :, bbx1:bbx2, bby1:bby2]
            
            # Adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_mixed.size()[-1] * batch_mixed.size()[-2]))
            
            # Get discriminator predictions (using decoder features for CutMix)
            if hasattr(self.net_d, 'forward_with_features'):
                mixed_e, mixed_d, mixed_enc, mixed_dec = self.net_d.forward_with_features(batch_mixed)
                # Use decoder's last feature for consistency loss
                mixed_pred = mixed_dec[-1] if mixed_dec else mixed_enc[-1]
            else:
                mixed_e, mixed_d = self.net_d(batch_mixed)
                # For discriminators without features, use decoder output
                mixed_pred = mixed_d
            
            # Create target by mixing real and fake predictions
            with torch.no_grad():
                if hasattr(self.net_d, 'forward_with_features'):
                    real_e, real_d, real_enc, real_dec = self.net_d.forward_with_features(real_batch)
                    fake_e, fake_d, fake_enc, fake_dec = self.net_d.forward_with_features(fake_batch)
                    # Use decoder's last feature for consistency target
                    real_pred = real_dec[-1] if real_dec else real_enc[-1]
                    fake_pred = fake_dec[-1] if fake_dec else fake_enc[-1]
                else:
                    real_e, real_d = self.net_d(real_batch)
                    fake_e, fake_d = self.net_d(fake_batch)
                    # Use decoder outputs for consistency
                    real_pred = real_d
                    fake_pred = fake_d
                
                # Mix the predictions according to the cut region
                target_pred = fake_pred.clone()
                target_pred[:, :, bbx1:bbx2, bby1:bby2] = real_pred[:, :, bbx1:bbx2, bby1:bby2]
            
            # Consistency loss
            consistency_loss = self.cri_consistency(mixed_pred, target_pred) * self.consistency_weight
            return consistency_loss
        
        return 0
    
    def resume_training(self, resume_state):
        """Resume training from a checkpoint, including discriminator if present."""
        # Call parent's resume_training to handle optimizers and schedulers
        super().resume_training(resume_state)
        
        # Load discriminator if we're using GAN training and a checkpoint exists
        if self.use_discriminator:
            # Get the iteration number from resume_state
            current_iter = resume_state.get('iter', 0)
            # Construct discriminator path based on the standard naming convention
            experiment_root = os.path.dirname(os.path.dirname(self.opt['path'].get('resume_state', '')))
            net_d_path = os.path.join(experiment_root, 'models', f'net_d_{current_iter}.pth')
            
            if os.path.exists(net_d_path):
                self.load_network(self.net_d, net_d_path, self.opt['path'].get('strict_load_d', True))
                logger = get_root_logger()
                logger.info(f"Resumed discriminator from {net_d_path}")
            else:
                logger = get_root_logger()
                logger.warning(f"Discriminator checkpoint not found at {net_d_path}, starting with random weights")
    
    def save(self, epoch, current_iter):
        """Override save to include discriminator"""
        super().save(epoch, current_iter)
        if self.use_discriminator:
            self.save_network(self.net_d, 'net_d', current_iter)
            logger = get_root_logger()
            logger.info(f"Saved discriminator checkpoint at iteration {current_iter}")


