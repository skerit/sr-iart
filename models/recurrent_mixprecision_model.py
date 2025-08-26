import torch
import math
import os
from collections import OrderedDict
import torchvision
from torch.cuda.amp import autocast
from torch.nn.parallel import DataParallel, DistributedDataParallel

from basicsr.archs import build_network
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
                    reduction=fft_opt.get('reduction', 'mean')
                ).to(self.device)
                logger.info(f"FFT Loss initialized with weight {fft_opt.get('loss_weight', 1.0)}")
            else:
                self.cri_fft = None
    
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

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])

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

    def optimize_parameters(self, scaler, current_iter):

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

        # update the gradient when forward 4 times
        self.optimizer_g.zero_grad()

        with autocast():
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
                self.optimizer_g.zero_grad()
                return
            
            # Clamp output to valid range to prevent pure black
            self.output = torch.clamp(self.output, min=0.0, max=1.0)
            l_total = 0
            loss_dict = OrderedDict()
            # Check if model is outputting black/invalid/NaN frames
            if torch.isnan(self.output).any():
                logger = get_root_logger()
                logger.warning(f"Model outputting NaN after clamp! Skipping iteration.")
                # Clean up and skip
                del self.output
                torch.cuda.empty_cache()
                self.optimizer_g.zero_grad()
                return
            
            output_mean = self.output.mean().item()
            if output_mean < 0.001:
                logger = get_root_logger()
                logger.warning(f"Model outputting near-black frames! Mean: {output_mean:.6f}")
                # Clean up and skip
                del self.output
                torch.cuda.empty_cache()
                self.optimizer_g.zero_grad()
                return
            
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
                
                # Skip perceptual loss for very dark patches to avoid NaN
                output_mean = output_4d.mean()
                gt_mean = gt_4d.mean()
                if output_mean > 0.02 and gt_mean > 0.02:  # Only if not too dark
                    l_percep, l_style = self.cri_perceptual(output_4d, gt_4d)
                else:
                    l_percep = torch.tensor(0.0, device=self.device)
                    l_style = torch.tensor(0.0, device=self.device)
                    logger = get_root_logger()
                    logger.info(f"Skipped perceptual loss for dark patch (output mean: {output_mean:.4f}, gt mean: {gt_mean:.4f})")
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
            scaler.scale(l_total).backward()
            
            # Always unscale before gradient operations
            scaler.unscale_(self.optimizer_g)
            
            # Check gradient norm BEFORE clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=float('inf'))
            
            # Skip update if gradients are bad
            if torch.isnan(grad_norm) or grad_norm > 100:
                logger = get_root_logger()
                logger.warning(f"Skipping update - bad gradients detected! Norm: {grad_norm:.2f}")
                self.optimizer_g.zero_grad()
                scaler.update()  # Still need to update scaler
                return
            
            # Gradient clipping to prevent NaN
            if self.opt['train'].get('use_grad_clip', False):
                max_norm = self.opt['train'].get('grad_clip_norm', 0.5)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=max_norm)
            
            scaler.step(self.optimizer_g)
            scaler.update()

            # l_total.backward()
            # self.optimizer_g.step()

            self.log_dict = self.reduce_loss_dict(loss_dict)
            
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
                            # Take middle frame for saving
                            mid_frame = self.lq.shape[1] // 2
                            lq_save = self.lq[:, mid_frame, :, :, :]
                            gt_save = self.gt[:, mid_frame, :, :, :]
                            output_save = self.output[:, mid_frame, :, :, :]
                        else:
                            lq_save = self.lq
                            gt_save = self.gt
                            output_save = self.output
                        
                        # Save LQ, GT, and output
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


