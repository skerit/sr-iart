import torch
import math
import os
import cv2
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
                logger.info(f"LPIPS Loss initialized with weight {lpips_opt.get('loss_weight', 1.0)} "
                           f"using {lpips_opt.get('net_type', 'alex')} network")
            else:
                self.cri_lpips = None
    
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

        # update the gradient when forward 4 times
        self.optimizer_g.zero_grad()

        # Disable autocast for float32 training (more stable but uses 2x memory)
        use_mixed_precision = self.opt['train'].get('half_precision', True)
        
        with autocast(enabled=use_mixed_precision):
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
                
                # Skip ConvNeXt loss for very dark patches to avoid instability
                output_mean = output_4d.mean()
                gt_mean = gt_4d.mean()
                if output_mean > 0.02 and gt_mean > 0.02:  # Only if not too dark
                    l_convnext = self.cri_convnext(output_4d, gt_4d)
                else:
                    l_convnext = torch.tensor(0.0, device=self.device)
                    logger = get_root_logger()
                    logger.info(f"Skipped ConvNeXt loss for dark patch (output mean: {output_mean:.4f}, gt mean: {gt_mean:.4f})")
                
                l_total += l_convnext
                loss_dict['l_convnext'] = l_convnext
            
            # LPIPS perceptual loss - human-calibrated
            if hasattr(self, 'cri_lpips') and self.cri_lpips:
                # Reshape 5D video tensors to 4D for LPIPS loss
                if self.output.dim() == 5:
                    b, t, c, h, w = self.output.shape
                    output_4d = self.output.view(b * t, c, h, w)
                    gt_4d = self.gt.view(b * t, c, h, w)
                else:
                    output_4d = self.output
                    gt_4d = self.gt
                
                # Skip LPIPS loss for very dark patches to avoid instability
                output_mean = output_4d.mean()
                gt_mean = gt_4d.mean()
                if output_mean > 0.02 and gt_mean > 0.02:  # Only if not too dark
                    l_lpips = self.cri_lpips(output_4d, gt_4d)
                else:
                    l_lpips = torch.tensor(0.0, device=self.device)
                    logger = get_root_logger()
                    logger.info(f"Skipped LPIPS loss for dark patch (output mean: {output_mean:.4f}, gt mean: {gt_mean:.4f})")
                
                l_total += l_lpips
                loss_dict['l_lpips'] = l_lpips
            
            scaler.scale(l_total).backward()
            
            # Always unscale before gradient operations
            scaler.unscale_(self.optimizer_g)
            
            # Check gradient norm BEFORE clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=float('inf'))
            
            # Log gradient norm and learning rate periodically
            if self.current_iter % 50 == 0:
                logger = get_root_logger()
                current_lr = self.optimizer_g.param_groups[0]['lr']
                logger.info(f"Iter {self.current_iter}: grad_norm={grad_norm:.4f}, lr={current_lr:.2e}")
            
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
            
            # Gradient clipping to prevent NaN
            if self.opt['train'].get('use_grad_clip', False):
                max_norm = self.opt['train'].get('grad_clip_norm', 0.5)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=max_norm)
            
            scaler.step(self.optimizer_g)
            scaler.update()

            # l_total.backward()
            # self.optimizer_g.step()

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
                
                # Process output
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
                    else:
                        # Save individual images
                        save_folder = osp.join(self.opt['path']['visualization'], 
                                             f'val_images_iter_{current_iter:06d}')
                        os.makedirs(save_folder, exist_ok=True)
                        
                        # Save LQ, output, GT
                        imwrite(lq_img, osp.join(save_folder, f'{folder}_lq.png'))
                        imwrite(result_img, osp.join(save_folder, f'{folder}_output.png'))
                        imwrite(gt_img, osp.join(save_folder, f'{folder}_gt.png'))
                
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
        
        # Final cleanup of GPU memory after validation
        if hasattr(self, 'lq'):
            del self.lq
        if hasattr(self, 'gt'):
            del self.gt
        if hasattr(self, 'output'):
            del self.output
        torch.cuda.empty_cache()


