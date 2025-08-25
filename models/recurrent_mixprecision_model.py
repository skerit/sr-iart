import torch
from collections import OrderedDict
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
    
    def feed_data(self, data):
        """Override feed_data to track current sample key."""
        # Store the key if available
        self.current_key = data.get('key', 'unknown')
        # Call parent feed_data - this calls SRModel.feed_data through the inheritance chain
        super(RecurrentMixPrecisionRTModel, self).feed_data(data)

    def optimize_parameters(self, scaler, current_iter):
        from basicsr.utils import get_root_logger
        logger = get_root_logger()
        
        # Special check at iteration 11700 to see model state before NaN
        if current_iter == 11700:
            logger.warning("=== Checking model state at iteration 11700 (before NaN) ===")
            # Log current learning rate
            current_lr = self.optimizer_g.param_groups[0]['lr']
            logger.warning(f"  Current learning rate: {current_lr:.2e}")
            
            # Check for any extreme weights
            max_weight = 0
            max_weight_name = ""
            for name, param in self.net_g.named_parameters():
                if param.numel() > 0:
                    param_max = param.abs().max().item()
                    param_mean = param.abs().mean().item()
                    if param_max > max_weight:
                        max_weight = param_max
                        max_weight_name = name
                    if param_max > 100 or param_mean > 10:
                        logger.warning(f"  Large weights in {name}: max={param_max:.2f}, mean={param_mean:.4f}")
            logger.warning(f"  Maximum weight value: {max_weight:.4f} in {max_weight_name}")
            logger.warning("=== End of iteration 11700 check ===")

        if self.fix_flow_iter:
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
            l_total = 0
            loss_dict = OrderedDict()
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
                
                l_percep, l_style = self.cri_perceptual(output_4d, gt_4d)
                if l_percep is not None:
                    l_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_total += l_style
                    loss_dict['l_style'] = l_style
        
        # Log loss values periodically to track trends
        if current_iter % 100 == 0:
            logger.info(f"Loss at iteration {current_iter}: total={l_total.item():.6f}, pix={l_pix.item() if self.cri_pix else 0:.6f}, percep={l_percep.item() if self.cri_perceptual and l_percep is not None else 0:.6f}")
        
        # Check for NaN in loss before backward (outside autocast to save memory)
        if torch.isnan(l_total) or torch.isinf(l_total):
            logger.error(f"NaN/Inf loss detected at iteration {current_iter}!")
            logger.error(f"  Sample key: {self.current_key}")
            logger.error(f"  l_total: {l_total.item()}")
            if self.cri_pix:
                logger.error(f"  l_pix: {l_pix.item() if not torch.isnan(l_pix) else 'NaN'}")
            if self.cri_perceptual:
                logger.error(f"  l_percep: {l_percep.item() if l_percep is not None and not torch.isnan(l_percep) else 'NaN/None'}")
            # Log input/output statistics
            logger.error(f"  LQ min/max: {self.lq.min().item():.4f}/{self.lq.max().item():.4f}")
            logger.error(f"  GT min/max: {self.gt.min().item():.4f}/{self.gt.max().item():.4f}")
            logger.error(f"  Output min/max: {self.output.min().item() if not torch.isnan(self.output).any() else 'nan'}/{self.output.max().item() if not torch.isnan(self.output).any() else 'nan'}")
            
            # Check for NaN in model weights - check ALL to find the problem
            nan_weights = []
            nan_grads = []
            
            for name, param in self.net_g.named_parameters():
                if param.numel() > 0:
                    # Check weights
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        nan_weights.append(name)
                        # Only log first 3 to avoid spam
                        if len(nan_weights) <= 3:
                            logger.error(f"  NaN/Inf in weight: {name}")
                    
                    # Check gradients if they exist
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        nan_grads.append(name)
                        if len(nan_grads) <= 3:
                            logger.error(f"  NaN/Inf in gradient: {name}")
            
            if nan_weights:
                logger.error(f"  Total NaN weights: {len(nan_weights)} parameters")
            if nan_grads:
                logger.error(f"  Total NaN gradients: {len(nan_grads)} parameters")
            
            # Clean up to prevent OOM on next iteration
            self.optimizer_g.zero_grad()
            
            # Clear the output tensor to free memory
            if hasattr(self, 'output'):
                del self.output
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Track consecutive NaN iterations
            self.nan_count += 1
            
            # If too many consecutive NaNs, reset optimizer state
            if self.nan_count >= 3:  # More aggressive - reset after 3 failures
                logger.error(f"  Too many consecutive NaN iterations ({self.nan_count}). Resetting optimizer state.")
                # Reset optimizer state (momentum, etc.)
                self.optimizer_g.state = {}
                self.nan_count = 0
                logger.error(f"  Optimizer state cleared. This should help stabilize training.")
            
            # Skip this iteration - don't backward or step
            logger.warning(f"  Skipping iteration {current_iter} due to NaN/Inf (count: {self.nan_count})")
            return
        
        scaler.scale(l_total).backward()
        
        # Unscale gradients for checking
        scaler.unscale_(self.optimizer_g)
        
        # Gradient clipping to prevent NaN
        if self.opt['train'].get('use_grad_clip', False):
            # Clip gradients - default to 0.5 if not specified
            max_norm = self.opt['train'].get('grad_clip_norm', 0.5)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=max_norm)
            
            # Log gradient norm periodically and when it's large
            if current_iter % 10 == 0 or grad_norm > 10 or torch.isinf(grad_norm):
                if torch.isinf(grad_norm):
                    logger.info(f"Gradient norm at iteration {current_iter}: inf")
                elif torch.isnan(grad_norm):
                    logger.info(f"Gradient norm at iteration {current_iter}: nan")
                else:
                    logger.info(f"Gradient norm at iteration {current_iter}: {grad_norm:.4f}")
            
            if grad_norm > 100:
                logger.warning(f"Large gradient norm at iteration {current_iter}: {grad_norm:.2f}")
            elif grad_norm > max_norm:
                logger.debug(f"Gradients clipped at iteration {current_iter}: {grad_norm:.2f} -> {max_norm}")
        else:
            # Just compute norm for logging
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), float('inf'))
            if grad_norm > 100:
                logger.warning(f"Large gradient norm at iteration {current_iter}: {grad_norm:.2f}")
        
        # Check for NaN in gradients after clipping
        has_nan_grad = False
        for p in self.net_g.parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                has_nan_grad = True
                break
        
        if has_nan_grad:
            logger.error(f"NaN/Inf in gradients after clipping at iteration {current_iter}! Skipping optimizer step.")
            # Track this as a failed iteration
            self.nan_count += 1
            
            # If too many consecutive NaNs, reset optimizer state
            if self.nan_count >= 3:  # More aggressive - reset after 3 failures
                logger.error(f"  Too many consecutive failed iterations ({self.nan_count}). Resetting optimizer state.")
                # Reset optimizer state (momentum, etc.)
                self.optimizer_g.state = {}
                self.nan_count = 0
                logger.error(f"  Optimizer state cleared. This should help stabilize training.")
                
            # Reset gradients and skip this step
            self.optimizer_g.zero_grad()
            scaler.update()
            return
        
        scaler.step(self.optimizer_g)
        scaler.update()
        
        # Reset NaN counter on successful iteration
        self.nan_count = 0
        
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


