"""Simple discriminator training using pre-generated dataset."""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import wandb

from archs.concat_discriminator import ConcatDiscriminator
from archs.patch_discriminator import PatchSharpnessDiscriminator, MultiScalePatchDiscriminator, LocalPatchDiscriminator
from archs.ciplab_discriminator_arch import CIPLABUnetD


class PreGeneratedDataset(Dataset):
    """Dataset that loads pre-generated image pairs."""
    
    def __init__(self, dataset_json, dataset_dir, patch_size=128, 
                 h_flip=False, v_flip=False, rotation=False):
        """Initialize with pre-generated dataset.
        
        Args:
            dataset_json: Path to dataset JSON file
            dataset_dir: Directory containing dataset images  
            patch_size: Size of patches to extract
            h_flip: Enable random horizontal flipping
            v_flip: Enable random vertical flipping
            rotation: Enable random 90-degree rotations
        """
        self.dataset_dir = Path(dataset_dir)
        self.patch_size = patch_size
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.rotation = rotation
        
        # Load dataset pairs
        with open(dataset_json, 'r') as f:
            self.pairs = json.load(f)
        
        print(f"Loaded {len(self.pairs)} training pairs")
        if any([h_flip, v_flip, rotation]):
            augmentations = []
            if h_flip: augmentations.append("h_flip")
            if v_flip: augmentations.append("v_flip")
            if rotation: augmentations.append("rotation")
            print(f"Augmentations enabled: {', '.join(augmentations)}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Load a pre-generated pair."""
        pair = self.pairs[idx]
        
        # Load images
        gen_path = self.dataset_dir / pair['generated']
        gt_path = self.dataset_dir / pair['gt']
        
        gen_img = cv2.imread(str(gen_path))
        if gen_img is None or gen_img.size == 0:
            print(f"Warning: Could not load or empty image {gen_path}")
            # Return a random valid pair instead
            return self.__getitem__(np.random.randint(0, len(self)))
        
        gt_img = cv2.imread(str(gt_path))
        if gt_img is None or gt_img.size == 0:
            print(f"Warning: Could not load or empty image {gt_path}")
            return self.__getitem__(np.random.randint(0, len(self)))
        
        # Check image dimensions
        if gen_img.shape[0] == 0 or gen_img.shape[1] == 0:
            print(f"Warning: Zero dimension in generated image {gen_path}: shape={gen_img.shape}")
            return self.__getitem__(np.random.randint(0, len(self)))
        
        if gt_img.shape[0] == 0 or gt_img.shape[1] == 0:
            print(f"Warning: Zero dimension in GT image {gt_path}: shape={gt_img.shape}")
            return self.__getitem__(np.random.randint(0, len(self)))
        
        # Convert to RGB
        gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        
        # Random crop to patch size
        h, w = gen_img.shape[:2]
        if h >= self.patch_size and w >= self.patch_size:
            # Random crop
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            gen_patch = gen_img[top:top+self.patch_size, left:left+self.patch_size]
            gt_patch = gt_img[top:top+self.patch_size, left:left+self.patch_size]
        else:
            # Resize if too small
            try:
                gen_patch = cv2.resize(gen_img, (self.patch_size, self.patch_size))
                gt_patch = cv2.resize(gt_img, (self.patch_size, self.patch_size))
            except Exception as e:
                print(f"Warning: Failed to resize images - gen_img shape: {gen_img.shape}, gt_img shape: {gt_img.shape}")
                print(f"Error: {e}")
                return self.__getitem__(np.random.randint(0, len(self)))
        
        # Ensure patches have the correct shape
        if gen_patch.shape != (self.patch_size, self.patch_size, 3):
            print(f"Warning: Invalid gen_patch shape after crop/resize: {gen_patch.shape}, expected ({self.patch_size}, {self.patch_size}, 3)")
            print(f"  Original gen_img shape: {gen_img.shape}, Image path: {gen_path}")
            return self.__getitem__(np.random.randint(0, len(self)))
        
        if gt_patch.shape != (self.patch_size, self.patch_size, 3):
            print(f"Warning: Invalid gt_patch shape after crop/resize: {gt_patch.shape}, expected ({self.patch_size}, {self.patch_size}, 3)")
            print(f"  Original gt_img shape: {gt_img.shape}, Image path: {gt_path}")
            return self.__getitem__(np.random.randint(0, len(self)))
        
        # Apply augmentations (same transform to both images to maintain correspondence)
        if self.h_flip and np.random.random() > 0.5:
            gen_patch = np.fliplr(gen_patch).copy()
            gt_patch = np.fliplr(gt_patch).copy()
        
        if self.v_flip and np.random.random() > 0.5:
            gen_patch = np.flipud(gen_patch).copy()
            gt_patch = np.flipud(gt_patch).copy()
        
        if self.rotation:
            # Random 90-degree rotation (0, 90, 180, or 270 degrees)
            k = np.random.randint(0, 4)
            if k > 0:
                gen_patch = np.rot90(gen_patch, k).copy()
                gt_patch = np.rot90(gt_patch, k).copy()
        
        # Convert to tensors and normalize
        gen_tensor = torch.from_numpy(gen_patch.transpose(2, 0, 1).copy()).float() / 255.0
        gt_tensor = torch.from_numpy(gt_patch.transpose(2, 0, 1).copy()).float() / 255.0
        
        # Create target_score tensor in a way that's compatible with DataLoader collation
        # Using torch.as_tensor instead of torch.tensor to avoid storage resizing issues
        target_score = torch.as_tensor([pair['loss']], dtype=torch.float32)
        
        return {
            'generated': gen_tensor,
            'gt': gt_tensor,
            'target_score': target_score
        }


def detect_architecture_from_checkpoint(checkpoint_path):
    """Detect discriminator architecture from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get state dict - BasicSR uses 'params' key
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Detect architecture based on layer names
    keys = list(state_dict.keys())
    
    if 'enc_b1.conv1.weight' in keys or 'enc_b1.0.weight' in keys:
        # CIPLAB discriminator
        return 'ciplab', {'base_channels': 64}  # CIPLAB has fixed architecture
    elif 'initial.0.weight' in keys:
        # Concat discriminator
        base_channels = state_dict['initial.0.weight'].shape[0]
        return 'concat', {'base_channels': base_channels}
    elif 'conv_blocks.0.0.weight' in keys:
        # Patch discriminator
        base_channels = state_dict['conv_blocks.0.0.weight'].shape[0]
        return 'patch', {'base_channels': base_channels}
    elif 'scale_0.conv_blocks.0.0.weight' in keys:
        # MultiScale discriminator
        base_channels = state_dict['scale_0.conv_blocks.0.0.weight'].shape[0]
        return 'multiscale', {'base_channels': base_channels}
    elif 'conv1.weight' in keys:
        # Local discriminator
        base_channels = state_dict['conv1.weight'].shape[0]
        return 'local', {'base_channels': base_channels}
    else:
        print(f"Unable to detect architecture from checkpoint {checkpoint_path}")
        print(f"Found keys: {keys[:5]}...")
        raise ValueError(f"Unknown architecture in checkpoint {checkpoint_path}")


def train_discriminator(args):
    """Train the discriminator."""
    
    # Auto-detect architecture if resuming
    if args.resume and os.path.exists(args.resume):
        detected_arch, arch_params = detect_architecture_from_checkpoint(args.resume)
        if args.arch is None:
            print(f"Auto-detected architecture: {detected_arch} with params {arch_params}")
            args.arch = detected_arch
            # Update architecture-specific parameters
            for key, value in arch_params.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)
        else:
            print(f"Using specified architecture: {args.arch} (detected: {detected_arch})")
    elif args.arch is None:
        print("Error: --arch must be specified when not resuming from checkpoint")
        sys.exit(1)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"disc_lr{args.lr}_bs{args.batch_size}",
            config={
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "base_channels": args.base_channels,
                "num_layers": args.num_layers,
                "patch_size": args.patch_size,
            },
            resume="allow"  # Allow resuming wandb runs
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model based on architecture choice
    if args.arch == 'concat':
        model = ConcatDiscriminator(
            in_channels=6,
            base_channels=args.base_channels,
            num_layers=args.num_layers
        ).to(device)
    elif args.arch == 'patch':
        model = PatchSharpnessDiscriminator(
            in_channels=6,
            base_channels=args.base_channels,
            num_layers=args.num_layers
        ).to(device)
    elif args.arch == 'multiscale':
        model = MultiScalePatchDiscriminator(
            base_channels=args.base_channels
        ).to(device)
    elif args.arch == 'local':
        model = LocalPatchDiscriminator(
            receptive_field=args.local_patch_size,  # Using patch_size as receptive field
            in_channels=6,
            base_channels=args.base_channels
        ).to(device)
    elif args.arch == 'ciplab':
        model = CIPLABUnetD().to(device)  # CIPLAB has fixed architecture, no parameters
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset
    dataset = PreGeneratedDataset(
        args.dataset_json,
        args.dataset_dir,
        patch_size=args.patch_size,
        h_flip=args.h_flip,
        v_flip=args.v_flip,
        rotation=args.rotation
    )
    
    # Split into train/val
    if not args.no_val:
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        print(f"Training samples: {train_size}, Validation samples: {val_size}")
    else:
        train_dataset = dataset
        train_size = len(dataset)
        val_dataset = None
        print(f"Training samples: {train_size}, Validation disabled")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    if not args.no_val:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    # Only create scheduler if validation is enabled
    if not args.no_val:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    else:
        scheduler = None
    
    # Use MSELoss for regression task (predicting continuous quality scores)
    # We'll apply sigmoid to convert logits to 0-1 range, then use MSE
    criterion = nn.MSELoss()
    
    # Resume from checkpoint if provided
    start_epoch = 0
    start_iteration = 0
    best_val_loss = float('inf')
    global_iteration = 0
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Load model state - handle different checkpoint formats
        if 'params' in checkpoint:
            # BasicSR format (from GAN training)
            model.load_state_dict(checkpoint['params'])
            print("Loaded model from BasicSR checkpoint (params)")
        elif 'model_state_dict' in checkpoint:
            # Our discriminator training format
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model from discriminator training checkpoint")
        elif 'state_dict' in checkpoint:
            # Alternative format
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded model from state_dict")
        else:
            # Assume checkpoint is the state dict itself
            model.load_state_dict(checkpoint)
            print("Loaded model directly from checkpoint")
        
        # Try to load optimizer if available (may not exist in BasicSR checkpoints)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state")
        
        # Restore training state
        if 'iteration' in checkpoint:
            start_iteration = checkpoint['iteration']
            global_iteration = start_iteration
            print(f"Resuming from iteration {start_iteration}")
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
            print(f"Resuming from epoch {start_epoch}")
        
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        elif 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
        elif 'train_loss' in checkpoint:
            best_val_loss = checkpoint['train_loss']
        
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Resumed with best_val_loss: {best_val_loss:.4f}")
    
    # Training loop
    avg_train_loss = 0.0
    avg_val_loss = float('inf') if args.no_val else 0.0  # Initialize to prevent NameError

    for epoch in range(start_epoch, args.epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_loss_lq = 0  # Track LQ loss for CIPLAB
        train_loss_gt = 0  # Track GT loss for CIPLAB
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]") as pbar:
            for batch_idx, batch in enumerate(pbar):
                generated = batch['generated'].to(device)
                gt = batch['gt'].to(device)
                target_scores = batch['target_score'].to(device)
            
                # Forward pass
                if args.arch == 'ciplab':
                    # CIPLAB expects single images, not concatenated
                    # Process generated and GT separately
                    gen_e, gen_d = model(generated)
                    gt_e, gt_d = model(gt)
                    # Reduce spatial dimensions to get single scores
                    # Encoder output is small (2x2), decoder is full resolution (128x128)
                    gen_e_score = gen_e.mean(dim=[2, 3], keepdim=True)  # Average over spatial dims
                    gen_d_score = gen_d.mean(dim=[2, 3], keepdim=True)
                    gt_e_score = gt_e.mean(dim=[2, 3], keepdim=True)
                    gt_d_score = gt_d.mean(dim=[2, 3], keepdim=True)
                    # Average the encoder and decoder outputs
                    predicted_logits = (gen_e_score + gen_d_score) / 2
                    gt_logits = (gt_e_score + gt_d_score) / 2
                    # For training, we want generated to score low and GT to score high
                    # Combine both objectives
                    predicted_scores = torch.sigmoid(predicted_logits)
                    gt_scores = torch.sigmoid(gt_logits)
                    # Calculate losses separately for tracking
                    loss_lq = criterion(predicted_scores.squeeze(), target_scores.squeeze()) * 50
                    loss_gt = criterion(gt_scores.squeeze(), torch.ones_like(target_scores).squeeze()) * 50
                    loss = loss_lq + loss_gt
                else:
                    predicted_logits = model(generated, gt)
                    # Apply sigmoid to get probabilities for regression
                    predicted_scores = torch.sigmoid(predicted_logits)
                    # Scale loss by 100 to get more reasonable gradients
                    loss = criterion(predicted_scores, target_scores) * 100
                
                # Log worst predictions occasionally
                if batch_idx % 100 == 0 and args.arch != 'ciplab':
                    with torch.no_grad():
                        # predicted_scores already contains probabilities (after sigmoid)
                        errors = torch.abs(predicted_scores - target_scores)
                    max_error_idx = torch.argmax(errors)
                    worst_pred = predicted_scores[max_error_idx].item()
                    worst_target = target_scores[max_error_idx].item()
                    worst_error = errors[max_error_idx].item()
                    if worst_error > 0.3:
                        print(f"\n  WARNING: Large error! Predicted: {worst_pred:.3f}, Target: {worst_target:.3f}, Error: {worst_error:.3f}")
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability with high learning rate
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                if args.arch == 'ciplab':
                    train_loss_lq += loss_lq.item()
                    train_loss_gt += loss_gt.item()
                global_iteration += 1
                if args.arch == 'ciplab':
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", 
                                    'loss_lq': f"{loss_lq.item():.4f}",
                                    'loss_gt': f"{loss_gt.item():.4f}",
                                    'iter': global_iteration})
                else:
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", 'iter': global_iteration})
                
                # Log to wandb
                if args.use_wandb:
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/epoch': epoch + 1,
                        'iteration': global_iteration,
                    }
                    # Add CIPLAB-specific losses
                    if args.arch == 'ciplab':
                        log_dict.update({
                            'train/loss_lq': loss_lq.item(),
                            'train/loss_gt': loss_gt.item(),
                        })
                    # Only add worst prediction info if not CIPLAB and on logging interval
                    elif batch_idx % 100 == 0:
                        log_dict.update({
                            'train/worst_pred': worst_pred,
                            'train/worst_target': worst_target,
                            'train/worst_error': worst_error,
                        })
                    wandb.log(log_dict)
                
                # Save checkpoint every N iterations
                if global_iteration % args.save_iter == 0:
                    iter_checkpoint = {
                        'iteration': global_iteration,
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'best_val_loss': best_val_loss
                    }
                    if scheduler is not None:
                        iter_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                    iter_save_path = f"{args.save_dir}/discriminator_iter_{global_iteration}.pth"
                    os.makedirs(args.save_dir, exist_ok=True)
                    torch.save(iter_checkpoint, iter_save_path)
                    print(f"\nSaved iteration checkpoint at iteration {global_iteration}")
        
        avg_train_loss = train_loss / len(train_loader)
        if args.arch == 'ciplab':
            avg_train_loss_lq = train_loss_lq / len(train_loader)
            avg_train_loss_gt = train_loss_gt / len(train_loader)
        
        if not args.no_val:
            # Validation phase
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                with tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]") as pbar:
                    for batch in pbar:
                        generated = batch['generated'].to(device)
                        gt = batch['gt'].to(device)
                        target_scores = batch['target_score'].to(device)
                        
                        if args.arch == 'ciplab':
                            # CIPLAB validation
                            gen_e, gen_d = model(generated)
                            gt_e, gt_d = model(gt)
                            # Reduce spatial dimensions
                            gen_e_score = gen_e.mean(dim=[2, 3], keepdim=True)
                            gen_d_score = gen_d.mean(dim=[2, 3], keepdim=True)
                            gt_e_score = gt_e.mean(dim=[2, 3], keepdim=True)
                            gt_d_score = gt_d.mean(dim=[2, 3], keepdim=True)
                            predicted_logits = (gen_e_score + gen_d_score) / 2
                            gt_logits = (gt_e_score + gt_d_score) / 2
                            predicted_scores = torch.sigmoid(predicted_logits)
                            gt_scores = torch.sigmoid(gt_logits)
                            loss = (criterion(predicted_scores.squeeze(), target_scores.squeeze()) + 
                                   criterion(gt_scores.squeeze(), torch.ones_like(target_scores).squeeze())) * 50
                        else:
                            predicted_logits = model(generated, gt)
                            # Apply sigmoid to get probabilities for regression
                            predicted_scores = torch.sigmoid(predicted_logits)
                            # Scale loss by 100 to match training
                            loss = criterion(predicted_scores, target_scores) * 100
                        
                        val_loss += loss.item()
                        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Log to wandb
            if args.use_wandb:
                wandb.log({
                    'train/loss_epoch': avg_train_loss,
                    'val/loss': avg_val_loss,
                    'epoch': epoch + 1,
                })
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                }
                save_path = f"{args.save_dir}/discriminator_best.pth"
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(checkpoint, save_path)
                print(f"Saved best model with val_loss: {avg_val_loss:.4f}")
                
                if args.use_wandb:
                    wandb.log({'val/best_loss': avg_val_loss})
        else:
            # No validation - just print train loss
            if args.arch == 'ciplab':
                print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} (LQ: {avg_train_loss_lq:.4f}, GT: {avg_train_loss_gt:.4f})")
            else:
                print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
            
            # Log to wandb
            if args.use_wandb:
                log_dict = {
                    'train/loss_epoch': avg_train_loss,
                    'epoch': epoch + 1,
                }
                if args.arch == 'ciplab':
                    log_dict.update({
                        'train/loss_lq_epoch': avg_train_loss_lq,
                        'train/loss_gt_epoch': avg_train_loss_gt,
                    })
                wandb.log(log_dict)
            
            # Save best model based on train loss
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss
                }
                save_path = f"{args.save_dir}/discriminator_best.pth"
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(checkpoint, save_path)
                print(f"Saved best model with train_loss: {avg_train_loss:.4f}")
                
                if args.use_wandb:
                    wandb.log({'train/best_loss': avg_train_loss})
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
            save_path = f"{args.save_dir}/discriminator_epoch_{epoch+1}.pth"
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss
    }
    final_path = f"{args.save_dir}/discriminator_final.pth"
    torch.save(final_checkpoint, final_path)
    print(f"Training complete! Final model saved to {final_path}")
    
    # Log final metrics and finish wandb
    if args.use_wandb:
        wandb.log({
            'final/train_loss': avg_train_loss,
            'final/val_loss': avg_val_loss if not args.no_val else None,
            'final/best_val_loss': best_val_loss,
        })
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train discriminator on pre-generated dataset')
    
    # Dataset arguments
    parser.add_argument('dataset_json', type=str, help='Path to dataset JSON file')
    parser.add_argument('dataset_dir', type=str, help='Directory containing dataset images')
    
    # Model arguments
    parser.add_argument('--arch', type=str, default=None,
                        choices=['concat', 'patch', 'multiscale', 'local', 'ciplab'],
                        help='Discriminator architecture (auto-detected from checkpoint if resuming)')
    parser.add_argument('--base_channels', type=int, default=64, 
                        help='Base number of channels')
    parser.add_argument('--num_layers', type=int, default=4, 
                        help='Number of discriminator layers')
    parser.add_argument('--local_patch_size', type=int, default=32,
                        help='Patch size for local discriminator')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, 
                        help='Validation split ratio')
    parser.add_argument('--no_val', action='store_true',
                        help='Disable validation (for ROCm compatibility)')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    # Data augmentation arguments
    parser.add_argument('--h_flip', action='store_true',
                        help='Enable random horizontal flipping during training')
    parser.add_argument('--v_flip', action='store_true',
                        help='Enable random vertical flipping during training')
    parser.add_argument('--rotation', action='store_true',
                        help='Enable random 90-degree rotations during training')
    
    # Save arguments
    parser.add_argument('--save_dir', type=str, default='experiments/discriminator',
                        help='Directory to save models')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--save_iter', type=int, default=100,
                        help='Save checkpoint every N iterations')
    
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='discriminator-training',
                        help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Wandb run name (auto-generated if not provided)')
    
    args = parser.parse_args()
    train_discriminator(args)


if __name__ == "__main__":
    main()