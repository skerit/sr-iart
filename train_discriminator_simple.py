"""Simple discriminator training using pre-generated dataset."""

import os
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


class PreGeneratedDataset(Dataset):
    """Dataset that loads pre-generated image pairs."""
    
    def __init__(self, dataset_json, dataset_dir, patch_size=128):
        """Initialize with pre-generated dataset."""
        self.dataset_dir = Path(dataset_dir)
        self.patch_size = patch_size
        
        # Load dataset pairs
        with open(dataset_json, 'r') as f:
            self.pairs = json.load(f)
        
        print(f"Loaded {len(self.pairs)} training pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Load a pre-generated pair."""
        pair = self.pairs[idx]
        
        # Load images
        gen_path = self.dataset_dir / pair['generated']
        gt_path = self.dataset_dir / pair['gt']
        
        gen_img = cv2.imread(str(gen_path))
        if gen_img is None:
            print(f"Warning: Could not load {gen_path}")
            # Return a random valid pair instead
            return self.__getitem__(np.random.randint(0, len(self)))
        
        gt_img = cv2.imread(str(gt_path))
        if gt_img is None:
            print(f"Warning: Could not load {gt_path}")
            return self.__getitem__(np.random.randint(0, len(self)))
        
        # Convert to RGB
        gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        
        # Random crop to patch size
        h, w = gen_img.shape[:2]
        if h > self.patch_size and w > self.patch_size:
            # Random crop
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            gen_patch = gen_img[top:top+self.patch_size, left:left+self.patch_size]
            gt_patch = gt_img[top:top+self.patch_size, left:left+self.patch_size]
        else:
            # Resize if too small
            gen_patch = cv2.resize(gen_img, (self.patch_size, self.patch_size))
            gt_patch = cv2.resize(gt_img, (self.patch_size, self.patch_size))
        
        # Convert to tensors and normalize
        gen_tensor = torch.from_numpy(gen_patch.transpose(2, 0, 1)).float() / 255.0
        gt_tensor = torch.from_numpy(gt_patch.transpose(2, 0, 1)).float() / 255.0
        
        return {
            'generated': gen_tensor,
            'gt': gt_tensor,
            'target_score': torch.tensor([pair['loss']], dtype=torch.float32)
        }


def train_discriminator(args):
    """Train the discriminator."""
    
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
    
    # Create model
    model = ConcatDiscriminator(
        in_channels=6,
        base_channels=args.base_channels,
        num_layers=args.num_layers
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset
    dataset = PreGeneratedDataset(
        args.dataset_json,
        args.dataset_dir,
        patch_size=args.patch_size
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
    avg_val_loss = float('inf')  # Initialize to prevent NameError when no_val is True
    global_iteration = 0
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
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
    
    if args.seamless_epochs:
        # Create a single continuous iterator for all epochs
        total_iterations = len(train_loader) * args.epochs
        print(f"Training for {total_iterations} total iterations ({args.epochs} epochs)")
        
        epoch = start_epoch
        iteration_in_epoch = start_iteration % len(train_loader) if start_iteration > 0 else 0
        
        # Skip already completed iterations
        if start_iteration > 0:
            print(f"Skipping {start_iteration} iterations to resume training")
        
        with tqdm(total=total_iterations, initial=start_iteration, desc="Training") as pbar:
            for _ in range(args.epochs):
                for batch_idx, batch in enumerate(train_loader):
                    # Skip iterations if resuming
                    if global_iteration < start_iteration:
                        global_iteration += 1
                        continue
                    
                    generated = batch['generated'].to(device)
                    gt = batch['gt'].to(device)
                    target_scores = batch['target_score'].to(device)
                    
                    # Forward pass
                    predicted_logits = model(generated, gt)
                    # Apply sigmoid to get probabilities for regression
                    predicted_scores = torch.sigmoid(predicted_logits)
                    # Scale loss by 100 to get more reasonable gradients
                    loss = criterion(predicted_scores, target_scores) * 100
                    
                    # Log worst predictions occasionally
                    if global_iteration % 100 == 0:
                        with torch.no_grad():
                            # Convert logits to probabilities for meaningful comparison
                            predicted_probs = torch.sigmoid(predicted_scores)
                            errors = torch.abs(predicted_probs - target_scores)
                        max_error_idx = torch.argmax(errors)
                        worst_pred = predicted_probs[max_error_idx].item()
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
                    global_iteration += 1
                    iteration_in_epoch += 1
                    pbar.update(1)
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", 'epoch': epoch+1})
                    
                    # Log to wandb
                    if args.use_wandb:
                        wandb.log({
                            'train/loss': loss.item(),
                            'train/worst_pred': worst_pred if global_iteration % 100 == 0 else None,
                            'train/worst_target': worst_target if global_iteration % 100 == 0 else None,
                            'train/worst_error': worst_error if global_iteration % 100 == 0 else None,
                            'train/epoch': epoch + 1,
                            'iteration': global_iteration,
                        })
                    
                    # Save checkpoint every N iterations
                    if global_iteration % args.save_iter == 0:
                        iter_checkpoint = {
                            'iteration': global_iteration,
                            'epoch': epoch,
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
                    
                    # Check if epoch completed
                    if iteration_in_epoch >= len(train_loader):
                        epoch += 1
                        iteration_in_epoch = 0
                        
                        # Save epoch checkpoint
                        if epoch % args.save_freq == 0:
                            checkpoint = {
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss.item()
                            }
                            save_path = f"{args.save_dir}/discriminator_epoch_{epoch}.pth"
                            torch.save(checkpoint, save_path)
                            print(f"\nSaved epoch checkpoint at epoch {epoch}")
    else:
        # Original epoch-based training
        for epoch in range(start_epoch, args.epochs):
            # Training phase
            model.train()
            train_loss = 0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]") as pbar:
                for batch_idx, batch in enumerate(pbar):
                    generated = batch['generated'].to(device)
                    gt = batch['gt'].to(device)
                    target_scores = batch['target_score'].to(device)
                
                    # Forward pass
                    predicted_logits = model(generated, gt)
                    # Apply sigmoid to get probabilities for regression
                    predicted_scores = torch.sigmoid(predicted_logits)
                    # Scale loss by 100 to get more reasonable gradients
                    loss = criterion(predicted_scores, target_scores) * 100
                    
                    # Log worst predictions occasionally
                    if batch_idx % 100 == 0:
                        with torch.no_grad():
                            # Convert logits to probabilities for meaningful comparison
                            predicted_probs = torch.sigmoid(predicted_scores)
                            errors = torch.abs(predicted_probs - target_scores)
                        max_error_idx = torch.argmax(errors)
                        worst_pred = predicted_probs[max_error_idx].item()
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
                    global_iteration += 1
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", 'iter': global_iteration})
                    
                    # Log to wandb
                    if args.use_wandb:
                        wandb.log({
                            'train/loss': loss.item(),
                            'train/worst_pred': worst_pred if batch_idx % 100 == 0 else None,
                            'train/worst_target': worst_target if batch_idx % 100 == 0 else None,
                            'train/worst_error': worst_error if batch_idx % 100 == 0 else None,
                            'train/epoch': epoch + 1,
                            'iteration': global_iteration,
                        })
                    
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
                        
                        predicted_scores = model(generated, gt)
                        loss = criterion(predicted_scores, target_scores)
                        
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
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
            
            # Log to wandb
            if args.use_wandb:
                wandb.log({
                    'train/loss_epoch': avg_train_loss,
                    'epoch': epoch + 1,
                })
            
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
    parser.add_argument('--base_channels', type=int, default=128, 
                        help='Base number of channels')
    parser.add_argument('--num_layers', type=int, default=5, 
                        help='Number of discriminator layers')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, 
                        help='Validation split ratio')
    parser.add_argument('--no_val', action='store_true',
                        help='Disable validation (for ROCm compatibility)')
    parser.add_argument('--seamless_epochs', action='store_true',
                        help='No reset between epochs (for ROCm compatibility)')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
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