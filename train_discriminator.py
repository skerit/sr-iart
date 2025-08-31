"""Train Concatenated Discriminator for SR Quality Assessment"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import random
import argparse
from tqdm import tqdm
import json

from archs.concat_discriminator import ConcatDiscriminator


class DiscriminatorDataset(Dataset):
    """Dataset for training the discriminator."""
    
    def __init__(self, gt_dir, lq_dir, patch_size=128, num_samples=10000,
                 saved_pairs_config=None):
        self.gt_dir = Path(gt_dir)
        self.lq_dir = Path(lq_dir)
        self.patch_size = patch_size
        self.num_samples = num_samples
        
        # Get all image files for synthetic degradation
        self.gt_files = sorted(list(self.gt_dir.glob('**/*.png')) + 
                               list(self.gt_dir.glob('**/*.jpg')))
        self.lq_files = sorted(list(self.lq_dir.glob('**/*.png')) + 
                               list(self.lq_dir.glob('**/*.jpg')))
        
        # Load saved artifact triplets from JSON config
        self.saved_triplets = []
        self.upscale_penalties = {'bicubic': 0.10, 'bilinear': 0.05, 'nearest': 0.01}
        
        if saved_pairs_config:
            with open(saved_pairs_config, 'r') as f:
                config = json.load(f)
            
            # Get upscale penalties if specified
            self.upscale_penalties = config.get('upscale_penalties', self.upscale_penalties)
            
            base_dir = Path(saved_pairs_config).parent
            
            for pair_set in config.get('pairs', []):
                path = base_dir / pair_set['path']
                output_quality = pair_set['output_quality']
                use_input = pair_set.get('use_input', False)
                input_base_quality = pair_set.get('input_base_quality', 0.7)
                source_type = pair_set.get('source_type', 'unknown')
                
                # Find all triplets in this directory
                # Pattern: *_gt.png, *_output.png, *_lq.png
                gt_files = sorted(path.glob('*_gt.png'))
                
                for gt_file in gt_files:
                    base_name = str(gt_file).replace('_gt.png', '')
                    output_file = Path(base_name + '_output.png')
                    lq_file = Path(base_name + '_lq.png')
                    
                    if output_file.exists():
                        triplet = {
                            'gt': gt_file,
                            'output': output_file,
                            'output_quality': output_quality,
                            'use_input': use_input,
                            'input_base_quality': input_base_quality,
                            'source_type': source_type,
                            'lq': lq_file if lq_file.exists() else None
                        }
                        self.saved_triplets.append(triplet)
                
                print(f"Found {len(gt_files)} triplets in {pair_set['name']} ({source_type}) - "
                      f"output: {output_quality:.2f}, input base: {input_base_quality:.2f}")
            
            print(f"Total saved triplets: {len(self.saved_triplets)}")
        
        print(f"Found {len(self.gt_files)} GT files and {len(self.lq_files)} LQ files for synthetic data")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 40% chance to use saved triplets if available
        if self.saved_triplets and random.random() < 0.4:
            triplet = random.choice(self.saved_triplets)
            
            # Load GT image
            gt_img = cv2.imread(str(triplet['gt']))
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            
            # Decide what to use as generated image
            if triplet['use_input'] and triplet['lq'] and random.random() < 0.5:
                # 50% chance to use upscaled LQ input when use_input is True
                lq_img = cv2.imread(str(triplet['lq']))
                lq_img = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)
                
                # Upscale the LQ image with different methods
                h, w = gt_img.shape[:2]
                upscale_method = random.choice(['bicubic', 'bilinear', 'nearest'])
                
                # Calculate score: base_quality + penalty (higher = worse)
                base_quality = triplet['input_base_quality']
                penalty = self.upscale_penalties[upscale_method]
                target_score = min(base_quality + penalty, 1.0)  # Don't go above 1
                
                if upscale_method == 'bicubic':
                    gen_img = cv2.resize(lq_img, (w, h), interpolation=cv2.INTER_CUBIC)
                elif upscale_method == 'bilinear':
                    gen_img = cv2.resize(lq_img, (w, h), interpolation=cv2.INTER_LINEAR)
                else:  # nearest
                    gen_img = cv2.resize(lq_img, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                # Use the actual model output
                gen_img = cv2.imread(str(triplet['output']))
                gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
                target_score = triplet['output_quality']
            
            # These are already patches, just resize if needed
            if gen_img.shape[0] != self.patch_size:
                gen_img = cv2.resize(gen_img, (self.patch_size, self.patch_size))
                gt_img = cv2.resize(gt_img, (self.patch_size, self.patch_size))
            
            # Convert to tensors
            generated = torch.from_numpy(gen_img.transpose(2, 0, 1)).float() / 255.0
            gt_tensor = torch.from_numpy(gt_img.transpose(2, 0, 1)).float() / 255.0
            
            return {
                'generated': generated,
                'gt': gt_tensor,
                'target_score': torch.tensor([target_score], dtype=torch.float32)
            }
        
        # Otherwise, use regular synthetic degradations
        # Randomly select a file
        file_idx = random.randint(0, len(self.gt_files) - 1)
        
        # Load images
        gt_img = cv2.imread(str(self.gt_files[file_idx]))
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        
        lq_img = cv2.imread(str(self.lq_files[file_idx]))
        lq_img = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)
        
        # Upscale LQ to match GT size
        h, w = gt_img.shape[:2]
        lq_img = cv2.resize(lq_img, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Random crop
        h, w = gt_img.shape[:2]
        if h > self.patch_size and w > self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            gt_patch = gt_img[top:top+self.patch_size, left:left+self.patch_size]
            lq_patch = lq_img[top:top+self.patch_size, left:left+self.patch_size]
        else:
            gt_patch = cv2.resize(gt_img, (self.patch_size, self.patch_size))
            lq_patch = cv2.resize(lq_img, (self.patch_size, self.patch_size))
        
        # Convert to tensors and normalize to [0, 1]
        gt_tensor = torch.from_numpy(gt_patch.transpose(2, 0, 1)).float() / 255.0
        lq_tensor = torch.from_numpy(lq_patch.transpose(2, 0, 1)).float() / 255.0
        
        # Create different quality pairs with diverse degradations
        pair_type = random.choice([
            'perfect', 'good_sharp', 'color_shift', 'desaturated', 
            'oversaturated', 'brightness', 'real_vhs', 'bicubic', 'terrible'
        ])
        
        if pair_type == 'perfect':
            # Perfect match (GT, GT)
            generated = gt_tensor
            target = 0.0  # Perfect score
            
        elif pair_type == 'good_sharp':
            # Small sharpness degradation - add slight blur or noise
            generated = gt_tensor.clone()
            if random.random() > 0.5:
                # Add very slight Gaussian noise
                noise = torch.randn_like(generated) * 0.01
                generated = torch.clamp(generated + noise, 0, 1)
            else:
                # Slight blur
                generated = nn.functional.avg_pool2d(
                    generated.unsqueeze(0), kernel_size=3, stride=1, padding=1
                ).squeeze(0) * 0.7 + generated * 0.3
            target = 0.2  # Good score
            
        elif pair_type == 'color_shift':
            # Hue shift - rotate colors
            generated = gt_tensor.clone()
            # Simple hue shift by rotating RGB channels
            shift = random.choice([1, 2])
            generated = torch.roll(generated, shifts=shift, dims=0)
            target = 0.4  # Noticeable but not terrible
            
        elif pair_type == 'desaturated':
            # Make image washed out
            generated = gt_tensor.clone()
            gray = generated.mean(dim=0, keepdim=True)
            saturation = random.uniform(0.3, 0.7)
            generated = generated * saturation + gray * (1 - saturation)
            target = 0.45  # Medium-bad
            
        elif pair_type == 'oversaturated':
            # Boost saturation too much
            generated = gt_tensor.clone()
            gray = generated.mean(dim=0, keepdim=True)
            saturation = random.uniform(1.3, 1.6)
            generated = torch.clamp(generated * saturation + gray * (1 - saturation), 0, 1)
            target = 0.35  # Noticeable but not as bad as desaturated
            
        elif pair_type == 'brightness':
            # Wrong brightness/contrast
            generated = gt_tensor.clone()
            if random.random() > 0.5:
                # Darker
                generated = generated * random.uniform(0.6, 0.85)
            else:
                # Brighter (with clipping)
                generated = torch.clamp(generated * random.uniform(1.15, 1.3), 0, 1)
            target = 0.4  # Noticeable issue
            
        elif pair_type == 'real_vhs':
            # Use actual VHS input (already upscaled)
            generated = lq_tensor
            # VHS is not "terrible", it's just different
            # Has legitimate differences that can't be fixed
            target = 0.55  # Medium score - acceptable but not perfect
            
        elif pair_type == 'bicubic':
            # Bicubic upscale (baseline quality)
            generated = lq_tensor
            # Add slight bicubic-specific artifacts
            if random.random() > 0.3:
                # Bicubic often has slight ringing
                kernel = torch.tensor([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]).float() / 9
                kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
                generated = nn.functional.conv2d(
                    generated.unsqueeze(0), kernel, padding=1, groups=3
                ).squeeze(0)
                generated = torch.clamp(generated, 0, 1)
            target = 0.65  # Below average
            
        else:  # terrible
            # Very degraded - heavy blur or severe issues
            generated = lq_tensor.clone()
            degradation = random.choice(['heavy_blur', 'heavy_noise', 'blocky'])
            if degradation == 'heavy_blur':
                # Make it very blurry
                for _ in range(3):
                    generated = nn.functional.avg_pool2d(
                        generated.unsqueeze(0), kernel_size=5, stride=1, padding=2
                    ).squeeze(0)
            elif degradation == 'heavy_noise':
                # Add heavy noise
                noise = torch.randn_like(generated) * 0.15
                generated = torch.clamp(generated + noise, 0, 1)
            else:  # blocky
                # Simulate heavy compression artifacts
                scale = 8
                _, h, w = generated.shape
                generated = nn.functional.interpolate(
                    generated.unsqueeze(0), size=(h//scale, w//scale), mode='nearest'
                ).squeeze(0)
                generated = nn.functional.interpolate(
                    generated.unsqueeze(0), size=(h, w), mode='nearest'
                ).squeeze(0)
            target = 0.85  # Bad score
        
        return {
            'generated': generated,
            'gt': gt_tensor,
            'target_score': torch.tensor([target], dtype=torch.float32)
        }


def train_discriminator(args):
    """Main training function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = ConcatDiscriminator().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset and dataloader
    dataset = DiscriminatorDataset(
        args.gt_dir, args.lq_dir, 
        patch_size=args.patch_size,
        num_samples=args.num_samples,
        saved_pairs_config=args.saved_pairs_config
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                generated = batch['generated'].to(device)
                gt = batch['gt'].to(device)
                target_scores = batch['target_score'].to(device)
                
                # Forward pass
                predicted_scores = model(generated, gt)
                loss = criterion(predicted_scores, target_scores)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'discriminator': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss
            }
            save_path = f"{args.save_dir}/discriminator_epoch_{epoch+1}.pth"
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint to {save_path}")
    
    # Save final model
    final_path = f"{args.save_dir}/discriminator_final.pth"
    torch.save({'discriminator': model.state_dict()}, final_path)
    print(f"Training complete! Final model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory with GT images')
    parser.add_argument('--lq_dir', type=str, required=True, help='Directory with LQ images')
    parser.add_argument('--save_dir', type=str, default='experiments/discriminator', 
                        help='Directory to save checkpoints')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_samples', type=int, default=10000, 
                        help='Number of samples per epoch')
    parser.add_argument('--save_freq', type=int, default=2, 
                        help='Save checkpoint every N epochs')
    parser.add_argument('--saved_pairs_config', type=str, default=None,
                        help='JSON config file with saved artifact triplets')
    
    args = parser.parse_args()
    train_discriminator(args)