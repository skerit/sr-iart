"""Test discriminator checkpoint on sample images."""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from archs.concat_discriminator import ConcatDiscriminator


def load_discriminator(checkpoint_path, device='cuda', override_base_channels=None, override_num_layers=None):
    """Load discriminator from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to detect model configuration from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'discriminator' in checkpoint:
        state_dict = checkpoint['discriminator']
    else:
        state_dict = checkpoint
    
    # Detect base_channels from first layer weight shape
    first_layer_weight = state_dict.get('features.0.weight')
    if first_layer_weight is not None:
        base_channels = first_layer_weight.shape[0]
    else:
        base_channels = 192  # Default to new size
    
    # Detect num_layers by counting conv layers in features
    num_layers = 0
    for key in state_dict.keys():
        if key.startswith('features.') and 'weight' in key:
            # Count only Conv2d layers (not BatchNorm)
            layer_idx = int(key.split('.')[1])
            if layer_idx % 3 == 0:  # Conv layers are at indices 0, 3, 6, 9...
                num_layers += 1
    
    # If detection failed, use defaults
    if num_layers == 0:
        num_layers = 7  # Default to new size
    
    # Apply overrides if provided
    if override_base_channels is not None:
        base_channels = override_base_channels
    if override_num_layers is not None:
        num_layers = override_num_layers
    
    print(f"Model configuration: base_channels={base_channels}, num_layers={num_layers}")
    
    # Create model with detected/overridden configuration
    model = ConcatDiscriminator(in_channels=6, base_channels=base_channels, num_layers=num_layers)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'discriminator' in checkpoint:
        model.load_state_dict(checkpoint['discriminator'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Print checkpoint info if available
    if 'iteration' in checkpoint:
        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"Training loss: {checkpoint['loss']:.4f}")
    
    return model


def score_image_pair(model, generated_path, gt_path, patch_size=128, device='cuda'):
    """Score a generated/GT image pair."""
    # Load images
    gen_img = cv2.imread(str(generated_path))
    if gen_img is None:
        raise ValueError(f"Could not load {generated_path}")
    gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
    
    gt_img = cv2.imread(str(gt_path))
    if gt_img is None:
        raise ValueError(f"Could not load {gt_path}")
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    
    # Get dimensions
    h, w = gen_img.shape[:2]
    
    # If images are larger than patch_size, extract multiple patches
    if h > patch_size and w > patch_size:
        scores = []
        
        # Grid of patches
        for y in range(0, h - patch_size + 1, patch_size // 2):
            for x in range(0, w - patch_size + 1, patch_size // 2):
                gen_patch = gen_img[y:y+patch_size, x:x+patch_size]
                gt_patch = gt_img[y:y+patch_size, x:x+patch_size]
                
                # Convert to tensors
                gen_tensor = torch.from_numpy(gen_patch.transpose(2, 0, 1)).float() / 255.0
                gt_tensor = torch.from_numpy(gt_patch.transpose(2, 0, 1)).float() / 255.0
                
                # Add batch dimension
                gen_tensor = gen_tensor.unsqueeze(0).to(device)
                gt_tensor = gt_tensor.unsqueeze(0).to(device)
                
                # Get score (use sigmoid for testing)
                with torch.no_grad():
                    logits = model(gen_tensor, gt_tensor)
                    score = torch.sigmoid(logits)
                scores.append(score.item())
        
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        return avg_score, std_score, scores
    else:
        # Single patch
        gen_tensor = torch.from_numpy(gen_img.transpose(2, 0, 1)).float() / 255.0
        gt_tensor = torch.from_numpy(gt_img.transpose(2, 0, 1)).float() / 255.0
        
        gen_tensor = gen_tensor.unsqueeze(0).to(device)
        gt_tensor = gt_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(gen_tensor, gt_tensor)
            score = torch.sigmoid(logits)
        return score.item(), 0.0, [score.item()]


def test_on_dataset(model, dataset_dir, num_samples=10, patch_size=128, device='cuda'):
    """Test discriminator on dataset samples."""
    dataset_dir = Path(dataset_dir)
    
    print("\n" + "="*60)
    print("Testing on dataset samples")
    print("="*60)
    
    # Test different quality levels if they exist
    test_cases = [
        ('Perfect (GT vs GT)', 'gt', 'gt', 0.0),
        ('Good (output vs GT)', '*/output', 'gt', 0.45),
        ('Bicubic upscale', '*/lqinput-bicubic', 'gt', 0.85),
        ('Bilinear upscale', '*/lqinput-bilinear', 'gt', 0.90),
        ('Nearest upscale', '*/lqinput-nearest', 'gt', 0.95),
        ('Synthetic blur', 'synthetic/slight_blur', 'gt', 0.60),
        ('Synthetic heavy blur', 'synthetic/heavy_blur', 'gt', 0.85),
        ('Synthetic noise', 'synthetic/noise_low', 'gt', 0.70),
        ('Synthetic desaturated', 'synthetic/desaturated', 'gt', 0.75),
    ]
    
    for test_name, gen_pattern, gt_pattern, expected_score in test_cases:
        # Find matching files
        gen_files = list(dataset_dir.glob(f"{gen_pattern}/*.png"))[:num_samples]
        
        if not gen_files:
            print(f"\n{test_name}: No files found")
            continue
        
        print(f"\n{test_name} (Expected: ~{expected_score:.2f}):")
        scores = []
        
        for gen_file in gen_files[:min(num_samples, len(gen_files))]:
            # Find corresponding GT file
            if gt_pattern == 'gt':
                # Find GT file with same base name
                base_name = gen_file.stem.split('_')[0]
                gt_files = list(dataset_dir.glob(f"*/gt/{base_name}*_gt.png"))
                if not gt_files:
                    gt_files = list(dataset_dir.glob(f"gt/{base_name}*.png"))
                if not gt_files:
                    continue
                gt_file = gt_files[0]
            else:
                gt_file = gen_file  # For GT vs GT test
            
            try:
                score, std, _ = score_image_pair(model, gen_file, gt_file, patch_size, device)
                scores.append(score)
            except Exception as e:
                print(f"  Error processing {gen_file.name}: {e}")
        
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  Mean: {mean_score:.4f} ± {std_score:.4f}")
            print(f"  Range: [{min(scores):.4f}, {max(scores):.4f}]")
            print(f"  Difference from expected: {mean_score - expected_score:+.4f}")


def test_specific_pair(model, generated_path, gt_path, patch_size=128, device='cuda'):
    """Test a specific image pair."""
    print("\n" + "="*60)
    print("Testing specific image pair")
    print("="*60)
    print(f"Generated: {generated_path}")
    print(f"GT: {gt_path}")
    
    score, std, patch_scores = score_image_pair(
        model, generated_path, gt_path, patch_size, device
    )
    
    print(f"\nQuality Score: {score:.4f}")
    if len(patch_scores) > 1:
        print(f"Std Dev: {std:.4f}")
        print(f"Min/Max: [{min(patch_scores):.4f}, {max(patch_scores):.4f}]")
        print(f"Num patches: {len(patch_scores)}")
    
    # Interpret score
    if score < 0.1:
        quality = "Perfect/Near-perfect"
    elif score < 0.3:
        quality = "Very Good"
    elif score < 0.5:
        quality = "Good"
    elif score < 0.7:
        quality = "Acceptable"
    elif score < 0.85:
        quality = "Poor"
    else:
        quality = "Very Poor"
    
    print(f"Quality Assessment: {quality}")


def main():
    parser = argparse.ArgumentParser(description='Test discriminator checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--generated', type=str, help='Path to generated image')
    parser.add_argument('--gt', type=str, help='Path to ground truth image')
    parser.add_argument('--dataset_dir', type=str, help='Dataset directory for testing')
    parser.add_argument('--patch_size', type=int, default=128, 
                        help='Patch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples per category to test')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--base_channels', type=int, help='Override base channels (auto-detect if not specified)')
    parser.add_argument('--num_layers', type=int, help='Override num layers (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_discriminator(args.checkpoint, device, args.base_channels, args.num_layers)
    print(f"Loaded discriminator from {args.checkpoint}")
    
    # Test specific pair if provided
    if args.generated and args.gt:
        test_specific_pair(model, args.generated, args.gt, args.patch_size, device)
    
    # Test on dataset samples if provided
    if args.dataset_dir:
        test_on_dataset(model, args.dataset_dir, args.num_samples, args.patch_size, device)
    
    if not (args.generated or args.dataset_dir):
        print("\nNo test images specified. Use --generated/--gt or --dataset_dir")


if __name__ == "__main__":
    main()