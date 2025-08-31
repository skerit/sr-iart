"""Generate dataset for discriminator training with configurable operations."""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import random


class DatasetGenerator:
    """Generate discriminator training dataset from config."""
    
    def __init__(self, config_path, output_dir):
        """Initialize generator with config and output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.config_dir = Path(config_path).parent
        self.dataset_pairs = []
        
    def apply_operations(self, image, operations, target_size=None):
        """Apply a series of operations to an image."""
        result = image.copy()
        h, w = result.shape[:2]
        
        for op in operations:
            op_type = op['type']
            
            if op_type == 'resize':
                method = op.get('method', 'bicubic')
                scale = op.get('scale', None)
                
                if scale:
                    # Scale relative to current size
                    new_h = int(h * scale)
                    new_w = int(w * scale)
                elif target_size:
                    # Scale to target size (GT size)
                    new_h, new_w = target_size
                else:
                    continue
                
                interp_map = {
                    'nearest': cv2.INTER_NEAREST,
                    'bilinear': cv2.INTER_LINEAR,
                    'bicubic': cv2.INTER_CUBIC,
                    'lanczos': cv2.INTER_LANCZOS4
                }
                interp = interp_map.get(method, cv2.INTER_CUBIC)
                result = cv2.resize(result, (new_w, new_h), interpolation=interp)
                h, w = result.shape[:2]  # Update dimensions
                
            elif op_type == 'noise':
                level = float(op.get('level', 0.05))
                noise = np.random.randn(*result.shape) * level * 255
                result = np.clip(result + noise, 0, 255).astype(np.uint8)
                
            elif op_type == 'saturation':
                scale = float(op.get('scale', 1.0))  # Ensure it's a float
                # Convert to float for processing
                img_float = result.astype(np.float32) / 255.0
                
                # Convert RGB to HSV
                hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV)
                
                # Scale saturation channel
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 1)
                
                # Convert back to RGB
                result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                result = (result * 255).astype(np.uint8)
                
            elif op_type == 'brightness':
                scale = float(op.get('scale', 1.0))
                result = np.clip(result * scale, 0, 255).astype(np.uint8)
                
            elif op_type == 'hue_shift':
                shift = float(op.get('shift', 0))  # Hue shift in degrees
                img_float = result.astype(np.float32) / 255.0
                hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV)
                hsv[:, :, 0] = (hsv[:, :, 0] + shift/360.0) % 1.0
                result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                result = (result * 255).astype(np.uint8)
                
            elif op_type == 'blur':
                kernel_size = int(op.get('kernel_size', 3))
                if kernel_size % 2 == 0:
                    kernel_size += 1  # Ensure odd number
                result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
                
            elif op_type == 'sharpen':
                amount = float(op.get('amount', 1.0))
                kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]]) * amount
                kernel[1, 1] = 1 + 8 * amount
                result = cv2.filter2D(result, -1, kernel)
                result = np.clip(result, 0, 255).astype(np.uint8)
                
            elif op_type == 'vignette':
                # Non-circular (rectangular) vignette
                strength = float(op.get('strength', 0.5))  # 0-1, higher = darker edges
                edge_fade = float(op.get('edge_fade', 0.3))  # Portion of image that fades (0-0.5)
                
                h, w = result.shape[:2]
                
                # Create gradient masks for each edge
                fade_h = int(h * edge_fade)
                fade_w = int(w * edge_fade)
                
                # Create vignette mask
                mask = np.ones((h, w), dtype=np.float32)
                
                # Top and bottom gradients
                for i in range(fade_h):
                    fade_factor = i / float(fade_h)
                    mask[i, :] *= fade_factor
                    mask[h-1-i, :] *= fade_factor
                
                # Left and right gradients
                for j in range(fade_w):
                    fade_factor = j / float(fade_w)
                    mask[:, j] *= fade_factor
                    mask[:, w-1-j] *= fade_factor
                
                # Apply strength
                mask = 1.0 - (1.0 - mask) * strength
                
                # Apply mask to all channels
                mask_3d = np.stack([mask, mask, mask], axis=2)
                result = (result * mask_3d).astype(np.uint8)
        
        return result
    
    def process_sample(self, sample):
        """Process a single sample configuration."""
        sample_name = sample['name']
        sample_type = sample.get('type', 'images')
        path = self.config_dir / sample['path']
        gt_suffix = sample.get('gt_suffix', '_gt.png')
        
        print(f"\nProcessing sample: {sample_name}")
        
        # Create output directory for this sample
        sample_dir = self.output_dir / sample_name
        sample_dir.mkdir(exist_ok=True)
        
        # Find all GT files
        gt_files = sorted(path.glob(f'*{gt_suffix}'))
        print(f"Found {len(gt_files)} GT files")
        
        # Copy GT files to output directory
        gt_output_dir = sample_dir / 'gt'
        gt_output_dir.mkdir(exist_ok=True)
        
        for gt_file in tqdm(gt_files, desc="Processing files"):
            # Load GT image
            gt_img = cv2.imread(str(gt_file))
            if gt_img is None:
                print(f"Warning: Could not load {gt_file}")
                continue
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            gt_h, gt_w = gt_img.shape[:2]
            
            # Get base name for this file
            base_name = gt_file.name.replace(gt_suffix, '')
            
            # Save GT image
            gt_output_path = gt_output_dir / gt_file.name
            cv2.imwrite(str(gt_output_path), cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))
            
            # Process each generation configuration
            for gen_config in sample.get('generations', []):
                gen_name = gen_config['name']
                source_suffix = gen_config.get('source_suffix', gt_suffix)
                loss_value = gen_config['loss']
                operations = gen_config.get('operations', [])
                
                # Create output directory for this generation
                gen_output_dir = sample_dir / gen_name
                gen_output_dir.mkdir(exist_ok=True)
                
                # Load source image
                if source_suffix == gt_suffix:
                    source_img = gt_img
                else:
                    source_file = path / f"{base_name}{source_suffix}"
                    if not source_file.exists():
                        continue
                    source_img = cv2.imread(str(source_file))
                    if source_img is None:
                        continue
                    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
                
                # Apply operations
                processed_img = self.apply_operations(
                    source_img, operations, 
                    target_size=(gt_h, gt_w)
                )
                
                # Save processed image
                output_filename = f"{base_name}_gen.png"
                output_path = gen_output_dir / output_filename
                cv2.imwrite(str(output_path), cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
                
                # Add to dataset pairs
                self.dataset_pairs.append({
                    'generated': str(output_path.relative_to(self.output_dir)),
                    'gt': str(gt_output_path.relative_to(self.output_dir)),
                    'loss': loss_value
                })
    
    def generate_synthetic_samples(self):
        """Generate additional synthetic degradations from GT images."""
        print("\nGenerating synthetic degradations...")
        
        synthetic_dir = self.output_dir / 'synthetic'
        synthetic_dir.mkdir(exist_ok=True)
        
        # Collect all GT images
        all_gt_files = list(self.output_dir.glob('*/gt/*.png'))
        
        if not all_gt_files:
            print("No GT files found for synthetic generation")
            return
        
        # Define synthetic degradations
        degradations = [
            {
                'name': 'perfect',
                'loss': 0.0,
                'operations': []  # No change
            },
            {
                'name': 'slight_blur',
                'loss': 0.60,
                'operations': [{'type': 'blur', 'kernel_size': 3}]
            },
            {
                'name': 'heavy_blur',
                'loss': 0.85,
                'operations': [{'type': 'blur', 'kernel_size': 7}]
            },
            {
                'name': 'noise_low',
                'loss': 0.70,
                'operations': [{'type': 'noise', 'level': 0.02}]
            },
            {
                'name': 'noise_high',
                'loss': 0.86,
                'operations': [{'type': 'noise', 'level': 0.08}]
            },
            {
                'name': 'desaturated',
                'loss': 0.75,
                'operations': [{'type': 'saturation', 'scale': 0.5}]
            },
            {
                'name': 'oversaturated',
                'loss': 0.70,
                'operations': [{'type': 'saturation', 'scale': 1.5}]
            },
            {
                'name': 'dark',
                'loss': 0.70,
                'operations': [{'type': 'brightness', 'scale': 0.7}]
            },
            {
                'name': 'bright',
                'loss': 0.70,
                'operations': [{'type': 'brightness', 'scale': 1.3}]
            },
            {
                'name': 'hue_shift',
                'loss': 0.70,
                'operations': [{'type': 'hue_shift', 'shift': 20}]
            },
            {
                'name': 'downup_bicubic',
                'loss': 0.83,
                'operations': [
                    {'type': 'resize', 'method': 'bicubic', 'scale': 0.25},
                    {'type': 'resize', 'method': 'bicubic', 'scale': 4.0}
                ]
            },
            {
                'name': 'downup_nearest',
                'loss': 0.85,
                'operations': [
                    {'type': 'resize', 'method': 'nearest', 'scale': 0.25},
                    {'type': 'resize', 'method': 'nearest', 'scale': 4.0}
                ]
            },
            {
                'name': 'vignette_light',
                'loss': 0.65,
                'operations': [{'type': 'vignette', 'strength': 0.3, 'edge_fade': 0.2}]
            },
            {
                'name': 'vignette_heavy',
                'loss': 0.70,
                'operations': [{'type': 'vignette', 'strength': 0.7, 'edge_fade': 0.3}]
            },
            {
                'name': 'vignette_extreme',
                'loss': 0.75,
                'operations': [{'type': 'vignette', 'strength': 0.9, 'edge_fade': 0.15}]
            }
        ]
        
        # Sample some GT files for synthetic generation
        num_synthetic = min(len(all_gt_files), 500)  # Limit to 500 samples
        sampled_gt_files = random.sample(all_gt_files, num_synthetic)
        
        for gt_file in tqdm(sampled_gt_files, desc="Generating synthetic samples"):
            gt_img = cv2.imread(str(gt_file))
            if gt_img is None:
                continue
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            
            base_name = gt_file.stem
            
            # Apply each degradation
            for deg in degradations:
                deg_dir = synthetic_dir / deg['name']
                deg_dir.mkdir(exist_ok=True)
                
                # Apply operations
                processed = self.apply_operations(gt_img, deg['operations'])
                
                # Save
                output_path = deg_dir / f"{base_name}_syn.png"
                cv2.imwrite(str(output_path), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
                
                # Add to pairs
                self.dataset_pairs.append({
                    'generated': str(output_path.relative_to(self.output_dir)),
                    'gt': str(gt_file.relative_to(self.output_dir)),
                    'loss': deg['loss']
                })
    
    def save_dataset_json(self):
        """Save the dataset pairs to a JSON file."""
        output_json = self.output_dir / 'dataset.json'
        with open(output_json, 'w') as f:
            json.dump(self.dataset_pairs, f, indent=2)
        print(f"\nDataset JSON saved to {output_json}")
        print(f"Total pairs: {len(self.dataset_pairs)}")
    
    def generate(self):
        """Generate the complete dataset."""
        # Process each sample in config
        for sample in self.config.get('samples', []):
            self.process_sample(sample)
        
        # Generate synthetic samples
        if self.config.get('generate_synthetic', True):
            self.generate_synthetic_samples()
        
        # Save dataset JSON
        self.save_dataset_json()


def main():
    parser = argparse.ArgumentParser(description='Generate discriminator training dataset')
    parser.add_argument('config', type=str, help='Path to configuration JSON file')
    parser.add_argument('-o', '--output', type=str, required=True, 
                        help='Output directory for generated dataset')
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(args.config, args.output)
    generator.generate()
    
    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()