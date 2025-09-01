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
    
    def __init__(self, config_path, output_dir, skip_existing=True):
        """Initialize generator with config and output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_existing = skip_existing
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.config_dir = Path(config_path).parent
        self.dataset_pairs = []
        self.gt_files_by_sample = {}  # Track GT files per sample for synthetic generation
        
        # Load existing dataset.json if it exists (for incremental generation)
        self.existing_pairs = set()
        self.initial_pair_count = 0
        dataset_json_path = self.output_dir / 'dataset.json'
        if dataset_json_path.exists() and skip_existing:
            with open(dataset_json_path, 'r') as f:
                existing_data = json.load(f)
                # Create a set of (generated, gt) tuples for fast lookup
                for pair in existing_data:
                    self.existing_pairs.add((pair['generated'], pair['gt']))
                self.dataset_pairs = existing_data  # Start with existing pairs
                self.initial_pair_count = len(existing_data)
                print(f"Loaded {len(self.existing_pairs)} existing pairs from dataset.json")
        
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
        generate_synthetic = sample.get('synthetic', True)
        
        print(f"\nProcessing sample: {sample_name}")
        if not generate_synthetic:
            print(f"  (Synthetic generation disabled for this sample)")
        
        # Create output directory for this sample
        sample_dir = self.output_dir / sample_name
        sample_dir.mkdir(exist_ok=True)
        
        # Find all GT files
        gt_files = sorted(path.glob(f'*{gt_suffix}'))
        print(f"Found {len(gt_files)} GT files")
        
        # Copy GT files to output directory
        gt_output_dir = sample_dir / 'gt'
        gt_output_dir.mkdir(exist_ok=True)
        
        # Track GT files for this sample if synthetic generation is enabled
        if generate_synthetic:
            self.gt_files_by_sample[sample_name] = []
        
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
            
            # Save GT image with sample name prefix for consistency
            gt_output_filename = f"{sample_name}_{gt_file.name}"
            gt_output_path = gt_output_dir / gt_output_filename
            
            # Skip if GT file already exists
            if self.skip_existing and gt_output_path.exists():
                # Still track it for synthetic generation
                if generate_synthetic:
                    self.gt_files_by_sample[sample_name].append(gt_output_path)
            else:
                cv2.imwrite(str(gt_output_path), cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))
                # Track this GT file if synthetic generation is enabled
                if generate_synthetic:
                    self.gt_files_by_sample[sample_name].append(gt_output_path)
            
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
                
                # Save processed image with sample name prefix to avoid collisions
                output_filename = f"{sample_name}_{base_name}_gen.png"
                output_path = gen_output_dir / output_filename
                
                # Check if this pair already exists
                gen_rel_path = str(output_path.relative_to(self.output_dir))
                gt_rel_path = str(gt_output_path.relative_to(self.output_dir))
                pair_exists = (gen_rel_path, gt_rel_path) in self.existing_pairs
                
                if self.skip_existing and output_path.exists() and pair_exists:
                    continue  # Skip this generation
                
                cv2.imwrite(str(output_path), cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
                
                # Add to dataset pairs only if it's new
                if not pair_exists:
                    self.dataset_pairs.append({
                        'generated': gen_rel_path,
                        'gt': gt_rel_path,
                        'loss': loss_value
                    })
                    self.existing_pairs.add((gen_rel_path, gt_rel_path))
    
    def generate_synthetic_samples(self):
        """Generate additional synthetic degradations from GT images."""
        print("\nGenerating synthetic degradations...")
        
        synthetic_dir = self.output_dir / 'synthetic'
        synthetic_dir.mkdir(exist_ok=True)
        
        # Use the GT files we tracked during processing
        all_gt_files = []
        for sample_name, gt_files in self.gt_files_by_sample.items():
            all_gt_files.extend(gt_files)
            print(f"  Including {len(gt_files)} GT files from {sample_name} for synthetic generation")
        
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
                'loss': 0.50,
                'operations': [{'type': 'blur', 'kernel_size': 3}]
            },
            {
                'name': 'heavy_blur',
                'loss': 0.75,
                'operations': [{'type': 'blur', 'kernel_size': 7}]
            },
            {
                'name': 'noise_low',
                'loss': 0.60,
                'operations': [{'type': 'noise', 'level': 0.02}]
            },
            {
                'name': 'noise_high',
                'loss': 0.76,
                'operations': [{'type': 'noise', 'level': 0.08}]
            },
            {
                'name': 'desaturated',
                'loss': 0.70,
                'operations': [{'type': 'saturation', 'scale': 0.5}]
            },
            {
                'name': 'oversaturated',
                'loss': 0.70,
                'operations': [{'type': 'saturation', 'scale': 1.5}]
            },
            {
                'name': 'dark',
                'loss': 0.65,
                'operations': [{'type': 'brightness', 'scale': 0.7}]
            },
            {
                'name': 'bright',
                'loss': 0.65,
                'operations': [{'type': 'brightness', 'scale': 1.3}]
            },
            {
                'name': 'hue_shift',
                'loss': 0.65,
                'operations': [{'type': 'hue_shift', 'shift': 20}]
            },
            {
                'name': 'downup_bicubic',
                'loss': 0.73,
                'operations': [
                    {'type': 'resize', 'method': 'bicubic', 'scale': 0.25},
                    {'type': 'resize', 'method': 'bicubic', 'scale': 4.0}
                ]
            },
            {
                'name': 'downup_nearest',
                'loss': 0.75,
                'operations': [
                    {'type': 'resize', 'method': 'nearest', 'scale': 0.25},
                    {'type': 'resize', 'method': 'nearest', 'scale': 4.0}
                ]
            },
            {
                'name': 'vignette_light',
                'loss': 0.55,
                'operations': [{'type': 'vignette', 'strength': 0.3, 'edge_fade': 0.2}]
            },
            {
                'name': 'vignette_heavy',
                'loss': 0.60,
                'operations': [{'type': 'vignette', 'strength': 0.7, 'edge_fade': 0.3}]
            },
            {
                'name': 'vignette_extreme',
                'loss': 0.70,
                'operations': [{'type': 'vignette', 'strength': 0.9, 'edge_fade': 0.15}]
            }
        ]
        
        # Sample GT files for synthetic generation
        max_synthetic = self.config.get('max_synthetic_samples', None)  # None = use all
        if max_synthetic and max_synthetic < len(all_gt_files):
            num_synthetic = max_synthetic
            sampled_gt_files = random.sample(all_gt_files, num_synthetic)
            print(f"  Sampling {num_synthetic} GT files for synthetic generation (from {len(all_gt_files)} available)")
        else:
            sampled_gt_files = all_gt_files
            print(f"  Using all {len(all_gt_files)} GT files for synthetic generation")
        
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
                
                # Get the sample name from the gt_file path (parent of 'gt' directory)
                sample_name = gt_file.parent.parent.name
                
                # Save with sample name prefix to avoid collisions
                output_path = deg_dir / f"{sample_name}_{base_name}_syn.png"
                
                # Check if this pair already exists
                gen_rel_path = str(output_path.relative_to(self.output_dir))
                gt_rel_path = str(gt_file.relative_to(self.output_dir))
                pair_exists = (gen_rel_path, gt_rel_path) in self.existing_pairs
                
                if self.skip_existing and output_path.exists() and pair_exists:
                    continue  # Skip this synthetic generation
                
                cv2.imwrite(str(output_path), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
                
                # Add to pairs only if it's new
                if not pair_exists:
                    self.dataset_pairs.append({
                        'generated': gen_rel_path,
                        'gt': gt_rel_path,
                        'loss': deg['loss']
                    })
                    self.existing_pairs.add((gen_rel_path, gt_rel_path))
    
    def save_dataset_json(self):
        """Save the dataset pairs to a JSON file."""
        output_json = self.output_dir / 'dataset.json'
        
        # Count new pairs added in this run
        new_pairs = len(self.dataset_pairs) - self.initial_pair_count
        
        with open(output_json, 'w') as f:
            json.dump(self.dataset_pairs, f, indent=2)
        print(f"\nDataset JSON saved to {output_json}")
        print(f"Total pairs: {len(self.dataset_pairs)}")
        if self.skip_existing and new_pairs > 0:
            print(f"New pairs added: {new_pairs}")
        elif self.skip_existing and new_pairs == 0:
            print("No new pairs added (all files already existed)")
    
    def generate_mismatched_pairs(self):
        """Generate pairs of completely different images for maximum dissimilarity."""
        print("\nGenerating mismatched pairs (completely different images)...")
        
        # Use the GT files we tracked during processing
        sample_gt_files = {name: files for name, files in self.gt_files_by_sample.items() if files}
        
        if len(sample_gt_files) < 2:
            print("  Need at least 2 different samples to create mismatched pairs")
            return
        
        for sample_name, files in sample_gt_files.items():
            print(f"  Found {len(files)} GT files in {sample_name}")
        
        # Generate pairs between different samples
        sample_names = list(sample_gt_files.keys())
        num_mismatched = self.config.get('num_mismatched_pairs', 6000)  # Configurable number
        mismatched_count = 0
        
        for i in range(len(sample_names)):
            for j in range(i + 1, len(sample_names)):
                sample1 = sample_names[i]
                sample2 = sample_names[j]
                
                # Get files from each sample
                files1 = sample_gt_files[sample1]
                files2 = sample_gt_files[sample2]
                
                # Create some pairs (limit to avoid explosion)
                pairs_to_create = min(10, len(files1), len(files2), num_mismatched - mismatched_count)
                
                for k in range(pairs_to_create):
                    # Pick random files from each sample
                    file1 = random.choice(files1)
                    file2 = random.choice(files2)
                    
                    # Check if this mismatched pair already exists
                    gen_rel_path = str(file1.relative_to(self.output_dir))
                    gt_rel_path = str(file2.relative_to(self.output_dir))
                    pair_exists = (gen_rel_path, gt_rel_path) in self.existing_pairs
                    
                    if not pair_exists:
                        # Add mismatched pair with very high loss (0.95-1.0)
                        # These are completely different images
                        self.dataset_pairs.append({
                            'generated': gen_rel_path,
                            'gt': gt_rel_path,
                            'loss': 1.0,  # Very bad match
                            'type': 'mismatched'  # Tag for identification
                        })
                        self.existing_pairs.add((gen_rel_path, gt_rel_path))
                    mismatched_count += 1
                    
                    if mismatched_count >= num_mismatched:
                        break
                
                if mismatched_count >= num_mismatched:
                    break
        
        print(f"  Generated {mismatched_count} mismatched pairs")
    
    def generate(self):
        """Generate the complete dataset."""
        # Process each sample in config
        for sample in self.config.get('samples', []):
            self.process_sample(sample)
        
        # Generate synthetic samples
        if self.config.get('generate_synthetic', True):
            self.generate_synthetic_samples()
        
        # Generate mismatched pairs (completely different images)
        if self.config.get('generate_mismatched', True):
            self.generate_mismatched_pairs()
        
        # Save dataset JSON
        self.save_dataset_json()


def main():
    parser = argparse.ArgumentParser(description='Generate discriminator training dataset')
    parser.add_argument('config', type=str, help='Path to configuration JSON file')
    parser.add_argument('-o', '--output', type=str, required=True, 
                        help='Output directory for generated dataset')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration of all files (ignore existing)')
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(args.config, args.output, skip_existing=not args.force)
    generator.generate()
    
    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()