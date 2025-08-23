"""
Multi-scale dataset for handling VHS warble/flutter without pre-alignment
"""
import random
import torch
from pathlib import Path
from basicsr.data.reds_dataset import REDSRecurrentDataset
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class DVDMultiScaleDataset(REDSRecurrentDataset):
    """
    Alternates between different patch sizes to give the model both
    local detail and global context for learning VHS artifacts.
    """
    
    def __init__(self, opt):
        super().__init__(opt)
        
        # Multi-scale patch sizes (in GT resolution)
        self.patch_sizes = opt.get('patch_sizes', [256, 384, 512])
        self.patch_probabilities = opt.get('patch_probs', [0.6, 0.3, 0.1])
        
        # Build clip frame counts
        self.clip_frames = {}
        for key in self.keys:
            clip, frame = key.split('/')
            frame_idx = int(frame)
            if clip not in self.clip_frames:
                self.clip_frames[clip] = frame_idx
            else:
                self.clip_frames[clip] = max(self.clip_frames[clip], frame_idx)
        
        logger = get_root_logger()
        logger.info(f'Multi-scale DVD dataset: patch sizes {self.patch_sizes}')
    
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        
        # Randomly select patch size for this iteration
        gt_size = random.choices(self.patch_sizes, self.patch_probabilities)[0]
        
        key = self.keys[index]
        clip_name, frame_name = key.split('/')
        
        # Get max frames for this clip
        max_frame = self.clip_frames[clip_name]
        
        # determine the neighboring frames
        interval = random.choice(self.interval_list)
        
        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        max_start = max(0, max_frame - self.num_frame * interval)
        
        if start_frame_idx > max_start:
            start_frame_idx = random.randint(0, max_start)
        
        end_frame_idx = start_frame_idx + self.num_frame * interval
        
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))
        
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:08d}'
                img_gt_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:08d}.png'

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop with selected patch size
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'patch_size': gt_size}