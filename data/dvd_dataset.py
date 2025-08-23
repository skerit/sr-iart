"""
Custom dataset for DVD to Blu-ray restoration with variable frame counts per clip
"""
import random
import torch
from os import path as osp
from pathlib import Path
from torch.utils import data as data

from basicsr.data.reds_dataset import REDSRecurrentDataset
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class DVDRecurrentDataset(REDSRecurrentDataset):
    """DVD dataset for training recurrent networks with variable frame counts.
    
    Overrides REDSRecurrentDataset to handle clips with different frame counts.
    """
    
    def __init__(self, opt):
        super(DVDRecurrentDataset, self).__init__(opt)
        
        # Build a dictionary of max frame indices for each clip
        self.clip_max_frames = {}
        for key in self.keys:
            clip_name = key.split('/')[0]
            frame_idx = int(key.split('/')[1])
            if clip_name not in self.clip_max_frames:
                self.clip_max_frames[clip_name] = frame_idx
            else:
                self.clip_max_frames[clip_name] = max(self.clip_max_frames[clip_name], frame_idx)
        
        logger = get_root_logger()
        logger.info(f'DVD dataset initialized with {len(self.clip_max_frames)} clips')
        
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000
        
        # Get the maximum frame index for this clip (0-indexed)
        max_frame_idx = self.clip_max_frames[clip_name]
        
        # determine the neighboring frames
        interval = random.choice(self.interval_list)
        
        # ensure not exceeding the borders of THIS specific clip
        start_frame_idx = int(frame_name)
        max_start_idx = max(0, max_frame_idx - (self.num_frame - 1) * interval)
        
        if start_frame_idx > max_start_idx:
            start_frame_idx = random.randint(0, max_start_idx)
        
        end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval
        
        # Make sure we don't exceed the clip boundaries
        if end_frame_idx > max_frame_idx:
            # Adjust start_frame_idx to fit the sequence
            start_frame_idx = max(0, max_frame_idx - (self.num_frame - 1) * interval)
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval
        
        neighbor_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
        
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

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}