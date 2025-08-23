"""Video-based dataset that reads directly from video files instead of image sequences."""

import cv2
import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VideoRecurrentDataset(data.Dataset):
    """Video dataset for training recurrent networks using video files directly.
    
    Reads frames directly from video files instead of individual images.
    Supports FFV1 lossless codec for best quality with good compression.
    """

    def __init__(self, opt):
        super(VideoRecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_root = opt['dataroot_gt']
        self.lq_root = opt['dataroot_lq']
        
        self.num_frame = opt.get('num_frame', 5)
        self.num_half_frames = opt.get('num_half_frames', 2)
        
        # Video-specific options
        self.meta_info_file = opt.get('meta_info_file', None)
        self.clips = []
        
        # Load clip metadata
        if self.meta_info_file:
            import json
            with open(self.meta_info_file, 'r') as f:
                self.clips = json.load(f)
        else:
            # Auto-discover video files
            import os
            for video in sorted(os.listdir(self.gt_root)):
                if video.endswith(('.mkv', '.mp4', '.avi')):
                    clip_name = os.path.splitext(video)[0]
                    # Open video to get frame count
                    cap = cv2.VideoCapture(os.path.join(self.gt_root, video))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    self.clips.append({
                        'name': clip_name,
                        'gt_video': video,
                        'lq_video': video,  # Assume same filename
                        'frame_count': frame_count
                    })
        
        # Data augmentation
        self.random_reverse = opt.get('random_reverse', False)
        self.use_hflip = opt.get('use_hflip', False)
        self.use_rot = opt.get('use_rot', False)
        
        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        """Get training sample.
        
        Returns:
            dict: Training sample with keys:
                - lq: Low quality frames tensor
                - gt: Ground truth frames tensor
                - key: Sample identifier
        """
        clip_idx = index % len(self.clips)
        clip = self.clips[clip_idx]
        
        # Open video files
        gt_cap = cv2.VideoCapture(os.path.join(self.gt_root, clip['gt_video']))
        lq_cap = cv2.VideoCapture(os.path.join(self.lq_root, clip['lq_video']))
        
        # Get frame count and ensure we don't go out of bounds
        frame_count = clip['frame_count']
        max_start = max(1, frame_count - self.num_frame - max(self.interval_list))
        
        # Random start frame
        start_frame = np.random.randint(0, max_start)
        
        # Random interval
        interval = np.random.choice(self.interval_list)
        
        # Read frames
        img_gts = []
        img_lqs = []
        
        for i in range(self.num_frame):
            frame_idx = start_frame + i * interval
            
            # Read GT frame
            gt_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, gt_frame = gt_cap.read()
            if not ret:
                # If we can't read, use last valid frame
                if img_gts:
                    gt_frame = img_gts[-1]
                else:
                    gt_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                gt_frame = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2RGB)
            
            # Read LQ frame
            lq_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, lq_frame = lq_cap.read()
            if not ret:
                # If we can't read, use last valid frame
                if img_lqs:
                    lq_frame = img_lqs[-1]
                else:
                    lq_frame = np.zeros((180, 320, 3), dtype=np.uint8)
            else:
                lq_frame = cv2.cvtColor(lq_frame, cv2.COLOR_BGR2RGB)
            
            img_gts.append(gt_frame)
            img_lqs.append(lq_frame)
        
        # Release video captures
        gt_cap.release()
        lq_cap.release()
        
        # Convert to numpy arrays
        img_gts = np.stack(img_gts, axis=0)  # (T, H, W, C)
        img_lqs = np.stack(img_lqs, axis=0)  # (T, H, W, C)
        
        # Random reverse
        if self.random_reverse and np.random.random() < 0.5:
            img_gts = img_gts[::-1]
            img_lqs = img_lqs[::-1]
        
        # Data augmentation
        if self.use_hflip and np.random.random() < 0.5:
            img_gts = img_gts[:, :, ::-1]
            img_lqs = img_lqs[:, :, ::-1]
            
        if self.use_rot and np.random.random() < 0.5:
            # Random rotation (90, 180, or 270 degrees)
            k = np.random.randint(1, 4)
            img_gts = np.stack([np.rot90(img, k) for img in img_gts])
            img_lqs = np.stack([np.rot90(img, k) for img in img_lqs])
        
        # Convert to tensor (T, C, H, W)
        img_gts = torch.from_numpy(img_gts).float().permute(0, 3, 1, 2) / 255.0
        img_lqs = torch.from_numpy(img_lqs).float().permute(0, 3, 1, 2) / 255.0
        
        return {
            'lq': img_lqs,
            'gt': img_gts,
            'key': f"{clip['name']}_{start_frame:08d}"
        }

    def __len__(self):
        """Dataset length is clips * possible starting positions."""
        # Return a large number to allow many iterations
        return len(self.clips) * 1000