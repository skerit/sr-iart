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
        gt_path = os.path.join(self.gt_root, clip['gt_video'])
        lq_path = os.path.join(self.lq_root, clip['lq_video'])
        
        gt_cap = cv2.VideoCapture(gt_path)
        if not gt_cap.isOpened():
            logger = get_root_logger()
            logger.error(f"Failed to open GT video: {gt_path}")
            raise RuntimeError(f"Cannot open video: {gt_path}")
            
        lq_cap = cv2.VideoCapture(lq_path)
        if not lq_cap.isOpened():
            gt_cap.release()
            logger = get_root_logger()
            logger.error(f"Failed to open LQ video: {lq_path}")
            raise RuntimeError(f"Cannot open video: {lq_path}")
        
        # Get frame count and ensure we don't go out of bounds
        frame_count = clip['frame_count']
        
        # Random interval first so we know the span
        interval = np.random.choice(self.interval_list)
        
        # Calculate how many frames we need
        frames_needed = (self.num_frame - 1) * interval + 1
        
        # Ensure we have enough frames
        if frame_count < frames_needed:
            # Not enough frames, use interval 1 and start at 0
            interval = 1
            start_frame = 0
            if frame_count < self.num_frame:
                # Still not enough frames - this clip is too short
                logger = get_root_logger()
                logger.warning(f"Clip {clip['name']} has only {frame_count} frames, need {self.num_frame}")
                start_frame = 0
        else:
            # Calculate valid range for start frame
            max_start = frame_count - frames_needed + 1
            start_frame = np.random.randint(0, max(1, max_start))
        
        # Get first frame to determine dimensions
        gt_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, first_gt = gt_cap.read()
        if ret:
            first_gt = cv2.cvtColor(first_gt, cv2.COLOR_BGR2RGB)
            gt_h, gt_w = first_gt.shape[:2]
        else:
            gt_h, gt_w = 720, 1280  # Default dimensions
            
        lq_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, first_lq = lq_cap.read()
        if ret:
            first_lq = cv2.cvtColor(first_lq, cv2.COLOR_BGR2RGB)
            lq_h, lq_w = first_lq.shape[:2]
        else:
            lq_h, lq_w = gt_h // 4, gt_w // 4  # Assume 4x scale
        
        # Calculate random crop position (same for all frames)
        if self.gt_size is not None:
            # Ensure crop size doesn't exceed frame size
            gt_size = min(self.gt_size, gt_h, gt_w)
            lq_size = gt_size // 4  # Assuming 4x scale
            
            # Random crop position
            gt_top = np.random.randint(0, gt_h - gt_size + 1) if gt_h > gt_size else 0
            gt_left = np.random.randint(0, gt_w - gt_size + 1) if gt_w > gt_size else 0
            lq_top = gt_top // 4
            lq_left = gt_left // 4
        else:
            # No cropping
            gt_size = min(gt_h, gt_w)
            lq_size = min(lq_h, lq_w)
            gt_top = gt_left = 0
            lq_top = lq_left = 0
        
        # Read and crop frames
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
                    gt_frame = np.zeros((gt_size, gt_size, 3), dtype=np.uint8)
            else:
                gt_frame = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2RGB)
                # Crop the frame
                gt_frame = gt_frame[gt_top:gt_top+gt_size, gt_left:gt_left+gt_size]
            
            # Read LQ frame
            lq_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, lq_frame = lq_cap.read()
            if not ret:
                # If we can't read, use last valid frame
                if img_lqs:
                    lq_frame = img_lqs[-1]
                else:
                    lq_frame = np.zeros((lq_size, lq_size, 3), dtype=np.uint8)
            else:
                lq_frame = cv2.cvtColor(lq_frame, cv2.COLOR_BGR2RGB)
                # Crop the frame
                lq_frame = lq_frame[lq_top:lq_top+lq_size, lq_left:lq_left+lq_size]
            
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
        
        # Remove timing code entirely - no need to log this anymore
        # load_time = time.time() - start_time
        
        return {
            'lq': img_lqs,
            'gt': img_gts,
            'key': f"{clip['name']}_{start_frame:08d}"
        }

    def __len__(self):
        """Dataset length is clips * possible starting positions."""
        # Return a large number to allow many iterations
        return len(self.clips) * 1000