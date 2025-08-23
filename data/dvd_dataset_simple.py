"""
Simplified DVD dataset that only overrides the frame boundary checking
"""
import random
from basicsr.data.reds_dataset import REDSRecurrentDataset
from basicsr.utils import get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class DVDRecurrentDatasetSimple(REDSRecurrentDataset):
    """DVD dataset that handles variable frame counts per clip.
    
    Only overrides the frame selection logic to handle variable clip lengths.
    """
    
    def __init__(self, opt):
        super().__init__(opt)
        
        # Build a dictionary of max frame indices for each clip
        self.clip_frame_counts = {}
        for key in self.keys:
            clip_name, frame_name = key.split('/')
            frame_idx = int(frame_name)
            if clip_name not in self.clip_frame_counts:
                self.clip_frame_counts[clip_name] = 0
            self.clip_frame_counts[clip_name] = max(self.clip_frame_counts[clip_name], frame_idx + 1)
        
        logger = get_root_logger()
        logger.info(f'DVD dataset: Found {len(self.clip_frame_counts)} clips')
        for clip, count in list(self.clip_frame_counts.items())[:5]:
            logger.info(f'  {clip}: {count} frames')
    
    def __getitem__(self, index):
        """Override to handle variable frame counts."""
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000
        
        # Get the actual frame count for this clip
        clip_frame_count = self.clip_frame_counts[clip_name]
        
        # determine the neighboring frames
        interval = random.choice(self.interval_list)
        
        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        max_possible_start = max(0, clip_frame_count - self.num_frame * interval)
        
        if start_frame_idx > max_possible_start:
            start_frame_idx = random.randint(0, max_possible_start)
        
        end_frame_idx = start_frame_idx + self.num_frame * interval
        
        # Ensure we don't exceed clip boundaries
        if end_frame_idx > clip_frame_count:
            start_frame_idx = max(0, clip_frame_count - self.num_frame * interval)
            end_frame_idx = start_frame_idx + self.num_frame * interval
        
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))
        
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # Now call the parent class's frame loading logic
        # We'll need to reconstruct this part
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
            img_lq = self.imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = self.imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = self.paired_random_crop(img_gts, img_lqs, gt_size, scale)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = self.augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = self.img2tensor(img_results)
        img_gts = torch.stack(img_results[:self.num_frame], dim=0)
        img_lqs = torch.stack(img_results[self.num_frame:], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}