import torch
from basicsr.utils.registry import MODEL_REGISTRY
from .recurrent_mixprecision_model import RecurrentMixPrecisionRTModel

@MODEL_REGISTRY.register()
class RecurrentMixPrecisionRTModelDebug(RecurrentMixPrecisionRTModel):
    """Debug version with extra logging"""
    
    def feed_data(self, data):
        """Override feed_data with debug logging"""
        print(f"DEBUG feed_data: Starting, device = {self.device}")
        print(f"DEBUG feed_data: lq shape = {data['lq'].shape}, device = {data['lq'].device}")
        print(f"DEBUG feed_data: gt shape = {data['gt'].shape}, device = {data['gt'].device}")
        
        # Move data to device
        print(f"DEBUG feed_data: Moving lq to {self.device}...")
        self.lq = data['lq'].to(self.device)
        print(f"DEBUG feed_data: lq moved successfully")
        
        if 'gt' in data:
            print(f"DEBUG feed_data: Moving gt to {self.device}...")
            self.gt = data['gt'].to(self.device)
            print(f"DEBUG feed_data: gt moved successfully")
        
        print(f"DEBUG feed_data: Complete")