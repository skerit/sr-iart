import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from basicsr.utils.registry import ARCH_REGISTRY
from .iart_arch import IART
from integrate_dusa_iart import replace_window_attention_with_dusa

@ARCH_REGISTRY.register()
class DuSAIART(IART):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        replace_window_attention_with_dusa(self)