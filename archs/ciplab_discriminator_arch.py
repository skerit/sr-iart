"""
CIPLAB U-Net Discriminator with Feature Matching support
From NTIRE 2020 winning solution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class DBlock(nn.Module):
    """Residual block for the discriminator"""
    def __init__(self, in_channels, out_channels, which_conv=nn.Conv2d, which_bn=nn.BatchNorm2d, wide=True,
                preactivation=True, activation=nn.LeakyReLU(0.1, inplace=False), downsample=nn.AvgPool2d(2, stride=2)):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample
            
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels, kernel_size=3, padding=1)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, 
                                            kernel_size=1, padding=0)

        self.bn1 = self.which_bn(self.hidden_channels)
        self.bn2 = self.which_bn(out_channels)
        
    def forward(self, x):
        if self.preactivation:
            h = self.activation(x)
        else:
            h = x    
        h = self.bn1(self.conv1(h))
        if self.downsample:
            h = self.downsample(h)     
            
        return h


class GBlock(nn.Module):
    """Upsampling block for the discriminator decoder"""
    def __init__(self, in_channels, out_channels,
                which_conv=nn.Conv2d, which_bn=nn.BatchNorm2d, activation=nn.LeakyReLU(0.1, inplace=False), 
                upsample=nn.Upsample(scale_factor=2, mode='nearest')):
        super(GBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, 
                                            kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(out_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x):
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
        h = self.bn1(self.conv1(h))
        return h


@ARCH_REGISTRY.register()
class CIPLABUnetD(nn.Module):
    """CIPLAB U-Net Discriminator with Feature Matching support
    
    Returns discriminator scores and intermediate features for feature matching loss.
    Based on NTIRE 2020 winning solution.
    """
    def __init__(self):
        super(CIPLABUnetD, self).__init__()

        # Encoder path
        self.enc_b1 = DBlock(3, 64, preactivation=False)
        self.enc_b2 = DBlock(64, 128)
        self.enc_b3 = DBlock(128, 192)
        self.enc_b4 = DBlock(192, 256)
        self.enc_b5 = DBlock(256, 320)
        self.enc_b6 = DBlock(320, 384)

        self.enc_out = nn.Conv2d(384, 1, kernel_size=1, padding=0)

        # Decoder path
        self.dec_b1 = GBlock(384, 320)
        self.dec_b2 = GBlock(320*2, 256)
        self.dec_b3 = GBlock(256*2, 192)
        self.dec_b4 = GBlock(192*2, 128)
        self.dec_b5 = GBlock(128*2, 64)
        self.dec_b6 = GBlock(64*2, 32)

        self.dec_out = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        """Forward pass returning scores for CIPLAB-style GAN loss
        
        Args:
            x: Input image tensor
            
        Returns:
            e_out: Encoder discriminator score
            d_out: Decoder discriminator score
        """
        # Encoder
        e1 = self.enc_b1(x)
        e2 = self.enc_b2(e1)
        e3 = self.enc_b3(e2)
        e4 = self.enc_b4(e3)
        e5 = self.enc_b5(e4)
        e6 = self.enc_b6(e5)

        e_out = self.enc_out(F.leaky_relu(e6, 0.1))

        # Decoder with skip connections
        d1 = self.dec_b1(e6)
        d2 = self.dec_b2(torch.cat([d1, e5], 1))
        d3 = self.dec_b3(torch.cat([d2, e4], 1))
        d4 = self.dec_b4(torch.cat([d3, e3], 1))
        d5 = self.dec_b5(torch.cat([d4, e2], 1))
        d6 = self.dec_b6(torch.cat([d5, e1], 1))

        d_out = self.dec_out(F.leaky_relu(d6, 0.1))

        # Return both outputs separately for CIPLAB-style loss computation
        return e_out, d_out
    
    def forward_with_features(self, x):
        """Forward pass with features for the training loop that expects this method
        
        Returns discriminator scores and features for feature matching.
        """
        # Encoder
        e1 = self.enc_b1(x)
        e2 = self.enc_b2(e1)
        e3 = self.enc_b3(e2)
        e4 = self.enc_b4(e3)
        e5 = self.enc_b5(e4)
        e6 = self.enc_b6(e5)

        e_out = self.enc_out(F.leaky_relu(e6, 0.1))

        # Decoder with skip connections
        d1 = self.dec_b1(e6)
        d2 = self.dec_b2(torch.cat([d1, e5], 1))
        d3 = self.dec_b3(torch.cat([d2, e4], 1))
        d4 = self.dec_b4(torch.cat([d3, e3], 1))
        d5 = self.dec_b5(torch.cat([d4, e2], 1))
        d6 = self.dec_b6(torch.cat([d5, e1], 1))

        d_out = self.dec_out(F.leaky_relu(d6, 0.1))

        # Return both encoder and decoder outputs separately, plus features
        # This matches CIPLAB's exact implementation
        return e_out, d_out, [e1, e2, e3, e4, e5, e6], [d1, d2, d3, d4, d5, d6]