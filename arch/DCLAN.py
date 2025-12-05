import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from .modules import *
from .color_space import ColorSpace

class DCLAN(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 channels=[32, 32, 64, 128],  # Feature channels at each stage
                 heads=[8, 8, 16, 32],        # Attention heads at each stage
                 norm=False                   # Whether to use normalization
                 ):
        super(DCLAN, self).__init__()

        # Parse channel and head configurations
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        window_size = 8

        self.HS_conv1 = nn.Sequential(
            nn.ReplicationPad2d(1),  # Padding to maintain spatial dimensions
            nn.Conv2d(2, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.I_conv1 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )

        self.HS_down1 = NormDownsample(ch1, ch2)
        self.I_down1 = NormDownsample(ch1, ch2)

        self.HSI_block1 = Block(dim=ch2 * 2, num_heads=head2, window_size=window_size)

        self.HS_down2 = NormDownsample(ch2, ch3)
        self.I_down2 = NormDownsample(ch2, ch3)

        self.HSI_block2 = Block(dim=ch3 * 2, num_heads=head3, window_size=window_size)

        self.HS_down3 = NormDownsample(ch3, ch4)
        self.I_down3 = NormDownsample(ch3, ch4)

        self.HSI_block3 = Block(dim=ch4 * 2, num_heads=head4, window_size=window_size)
        self.HSI_block4 = Block(dim=ch4 * 2, num_heads=head4, window_size=window_size)

        self.HS_up1 = NormUpsample(ch4, ch3)
        self.I_up1 = NormUpsample(ch4, ch3)

        self.HSI_block5 = Block(dim=ch3 * 2, num_heads=head3, window_size=window_size)

        self.HS_up2 = NormUpsample(ch3, ch2)
        self.I_up2 = NormUpsample(ch3, ch2)

        self.HSI_block6 = Block(dim=ch2 * 2, num_heads=head4, window_size=window_size)

        self.HS_up3 = NormUpsample(ch2, ch1)
        self.I_up3 = NormUpsample(ch2, ch1)

        self.HS_conv2 = nn.Sequential(
            nn.ReplicationPad2d(1),  # Padding to maintain spatial dimensions
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )
        self.I_conv2 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False)
        )

        self.trans = ColorSpace()

    def forward(self, x):
        dtypes = x.dtype  # Save input dtype for consistency
        hsi = self.trans.rgb2hsv(x)

        hs = hsi[:, 1:3, :, :].to(dtypes)
        i = hsi.to(dtypes)

        hs_enc0 = self.HS_conv1(hs)  # Shape: (B, 32, 128, 128)
        i_enc0 = self.I_conv1(i)     # Shape: (B, 32, 128, 128)

        hs_enc1 = self.HS_down1(hs_enc0)  # Shape: (B, 32, 64, 64)
        i_enc1 = self.I_down1(i_enc0)     # Shape: (B, 32, 64, 64)

        i_jump0 = i_enc0   # Skip connection: (B, 32, 128, 128)
        hs_jump0 = hs_enc0 # Skip connection: (B, 32, 128, 128)

        i_enc2, hs_enc2 = self.HSI_block1(i_enc1, hs_enc1) # Shape: (B, 32, 64, 64)

        i_jump1 = i_enc2   # Skip connection: (B, 32, 64, 64)
        hs_jump1 = hs_enc2 # Skip connection: (B, 32, 64, 64)

        hs_enc2 = self.HS_down2(hs_enc2) # Shape: (B, 64, 32, 32)
        i_enc2 = self.I_down2(i_enc2)    # Shape: (B, 64, 32, 32)

        i_enc3, hs_enc3 = self.HSI_block2(i_enc2, hs_enc2) # Shape: (B, 64, 32, 32)

        # Save skip connection features
        i_jump2 = i_enc3   # Skip connection: (B, 64, 32, 32)
        hs_jump2 = hs_enc3 # Skip connection: (B, 64, 32, 32)

        hs_enc3 = self.HS_down3(hs_enc3) # Shape: (B, 128, 16, 16)
        i_enc3 = self.I_down3(i_enc3)    # Shape: (B, 128, 16, 16)

        i_enc4, hs_enc4 = self.HSI_block3(i_enc3, hs_enc3) # Shape: (B, 128, 16, 16)
        i_dec0, hs_dec0 = self.HSI_block4(i_enc4, hs_enc4) # Shape: (B, 128, 16, 16)
        
        hs_dec0 = self.HS_up1(hs_dec0, hs_jump2) # Shape: (B, 64, 32, 32)
        i_dec0 = self.I_up1(i_dec0, i_jump2)     # Shape: (B, 64, 32, 32)

        i_dec1, hs_dec1 = self.HSI_block5(i_dec0, hs_dec0) # Shape: (B, 64, 32, 32)

        i_dec1 = self.I_up2(i_dec1, i_jump1)
        hs_dec1 = self.HS_up2(hs_dec1, hs_jump1)

        i_dec2, hs_dec2 = self.HSI_block6(i_dec1, hs_dec1)

        i_dec2 = self.I_up3(i_dec2, i_jump0)
        hs_dec2 = self.HS_up3(hs_dec2, hs_jump0)
        
        hs = self.HS_conv2(hs_dec2)
        i = self.I_conv2(i_dec2)

        output_hsi = torch.cat([i, hs], dim=1) + hsi

        # HSV to RGB conversion: (b, 3, h, w) -> (b, 3, h, w) (Same dimensions as input)
        output_rgb, normal = self.trans.hsv2rgb(output_hsi)

        # Return restored/enhanced RGB image and normal vector info
        return output_rgb, normal
    
    def rgb2hsv(self, rgb):
        return self.trans.rgb2hsv(rgb)
