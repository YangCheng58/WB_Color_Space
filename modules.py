import torch
import torch.nn as nn
import math
try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp, PatchEmbed

import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from timm.models.layers import DropPath

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x


class MambaVisionMixer(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=8,
            d_conv=3,
            expand=1,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        self.hs_proj = nn.Linear(
            self.d_inner // 2,
            self.dt_rank + self.d_state * 2,
            bias=False, **factory_kwargs
        )

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        self.conv1d_v = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

        self.conv1d_hs = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner // 2, **factory_kwargs,
        )

    def forward(self, xy):
        _, seqlen, _ = xy.shape

        xy = self.in_proj(xy)
        xy = rearrange(xy, "b l d -> b d l")

        v, hs = xy.chunk(2, dim=1)

        A = -torch.exp(self.A_log.float())

        v = F.silu(F.conv1d(
            input=v,
            weight=self.conv1d_v.weight,
            bias=self.conv1d_v.bias,
            padding='same',
            groups=self.d_inner // 2
        ))

        hs = F.silu(F.conv1d(
            input=hs,
            weight=self.conv1d_hs.weight,
            bias=self.conv1d_hs.bias,
            padding='same',
            groups=self.d_inner // 2
        ))

        x_dbl = self.hs_proj(rearrange(hs, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        hs = selective_scan_fn(
            hs,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None
        )

        z = torch.cat([v,hs], dim=1)
        z = rearrange(z, "b d l -> b l d")
        out = self.out_proj(z)

        return out


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False, 
    ):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, y):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        x = F.scaled_dot_product_attention(q,k,v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=1.,
                 qkv_bias=False,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 window_size=8
                 ):
        super().__init__()

        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        
        self.att_x = Attention(
            dim=int(dim/2),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.att_y = Attention(
            dim=int(dim / 2),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.mam = MambaVisionMixer(
            d_model=dim,
            d_state=8,
            d_conv=3,
            expand=1
        )
        
        self.norm2 = norm_layer(dim)
        self.norm3_x = norm_layer(int(dim/2))
        self.norm3_y = norm_layer(int(dim/2))
        self.norm4_x = norm_layer(int(dim/2))
        self.norm4_y = norm_layer(int(dim/2))

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_hidden_dim2 = int(dim/2 * mlp_ratio)

        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )
        self.mlp_x = Mlp_block(
            in_features=int(dim/2),
            hidden_features=mlp_hidden_dim2,
            act_layer=act_layer
        )
        self.mlp_y = Mlp_block(
            in_features=int(dim/2),
            hidden_features=mlp_hidden_dim2,
            act_layer=act_layer
        )
    

    def forward(self, x, y):
        B, _, H, W = x.shape
        xy = torch.cat([x, y], dim=1)
        xy = window_partition(xy, self.window_size)
        
        xy = xy + self.mam(self.norm1(xy))
        xy = xy + self.mlp(self.norm2(xy))
        
        x, y = xy.chunk(2, dim=2)

        x = x + self.att_x(self.norm3_x(x), self.norm3_y(y))
        x = x + self.mlp_x(self.norm4_x(x))
        y = y + self.att_y(self.norm3_y(y), self.norm3_x(x))
        y = y + self.mlp_y(self.norm4_y(y))
        
        xy = torch.concat([x, y], dim=2)
        xy = window_reverse(xy, window_size=self.window_size, H=H, W=W)
        x, y = xy.chunk(2, dim=1)

        return x, y

class NormDownsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale=0.5): 
        super(NormDownsample, self).__init__()
        self.prelu = nn.PReLU()
        
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),  
            nn.UpsamplingBilinear2d(scale_factor=scale)
        )

    def forward(self, x):
        x = self.down(x)
        x = self.prelu(x)
        return x


class NormUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2):  
        super(NormUpsample, self).__init__()
        self.prelu = nn.PReLU()

        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),   
            nn.UpsamplingBilinear2d(scale_factor=scale)
        )

        self.up = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)   
        )

    def forward(self, x, y):
        
        x = self.up_scale(x)
        x = torch.cat([x, y], dim=1)
        x = self.up(x)
        x = self.prelu(x)
        
        return x