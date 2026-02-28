import torch.nn as nn
import torch
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base

        half_dim = self.dim//2

        freq = torch.exp(
            -torch.log(torch.tensor(self.base)) *
            torch.arange(half_dim).float() / (half_dim - 1)
        )

        self.register_buffer(
            "freq", freq
        )

    def forward(self, t):
        # t = (batch,)
        args = t[:, None].float() * self.freq[None, :]

        # Ordering of sin/cos components is irrelevant since 
        # a following MLP layer absorbs any permutation.
        
        # emb.shape = (batch, dim)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        
        return emb
        

# a ResNet-style UNet

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        num_groups_in = min(32, in_channels // 4)
        num_groups_out = min(32, out_channels//4)
        
        self.norm1 = nn.GroupNorm(num_groups_in, in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups_out, out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
            
        # FiLM
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2)   # half (out_channesl) is for scale, and other half is for shift
        )

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        t_out = self.time_proj(t_emb)
        t_out = t_out[:, :, None, None]

        # Normalization before scale and shift
        h = self.norm2(h)
        
        scale, shift = torch.chunk(t_out, 2, dim=1)
        h = h * (1+scale) + shift

        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)
    

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )

    def forward(self, x):
        return self.conv(x)
    

# To avoid checkboard artifacts
class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=3, 
            padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()

        num_groups = min(32, channels // 4)
        
        self.norm = nn.GroupNorm(num_groups, channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        batch, channels, H, W = x.shape
        h = self.norm(x)

        # h --> (batch, HW, channels)
        h = h.view(batch, channels, H*W).permute(0, 2, 1)
        h, _ = self.attention(h, h, h)

        h = h.permute(0, 2, 1).view(batch, channels, H, W)

        return x + h
    


