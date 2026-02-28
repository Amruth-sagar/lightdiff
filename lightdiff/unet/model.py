import torch
import torch.nn as nn
import torch.nn.functional as F
from lightdiff.unet.blocks import SinusoidalPosEmb, ResBlock, Upsample, Downsample, AttentionBlock


class UNet(nn.Module):
    def __init__(self, unet_cfg):

        super().__init__()

        num_down_layers = unet_cfg['num_down_layers'] 
        num_up_layers = unet_cfg['num_up_layers'] 
        down_channel_args = unet_cfg['down_channel_args'] 
        up_channel_args = unet_cfg['up_channel_args'] 
        latent_channels = unet_cfg['latent_channels'] 
        num_res_blocks_per_level = unet_cfg['num_res_blocks_per_level'] 
        time_dim = unet_cfg['time_dim']

        assert num_down_layers == num_up_layers, "class UNet: Down and Up should have same number of layers for skip connections to work."
        assert down_channel_args == up_channel_args[::-1], "Down channel args should reflect Up channel args."

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        self.latent_to_unet_input = nn.Conv2d(
            in_channels=latent_channels,
            out_channels=down_channel_args[0],
            kernel_size=3,
            padding=1
        )

        down_layers = []
        skip_channels = []
        for i in range(num_down_layers):
            curr_channel = down_channel_args[i]
            next_channel = down_channel_args[i+1]
            for _ in range(num_res_blocks_per_level):
                down_layers.append(
                    ResBlock(
                        in_channels=curr_channel,
                        out_channels=next_channel,
                        time_dim=time_dim
                    )
                )
                skip_channels.append(next_channel)

                # Once the first res block bring channel count from
                # 'current_channel' to 'next_channel', rest of them
                # have same input and output channels set to 'next_channel'
                curr_channel = next_channel
            
            # A down-sampling (x1/2) after residual blocks
            down_layers.append(Downsample(next_channel))
        
        self.down = nn.ModuleList(down_layers)


        self.mid_block1 = ResBlock(down_channel_args[-1], down_channel_args[-1], time_dim)
        self.mid_attn = AttentionBlock(down_channel_args[-1])
        self.mid_block2 = ResBlock(down_channel_args[-1], down_channel_args[-1], time_dim)

        up_layers = []
        for i in range(num_up_layers):
            curr_channel = up_channel_args[i]
            next_channel = up_channel_args[i+1]
            
            # An up-sampling (x2) first, then res-blocks handle skips
            up_layers.append(Upsample(curr_channel))

            for _ in range(num_res_blocks_per_level):
                up_layers.append(
                    ResBlock(
                        in_channels=curr_channel + skip_channels.pop(),
                        out_channels=next_channel,
                        time_dim=time_dim
                    )
                )

                # Once the first res block bring channel count from
                # 'current_channel' to 'next_channel', rest of them
                # have same input and output channels set to 'next_channel'
                curr_channel = next_channel
            
        
        self.up = nn.ModuleList(up_layers)

        self.final_norm = nn.GroupNorm(32, up_channel_args[-1])
        self.final_conv = nn.Conv2d(up_channel_args[-1], latent_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        x = self.latent_to_unet_input(x)

        skips = []

        # Down part of UNet
        for layer in self.down:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
                skips.append(x)
            else:
                x = layer(x)

        # Middle part of UNet
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # Up part of UNet
        for layer in self.up:
            if isinstance(layer, ResBlock):
                skip_data = skips.pop()
                x = torch.cat([x, skip_data], dim=1)
                x = layer(x, t_emb)
            else:
                x = layer(x)
        
        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)

        return x