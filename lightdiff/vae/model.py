import torch
from lightdiff.vae.blocks import DownScale, UpScale, ToLatent, FromLatent
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, num_enc_blocks, channel_args, num_conv_layers_per_block, latent_channels):
        super().__init__()
        assert (num_enc_blocks == len(channel_args)-1) and (num_enc_blocks == len(num_conv_layers_per_block)), "class Encoder (VAE): mismatch in number of encoder layers and their args" 

        self.img_to_feat = nn.Conv2d(in_channels=3, out_channels=channel_args[0], kernel_size=3, padding=1)

        enc_layers = []
        for i in range(num_enc_blocks):
            enc_layers.append(
                DownScale(
                        in_channels=channel_args[i],
                        out_channels=channel_args[i+1],
                        num_layers=num_conv_layers_per_block[i]
                    )
            )

        self.encoder_layers = nn.ModuleList(enc_layers)
        self.feat_to_latent = ToLatent(
                in_channels=channel_args[-1],
                latent_channels=latent_channels)

    def forward(self, x):
        x = self.img_to_feat(x)
        for enc_layer in self.encoder_layers:
            x = enc_layer(x)
        mu, logvar = self.feat_to_latent(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, num_dec_blocks, channel_args, num_conv_layers_per_block, latent_channels):
        super().__init__()
        assert (num_dec_blocks == len(channel_args)-1) and (num_dec_blocks == len(num_conv_layers_per_block)), "class Decoder (VAE): mismatch in number of decoder layers and their args"

        self.latent_to_feat = FromLatent(latent_channels=latent_channels, out_channels=channel_args[0])

        dec_layers = []
        for i in range(num_dec_blocks):
            dec_layers.append(
                UpScale(
                    in_channels=channel_args[i],
                    out_channels=channel_args[i+1],
                    num_layers=num_conv_layers_per_block[i]
                )
            )

        self.decoder_layers = nn.ModuleList(dec_layers)
        self.feat_to_img = nn.Conv2d(in_channels=channel_args[-1], out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.latent_to_feat(x)
        for dec_layer in self.decoder_layers:
            x = dec_layer(x)
        x = self.feat_to_img(x)
        return x
    

class VAE(nn.Module):
    def __init__(self, VAE_CFG):
        super().__init__()

        # Here, num_enc_
        num_enc_blocks = VAE_CFG["num_enc_blocks"]
        num_dec_blocks = VAE_CFG["num_dec_blocks"] 
        enc_channel_args = VAE_CFG["enc_channel_args"]
        dec_channel_args = VAE_CFG["dec_channel_args"]
        num_conv_per_enc_block = VAE_CFG["num_conv_per_enc_block"]
        num_conv_per_dec_block = VAE_CFG["num_conv_per_dec_block"]
        latent_channels = VAE_CFG["latent_channels"]
        final_activation = VAE_CFG["final_activation"]

        self.encoder = Encoder(
            num_enc_blocks=num_enc_blocks,
            channel_args=enc_channel_args,
            num_conv_layers_per_block=num_conv_per_enc_block,
            latent_channels=latent_channels,
        )

        self.decoder = Decoder(
            num_dec_blocks=num_dec_blocks,
            channel_args=dec_channel_args,
            num_conv_layers_per_block=num_conv_per_dec_block,
            latent_channels=latent_channels
        )

        # we bring back the output
        # to a range we want using activation

        # For range [0, 1]
        if final_activation == "sigmoid":
            self.final_act = nn.Sigmoid()
        
        # for range [-1, 1]
        elif final_activation == "tanh":
            self.final_act = nn.Tanh()
        else:
            self.final_act = nn.Identity()

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + (eps * std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        x = self.final_act(x)
        return {
            "reconstructed":x,
            "mu":mu,
            "logvar":logvar
        }
    def decode(self, z):
        x = self.decoder(z)
        x = self.final_act(x)
        return x


