import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2):
        super().__init__()

        assert (
            num_layers >= 1
        ), "class ConvBlock: Number of conv layers should atleast be 1"

        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.GroupNorm(8, out_channels))
            layers.append(nn.SiLU())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DownScale(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels, num_layers)
        self.downsample = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.downsample(x)
        return x


class UpScale(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.conv = ConvBlock(out_channels, out_channels, num_layers)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class ToLatent(nn.Module):
    def __init__(self, in_channels, latent_channels):
        super().__init__()

        self.mu = nn.Conv2d(in_channels, latent_channels, kernel_size=1)
        self.logvar = nn.Conv2d(in_channels, latent_channels, kernel_size=1)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


class FromLatent(nn.Module):
    def __init__(self, latent_channels, out_channels):
        super().__init__()

        self.proj = nn.Conv2d(latent_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.proj(x)
