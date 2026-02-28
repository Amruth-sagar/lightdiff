import torch.nn as nn
import lpips
import torch
import torch.nn.functional as F


class VAEloss(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.lpips_fn = lpips.LPIPS(net="vgg").to(device)
        self.lpips_fn.eval()

        for param in self.lpips_fn.parameters():
            param.requires_grad = False

    def forward(self, x, x_reconstructed, mu, logvar, beta, gamma):
        batch = x.shape[0]

        reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction="mean")

        # Lpips returns a tensor of shape (batch, 1, 1, 1)
        # NOTE: Make sure you are using 'tanh' as final act
        # to set the reconstructed pixel's range to [-1,1]
        # which is needed by LPIPS.
        perceptual_loss = self.lpips_fn(x, x_reconstructed).mean()

        logvar = torch.clamp(logvar, -15.0, 10.0)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        # mean over all latent dims.
        kl_loss = kl.mean()

        loss = (
            reconstruction_loss + (beta * kl_loss) + (gamma * perceptual_loss)
        )

        return {
            "total": loss,
            "mse": reconstruction_loss,
            "lpips": perceptual_loss,
            "kl": kl_loss,
        }

