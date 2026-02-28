import torch
from tqdm import tqdm


@torch.no_grad()
def ddim_sample(model, scheduler, shape, num_steps=50):
    model.eval()
    device = scheduler.device
    x = torch.randn(shape, device=device)

    step_indices = torch.linspace(
        scheduler.num_timesteps -1,
        0,
        num_steps
    ).long().to(device)

    for i in tqdm(range(len(step_indices) - 1)):
        t = step_indices[i].unsqueeze(0)
        t_next = step_indices[i+1].unsqueeze(0)

        eps = model(x, t)

        alpha_bar = scheduler.alpha_bars[t]
        alpha_bar_next = scheduler.alpha_bars[t_next]

        # estimating x_0
        x0_hat = (
            x - torch.sqrt(1-alpha_bar) * eps
        ) / torch.sqrt(alpha_bar)

        # deterministic update
        x = (
            torch.sqrt(alpha_bar_next) * x0_hat +
            torch.sqrt(1-alpha_bar_next) * eps
        )


    return x

