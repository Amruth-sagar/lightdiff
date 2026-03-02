import torch
from tqdm import tqdm



@torch.no_grad()
def ddim_sample(model, scheduler, shape, num_steps=50, num_images=1, random_seed=None):
    model.eval()
    device = scheduler.device

    if random_seed is not None:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    else:
        generator = None

    x = torch.randn((num_images, *shape), device=device, generator=generator)

    step_indices = torch.linspace(
        scheduler.num_timesteps - 1,
        0,
        num_steps
    ).long().to(device)

    batch_size = num_images

    for i in tqdm(range(len(step_indices) - 1)):

        t = torch.full(
            (batch_size,),
            step_indices[i],
            device=device,
            dtype=torch.long
        )

        t_next = torch.full(
            (batch_size,),
            step_indices[i+1],
            device=device,
            dtype=torch.long
        )

        eps = model(x, t)

        alpha_bar = scheduler.alpha_bars[t].view(-1,1,1,1)
        alpha_bar_next = scheduler.alpha_bars[t_next].view(-1,1,1,1)

        # x0 prediction
        x0_hat = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        
        # clamping anything outside -3*std and +3*std
        x0_hat = x0_hat.clamp(-3, 3)

        # deterministic DDIM update
        x = (
            torch.sqrt(alpha_bar_next) * x0_hat +
            torch.sqrt(1 - alpha_bar_next) * eps
        )

    return x

