import torch
import math

class NoiseScheduler:
    def __init__(self,
                 num_timesteps=1000,
                 schedule_type="linear",
                 beta_start=1e-4,
                 beta_end=0.02,
                 device='cpu'
                 ):
        self.device = device
        self.num_timesteps = num_timesteps

        if schedule_type == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, device=device
            )
            self.alphas = 1.0 - self.betas
            # cumulative product of alphas
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)


        elif schedule_type == "cosine":
            self.alpha_bars = self.__cosine_schedule(num_timesteps).to(device)
            alphas_prev = torch.cat([torch.ones(1, device=device), self.alpha_bars[:-1]])
            self.betas = torch.clamp(1.0 - (self.alpha_bars / alphas_prev), max=0.999)
            self.alphas = 1.0 - self.betas

        else:
            raise ValueError("Unknown schedule type")

        

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    
    def add_noise(self, x_0, noise, t):

        # Forward diffusion
        # x_t = sqrt(alphabar_t)* x_0 + sqrt(1-alphabar_t) * epsilon

        sqrt_ab_t = self.sqrt_alpha_bars[t]
        sqrt_one_minus_ab_t = self.sqrt_one_minus_alpha_bars[t]

        return sqrt_ab_t * x_0 + sqrt_one_minus_ab_t * noise

    def __cosine_schedule(self, timesteps, s=0.008):

        # t from 0 to T
        steps = torch.linspace(0, timesteps, timesteps + 1)
        
        # f(t) = cos^2( ((t/T + s)/(1+s)) * pi/2 )
        f_t = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        
        # alpha_bar_t = f(t) / f(0)
        alpha_bars = f_t / f_t[0]
        
        # return from t=1, because t=0 is the original image
        return alpha_bars[1:]
    