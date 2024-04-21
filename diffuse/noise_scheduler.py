import numpy as np
import torch
import torch.nn.functional as F
import math
from jaxtyping import Float
from torch import Tensor


# def cosine_beta_schedule(timesteps, s=0.008):
#     """
#     cosine schedule as proposed in https://arxiv.org/abs/2102.09672
#     """
#     steps = timesteps + 1
#     x = torch.linspace(0, timesteps, steps)
#     alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     return torch.clip(betas, 0.0001, 0.9999)
#
#
# def linear_beta_schedule(timesteps):
#     beta_start = 0.0001
#     beta_end = 0.02
#     return torch.linspace(beta_start, beta_end, timesteps)
#
#
# def quadratic_beta_schedule(timesteps):
#     beta_start = 0.0001
#     beta_end = 0.02
#     return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2
#
#
# def sigmoid_beta_schedule(timesteps):
#     beta_start = 0.0001
#     beta_end = 0.02
#     betas = torch.linspace(-6, 6, timesteps)
#     return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class NoiseScheduler:
    def __init__(self, beta1, beta2, n_timesteps, beta_schedule='linear'):
        if beta_schedule == "linear":
            b_t = torch.linspace(beta1, beta2, n_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            b_t = torch.linspace(beta1 ** 0.5, beta2 ** 0.5, n_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "log2":
            b_t = torch.logspace(math.log2(beta1), math.log2(beta2), n_timesteps, base=2)
        else:
            raise ValueError()

        a_t = 1 - b_t
        ab_t = torch.cumsum(a_t.log(), dim=0).exp()
        self.b_t, self.a_t, self.ab_t, self.n_timesteps = b_t, a_t, ab_t, n_timesteps

    def __len__(self):
        return self.n_timesteps

    def add_noise(self, x, t, noise):
        # move to correct device
        device = x.device
        t = t.to(device)
        noise = noise.to(device)
        ab_t = self.ab_t.to(device)

        def unsqueeze_n(z):
            if x.ndim == 2:
                return z[:, None]
            elif x.ndim == 3:
                return z[:, None, None]
            elif x.ndim == 4:
                return z[:, None, None, None]

        return unsqueeze_n(ab_t.sqrt()[t]) * x + (1 - unsqueeze_n(ab_t[t])) * noise

    @torch.no_grad()
    def sample(self, model, data_shape):
        a_t, ab_t, b_t, n_timesteps = self.a_t, self.ab_t, self.b_t, self.n_timesteps

        device = next(model.parameters())[0].device
        x_t = x_T = torch.randn(data_shape).to(device)
        traj = [x_T]
        for t in range(n_timesteps - 1, -1, -1):
            if t % 10 == 0:
                print('.', end='')
            # don't add any noise back in if x0
            z = torch.randn(data_shape) if t > 0 else torch.zeros(data_shape)
            z = z.to(device)

            # fmt: off
            t_arr = torch.tensor(t).repeat(data_shape[0], ) / n_timesteps
            x_hat = model(x_t.to(device), t_arr.to(device))

            c = 1 / torch.sqrt(a_t[t])
            noise = torch.sqrt(b_t[t]) * z
            x_t = c * (x_t - (1 - a_t[t]) / torch.sqrt(1 - ab_t[t]) * x_hat + noise)

            assert not torch.isinf(x_t).any()
            # fmt: on
            traj.append(x_t)

        traj = torch.stack(traj).detach().cpu()
        assert not traj.isinf().any()
        return traj
