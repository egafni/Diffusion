import torch
import torch.nn.functional as F
import math
from jaxtyping import Float
from torch import Tensor


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class NoiseScheduler():
    def __init__(self,
                 num_timesteps,  # =1000,
                 beta_start,  # =0.0001,
                 beta_end,  # =0.02,
                 beta_schedule):  # ="linear"):

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "log2":
            self.betas = torch.logspace(math.log2(beta_start), math.log2(beta_end), num_timesteps, base=2)
        else:
            raise ValueError()

        self.alphas = 1.0 - self.betas
        # alpha_cumprod = alpha_hat
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)

        # required for q_posterior (two terms of eq 7)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                1. - self.alphas_cumprod)

    # def reconstruct_x0(self, x_t, t, noise):
    #     s1 = self.sqrt_inv_alphas_cumprod[t]
    #     s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
    #     s1 = s1.reshape(-1, 1)
    #     s2 = s2.reshape(-1, 1)
    #     return s1 * x_t - s2 * noise

    # def q_posterior(self, x_0, x_t, t):
    #     # eq7 u_t_tilde(x_t,x_0)
    #     s1 = self.posterior_mean_coef1[t]
    #     s2 = self.posterior_mean_coef2[t]
    #     s1 = s1.reshape(-1, 1)
    #     s2 = s2.reshape(-1, 1)
    #     mu = s1 * x_0 + s2 * x_t
    #     return mu

    # def get_variance(self, t):
    #     if t == 0:
    #         return 0

    #     # eq 7, Bt_tilde
    #     variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
    #     variance = variance.clip(1e-20)
    #     return variance

    # def step(self, model_output, timestep, sample):
    #     t = timestep
    #     pred_original_sample = self.reconstruct_x0(sample, t, model_output)
    #     pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

    #     variance = 0
    #     if t > 0:
    #         noise = torch.randn_like(model_output)
    #         variance = (self.get_variance(t) ** 0.5) * noise

    #     pred_prev_sample = pred_prev_sample + variance

    #     return pred_prev_sample

    def add_noise(self, x_start: Float[Tensor, "B H W"], x_noise: Float[Tensor, "B H W"],
                  timesteps: Float[Tensor, "B"]):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        for dims in range(x_start.dim() - 1):  # handle any number of dimensions
            s1 = s1.unsqueeze(-1)  # (B,1,1)
            s2 = s2.unsqueeze(-1)  # (B,1,1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps

    def sample(self, model, shape):  # noqa: F821
        device = next(model.parameters())[0].device
        x_t = x_T = torch.randn(shape)  # t
        traj = [x_T]
        for t in reversed(range(self.num_timesteps)):
            if t > 0:
                z = torch.randn(shape)
            else:
                z = torch.zeros(shape)

            # fmt: off
            pred = model(x_t.to(device), torch.tensor(t).repeat(shape[0], ).to(device)).detach().cpu()
            x_t = 1 / torch.sqrt(self.alphas[t]) * (
                    x_t - (1 - self.alphas[t]) / torch.sqrt(1 - self.alphas_cumprod[t]) * pred + torch.sqrt(
                self.betas[t]) * z)
            # fmt: on
            traj.append(x_t)
        return traj
