import torch
from torch import nn
from diffuse.noise_scheduler import NoiseScheduler

from diffuse.positional_embeddings import PositionalEmbedding


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
    ):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = len(self.time_mlp.layer) + len(self.input_mlp1.layer) + len(self.input_mlp2.layer)

        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb,x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x

    def sample(self, noise_scheduler: NoiseScheduler):  # noqa: F821
        s = noise_scheduler
        x_t = x_T = torch.randn(1000, 2)  # t
        traj = [x_T]
        for t in reversed(range(s.num_timesteps)):
            if t > 0:
                z = torch.randn((1000, 2))
            else:
                z = torch.zeros((1000, 2))

            # fmt: off
            pred = self(x_t,torch.tensor(t).repeat(1000,)).detach().cpu()
            x_t = 1/torch.sqrt(s.alphas[t]) * (x_t - (1 - s.alphas[t])/torch.sqrt(1 - s.alphas_cumprod[t]) * pred + torch.sqrt(s.betas[t]) * z)            
            # fmt: on
            traj.append(x_t)
        return traj