import os
from dataclasses import dataclass

import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffuse.diffusion_utilities import *
from diffuse.diffusion_utilities import unorm
from diffuse.models.context_unet import ContextUnet
from diffuse.noise_scheduler import NoiseScheduler


@dataclass
class TrainConfig:
    save_dir: str

    # noise scheduler parameters
    timesteps: int = 500
    beta1: float = 1e-4
    beta2: float = 0.02

    # network hyperparameters
    device: str = "cuda:0" if torch.cuda.is_available() else 'cpu'
    n_feat: int = 64  # 64 hidden dimension feature
    n_cfeat: int = 5  # context vector is of size 5
    height: int = 16  # 16x16 image

    # training hyperparameters
    batch_size: int = 128
    n_epoch: int = 32
    lrate: float = 1e-3


def train(config: TrainConfig):
    c = config

    # diffusion hyperparameters
    s = NoiseScheduler(n_timesteps=c.timesteps, beta1=c.beta1, beta2=c.beta2)

    # construct model
    nn_model = ContextUnet(in_channels=3, n_feat=c.n_feat, n_cfeat=c.n_cfeat, height=c.height).to(c.device)

    # load dataset and construct optimizer
    dataset = CustomDataset("../data/sprites_1788_16x16.npy",
                            "../data/sprite_labels_nc_1788_16x16.npy",
                            transform,
                            null_context=False)
    dataloader = DataLoader(dataset, batch_size=c.batch_size, shuffle=True, num_workers=1)
    optim = torch.optim.Adam(nn_model.parameters(), lr=c.lrate)

    # set into train mode
    nn_model.train()

    for ep in range(c.n_epoch):
        print(f'epoch {ep}')

        # linearly decay learning rate
        optim.param_groups[0]['lr'] = c.lrate * (1 - ep / c.n_epoch)

        pbar = tqdm(dataloader, mininterval=2)
        for x, _ in pbar:  # x: images
            optim.zero_grad()
            x = x.to(c.device)

            # perturb data
            noise = torch.randn_like(x)
            t = torch.randint(1, c.timesteps, (x.shape[0],)).to(c.device)
            x_pert = s.add_noise(x, t, noise)

            # use network to recover noise
            pred_noise = nn_model(x_pert, t / c.timesteps)

            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            optim.step()

        # save model periodically
        if ep % 4 == 0 or ep == int(c.n_epoch - 1):
            if not os.path.exists(c.save_dir):
                os.mkdir(c.save_dir)
            torch.save(nn_model.state_dict(), c.save_dir + f"model_{ep}.pth")
            print('saved model at ' + c.save_dir + f"model_{ep}.pth")


def get_images(fname, nn_model, x, noise_scheduler: NoiseScheduler, save_dir, device, ):
    # load in model weights and set to eval mode
    nn_model.load_state_dict(torch.load(f"{save_dir}/{fname}", map_location=device))
    nn_model.eval()
    shape = (32,) + tuple(x.shape[1:])
    traj = noise_scheduler.sample(nn_model, shape).numpy()
    z = unorm(rearrange(traj, 't b c h w -> b t h w c'))
    return z
