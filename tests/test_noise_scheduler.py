import torch

from diffuse.noise_scheduler import NoiseScheduler


def test_noise_scheduler():
    torch.manual_seed(2)
    ns = NoiseScheduler(0.0001, 0.02, 100, beta_schedule='linear')

    z = ns.add_noise(x=torch.randn(10, 3, 28, 28),
                     t=torch.randint(0, 100, (10,)),
                     noise=torch.randn(10, 3, 28, 28))

    assert z.shape == (10, 3, 28, 28)
    assert z.mean() == 0.0011

    

