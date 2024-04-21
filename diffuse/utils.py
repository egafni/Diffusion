import math
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage
import numpy as np




def pil_image_grid(imgs: list[Image] | list[np.array], cols: int, resize=None, max_n=None):
    if isinstance(imgs[0], np.ndarray):
        imgs = [ToPILImage()(img) for img in list(imgs)]

    if resize:
        imgs = [img.resize(resize, resample=Image.NEAREST) for img in imgs]
    if max_n:
        # skip images but make sure to keep the last
        last = imgs[-1]
        imgs = imgs[::len(imgs) // (max_n - 2)]
        if last != imgs[-1:]: # add last image if not already there
            imgs += [last]

    rows = math.ceil(len(imgs) / cols)

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid
