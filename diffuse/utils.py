from PIL import Image

def image_grid(imgs, cols, resize=None):
    rows = len(imgs) // cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    if resize:
        grid = grid.resize((resize[0]//w * grid_w, resize[1]//h * grid_h))
    return grid
