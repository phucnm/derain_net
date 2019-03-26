import os
import torch
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from config import get_config, print_usage
from utils import transform_to_mpimg
from generator import Generator


if __name__ == '__main__':
    config, unparsed = get_config()
    if len(unparsed) > 0:
        print_usage()
        exit(1)
    # Generate images
    model = Generator((1, 3, 480, 720), config)
    inp_list = sorted(os.listdir(config.data_dir))
    for i in range(len(inp_list)):
        img = mpimg.imread(config.data_dir + inp_list[i])
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
        losses, masks, f1, f2, x = model.forward(img, img)

        # x = x.detach().numpy()
        # x = x.squeeze(0).transpose(1, 2, 0).astype(np.uint8)
