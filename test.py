import os
import torch
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from config import get_config, print_usage
import utils
from generator import Generator
import os


if __name__ == '__main__':
    config, unparsed = get_config()
    if len(unparsed) > 0:
        print_usage()
        exit(1)
    # Generate images
    model = Generator(config)
    weights_file = config.weights_dir
    if os.path.exists(weights_file):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        load_res = torch.load(weights_file, map_location=device)
        model.load_state_dict(load_res)
    model.eval()
    inp_list = sorted(os.listdir(config.data_dir))
    for i in range(len(inp_list)):
        img = mpimg.imread(config.data_dir + inp_list[i])
        print(img.shape)
        img = utils.image2tensor(img)
        masks, f1, f2, x = model.forward(img)
        mask = utils.transform_to_mpimg(masks[-1])
        mask = mask.squeeze(2)
        utils.show_img(mask)
        exit(0)
        # x = x.detach().numpy()
        # x = x.squeeze(0).transpose(1, 2, 0).astype(np.uint8)
