import os
import torch
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from config import get_config, print_usage
from utils import transform_to_mpimg, image2tensor
from generator import Generator
from discriminator import Discriminator


def discriminator_loss(mask, mask_o, mask_r, output_r, output_o,
                       lambda_loss):
    zeros_mask = torch.zeros(mask_r.shape)
    if torch.cuda.is_available():
        zeros_mask = zeros_mask.cuda()
    criterion = torch.nn.MSELoss()
    l_map = criterion(mask, mask_o) + \
        criterion(mask_r, zeros_mask)
    print("Loss L_map: {}".format(l_map))
    entropy_loss = -torch.log(output_r) - torch.log(1 - output_o)
    print("Loss entropy: {}".format(entropy_loss))
    entropy_loss = torch.mean(entropy_loss)
    print("Loss entropy: {}".format(entropy_loss))
    loss_d = entropy_loss + lambda_loss * l_map
    return loss_d

if __name__ == '__main__':
    config, unparsed = get_config()
    if len(unparsed) > 0:
        print_usage()
        exit(1)
    # Generate the image
    model_gen = Generator((1, 3, 480, 720), config)
    model_discriminator = Discriminator()
    if torch.cuda.is_available():
        model_gen = model_gen.cuda()
        model_discriminator = model_discriminator.cuda()
    # for name, param in model_gen.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # for name, param in model_discriminator.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # exit(0)
    optim_g = optim.Adam(
        model_gen.parameters(),
        lr=config.learning_rate
        )
    optim_d = optim.Adam(
        model_discriminator.parameters(),
        lr=config.learning_rate
        )
    inp_list = sorted(os.listdir(config.data_dir))
    label_list = sorted(os.listdir(config.label_dir))

    for epoch in range(config.num_epoch):
        prefix = "Training Epoch {:3d}: ".format(epoch)

        for i in tqdm(range(len(inp_list))):
            img = mpimg.imread(config.data_dir + inp_list[i])
            img = image2tensor(img)
            label_img = mpimg.imread(config.label_dir + label_list[i])
            label_img = image2tensor(label_img)
            label_img.requires_grad = False
            
            if torch.cuda.is_available():
                img = img.cuda()
                label_img = label_img.cuda()

            # Train generator
            loss_gen, masks, f1, f2, x = model_gen.forward(img, label_img)
            mask_o, output_o = model_discriminator.forward(x)
            loss_g = 0.01 * torch.log(1 - output_o) + loss_gen
            print("Loss G: {}".format(loss_g))
            loss_g.backward()
            optim_g.step()
            optim_g.zero_grad()

            # Train discriminator
            mask_o, output_o = model_discriminator.forward(x.detach())
            mask_r, output_r = model_discriminator.forward(label_img)
            loss_d = discriminator_loss(
                masks[-1].detach(), mask_o, mask_r,
                output_r, output_o, config.lambda_l
                )
            print("Loss D: {}".format(loss_d))
            loss_d.backward()
            optim_d.step()
            optim_d.zero_grad()
        # x = x.detach().numpy()
        # x = x.squeeze(0).transpose(1, 2, 0).astype(np.uint8)
