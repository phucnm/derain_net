import os
import torch
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from config import get_config, print_usage
import utils
# from utils import transform_to_mpimg, image2tensor, binary_mask, resize
# from generator import Generator
from discriminator import Discriminator
from generator_ori import Generator
from vgg16 import VGG16
from pytorch_modelsize import SizeEstimator


def attentive_loss(rain_img, label_img, mask_list, config):
    loss_att = 0
    bin_mask = utils.binary_mask(rain_img, label_img).detach_()
    for i in range(config.num_time_step):
        mse_loss = F.mse_loss(bin_mask, mask_list[i] + rain_img)
        loss_att += np.power(config.theta, config.num_time_step - i + 1) \
                    * mse_loss
    return loss_att


def multiscale_loss(decoded_imgs, label_img):
    lambdas = [0.6, 0.8, 1]
    scales = [0.25, 0.5, 1]
    loss_m = 0
    for i, img in enumerate(decoded_imgs):
        scaled_img = utils.resize(label_img, scales[i]).detach_()
        loss_m += lambdas[i] * F.mse_loss(scaled_img, img)
    return loss_m


def perceptual_loss(decoded_img, label_img, vgg16):
    if vgg16 is None:
        return 0
    outputs = vgg16.forward(decoded_img)
    outputs2 = vgg16.forward(label_img)
    losses = []
    for i, out in enumerate(outputs):
        l = F.mse_loss(out, outputs2[i])
        losses.append(l)
    l_p = torch.FloatTensor(losses).mean()

    return l_p


def generator_loss(rain_img, label_img, mask_list, frame1, frame2, x, config, vgg16=None):
    att_loss = attentive_loss(rain_img, label_img, mask_list, config)
    m_loss = multiscale_loss([frame1, frame2, x], label_img)
    p_loss = perceptual_loss(x, label_img, vgg16)
    # print(att_loss)
    # print(m_loss)
    # print(p_loss)
    return att_loss + m_loss + p_loss


def discriminator_loss(mask, mask_o, mask_r, output_r, output_o,
                       lambda_loss):
    zeros_mask = torch.zeros(mask_r.shape)
    if torch.cuda.is_available():
        zeros_mask = zeros_mask.cuda()
    criterion = torch.nn.MSELoss()
    l_map = criterion(mask, mask_o) + \
        criterion(mask_r, zeros_mask)
    # print("Loss L_map: {}".format(l_map))
    entropy_loss = -torch.log(output_r) - torch.log(1 - output_o)
    entropy_loss = torch.mean(entropy_loss)
    # print("Loss entropy: {}".format(entropy_loss))
    loss_d = entropy_loss + lambda_loss * l_map
    return loss_d

if __name__ == '__main__':
    config, unparsed = get_config()
    if len(unparsed) > 0:
        print_usage()
        exit(1)
    # Generate the image
    # vgg16 = VGG16()
    model_gen = Generator(config)
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

    label_real = torch.ones(1, 1)
    label_fake = torch.zeros(1, 1)
    criterion = torch.nn.MSELoss()

    if torch.cuda.is_available():
        label_real = label_real.cuda()
        label_fake = label_fake.cuda()

    for epoch in range(config.num_epoch):
        prefix = "Training Epoch {:3d}: ".format(epoch)

        for i in tqdm(range(len(inp_list))):
            img = mpimg.imread(config.data_dir + inp_list[i])
            img = utils.image2tensor(img)
            label_img = mpimg.imread(config.label_dir + label_list[i])
            label_img = utils.image2tensor(label_img)

            if torch.cuda.is_available():
                img = img.cuda()
                label_img = label_img.cuda()

            # Train Discriminator
            optim_d.zero_grad()
            mask_r, D_real = model_discriminator.forward(label_img)
            masks, f1, f2, x = model_gen.forward(img)
            mask_f, D_fake = model_discriminator.forward(x)
            D_loss = criterion(D_real, label_real) + criterion(D_fake, label_fake)
            D_loss = discriminator_loss(
                masks[-1], mask_f, mask_r,
                D_real, D_fake, config.gamma
            )
            print("Loss D: {}".format(D_loss))
            D_loss.backward()
            optim_d.step()

            # Train generator
            optim_g.zero_grad()
            masks, f1, f2, x = model_gen.forward(img)
            mask_f, D_fake = model_discriminator.forward(x)
            gen_loss = generator_loss(
                img, label_img, masks,
                f1, f2, x, config
            )
            G_loss = 0.01 * criterion(D_fake, label_real) + gen_loss
            # G_loss = 0.01 * torch.log(1 - D_fake) + loss_gen
            print("Loss G: {}".format(G_loss))
            G_loss.backward()
            optim_g.step()
            optim_g.zero_grad()
        # x = x.detach().numpy()
        # x = x.squeeze(0).transpose(1, 2, 0).astype(np.uint8)
