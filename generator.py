import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from resblock import ResBlock
from lstm import LSTM
import torch.nn.functional as F
from autoencoder import ContxtAutoEncoder
from utils import binary_mask, resize, show_img
from vgg16 import VGG16


class Generator(nn.Module):
    def __init__(self, input_shp, config):
        super(Generator, self).__init__()

        self.input_shp = input_shp
        self.indim = input_shp[1]
        self.num_res_layer = config.num_res_layer
        self.num_time_step = config.num_time_step
        self.theta = config.theta

        # ResNet Blocks
        # + 1 is the mask
        self.res_blocks = ResBlock(self.indim + 1, config)

        # LSTM
        self.lstm = LSTM(input_shp)

        self.det_conv_mask = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1)
        )
        
        self.autoencoder = ContxtAutoEncoder()

    def autoencoder_loss(self, decoded_imgs, label_img):
        loss_m = self.multiscale_loss(decoded_imgs, label_img)
        # The last item is the final decoded image
        loss_p = self.perceptual_loss(decoded_imgs[-1], label_img)
        print("Loss M: {}".format(loss_m))
        print("Loss P: {}".format(loss_p))
        return loss_m + loss_p

    def multiscale_loss(self, decoded_imgs, label_img):
        lambdas = [0.6, 0.8, 1]
        scales = [0.25, 0.5, 1]
        loss_m = 0
        for i, img in enumerate(decoded_imgs):
            scaled_img = resize(label_img, scales[i]).detach_()
            loss_m += lambdas[i] * F.mse_loss(scaled_img, img)
        return loss_m

    def perceptual_loss(self, decoded_img, label_img):
        model = VGG16()
        outputs = model.forward(decoded_img)
        outputs2 = model.forward(label_img)
        losses = []
        for i, out in enumerate(outputs):
            l = F.mse_loss(out, outputs2[i])
            losses.append(l)
        l_p = torch.FloatTensor(losses).mean()

        return l_p

    def forward(self, rain_img, label_img):
        batch_size, row, col = rain_img.shape[0], rain_img.shape[2], rain_img.shape[3]
        # Attention map is init to 0.5 according to paper
        mask = Variable(torch.ones(batch_size, 1, row, col)) / 2.
        if torch.cuda.is_available():
            mask = mask.cuda()
        mask_list = []
        # Number of time steps
        for i in range(self.num_time_step):
            # Concat the input image with the previous attention map
            # And feed to the next block of recurrent network
            x = torch.cat((rain_img, mask), 1)
            # ResNet block
            x = self.res_blocks(x)
            h = self.lstm(x)
            mask = self.det_conv_mask(h)
            mask_list.append(mask)
        # Calculate attentive-loss
        loss_att = 0.0
        bin_mask = binary_mask(rain_img, label_img).detach_()
        for i in range(self.num_time_step):
            mse_loss = F.mse_loss(bin_mask, mask_list[i])
            loss_att += np.power(self.theta, self.num_time_step - i + 1) \
                        * mse_loss

        x = torch.cat((rain_img, mask), 1)
        x, frame1, frame2 = self.autoencoder(x)
        # Loss multiscale and perceptual
        loss_ae = self.autoencoder_loss([frame1, frame2, x], label_img)
        print("Loss AE: {}".format(loss_ae))
        print("Loss ATT: {}".format(loss_att))
        loss = loss_ae + loss_att
        return loss_ae, mask_list, frame1, frame2, x
