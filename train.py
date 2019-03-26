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
from generator import Generator
from vgg16 import VGG16
from tensorboardX import SummaryWriter


def attentive_loss(rain_img, label_img, mask_list, config):
    """Compute attentive loss. Eq4 in the paper

    Parameters
    ----------

    rain_img: torch.Tensor
        Shape 1, 3, 480, 720
        The image degraded by raindrops

    label_img: torch.Tensor
        Shape 1, 3, 480, 720
        The clean image with clear background

    mask_list: list
        Each element has the same shape of input image
        List of attentive maps at time steps
        produced by attentive-recurrent network

    """

    loss_att = 0

    # Compute binary mask by subtracting the degraded image
    # with the clean image
    bin_mask = utils.binary_mask(rain_img, label_img).detach_()

    # Do the for loop to accumulate loss at each time step.
    for i in range(config.num_time_step):

        # The attention map is the concatenation of the input image
        # and the attention map from previous timestep
        # The initial attention map is initialized with 0.5
        mse_loss = F.mse_loss(bin_mask, mask_list[i] + rain_img)

        # Theta = 0.8 according to the paper
        loss_att += np.power(
            config.theta, config.num_time_step - i + 1
            ) * mse_loss
    return loss_att


def multiscale_loss(decoded_imgs, label_img):
    """Compute multiscale loss. Eq5 in the paper

    Parameters
    ----------

    decoded_imgs: list of images output from the decoder
        The list contains 3 elements with different scale ratio:
        0.25, 0.5, 1

    label_img: torch.Tensor
        Shape 1, 3, 480, 720
        The clean image with clear background

    """

    scales = [0.25, 0.5, 1]

    # Lambdas according to the paper
    # The larger image, the larger weight
    lambdas = [0.6, 0.8, 1]

    loss_m = 0
    for i, img in enumerate(decoded_imgs):
        # Scale the image
        scaled_img = utils.resize(label_img, scales[i]).detach_()
        # Compute MSE loss of an output of decoder and
        # corresponding scaled ground truth image
        loss_m += lambdas[i] * F.mse_loss(scaled_img, img)
    return loss_m


def perceptual_loss(decoded_img, label_img, vgg16):
    """Compute perceptual loss. Eq6 in the paper
    This simply compute loss between features extracted by pretrained model
    VGG16. The authors use MSE loss.

    Parameters
    ----------

    decoded_imgs: list of images output from the decoder
        The list contains 3 elements with different scale ratio:
        0.25, 0.5, 1

    label_img: torch.Tensor
        Shape 1, 3, 480, 720
        The clean image with clear background

    """

    # NOTE: I'm having trouble take this loss into account
    # because the model itself takes over 11GB of mem on K80 GPU
    # on Google Cloud Compute.
    # VGG needs around ~1GB in total to store the model and
    # to extract features.
    # I meet "Out of Memory" error all the time loading and running VGG16
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
    """Compute generative loss. This is the last 3 components of Eq7
    This is a helper method to make the training loop shorter

    Parameters
    ----------

    rain_img: torch.Tensor
        Shape 1, 3, 480, 720
        The image degraded by raindrops

    label_img: torch.Tensor
        Shape 1, 3, 480, 720
        The clean image with clear background

    mask_list: list
        Each element has the same shape of input image
        List of attentive maps at time steps
        produced by attentive-recurrent network

    frame1: torch.Tensor
        Scaled imaged at 0.25 ratio during decoding

    frame2: torch.Tensor
        Scaled imaged at 0.5 ratio during decoding

    x: torch.Tensor
        Output of the decoder (scale 1.0)

    vgg16: The pretrained model VGG16

    Returns
    ----------
    loss: total loss of the 3 components

    """
    att_loss = attentive_loss(rain_img, label_img, mask_list, config)
    m_loss = multiscale_loss([frame1, frame2, x], label_img)
    p_loss = perceptual_loss(x, label_img, vgg16)
    # print(att_loss)
    # print(m_loss)
    # print(p_loss)
    return att_loss + m_loss + p_loss


def d_map_loss(mask, mask_o, mask_r):
    """ Eq9 in paper

    Parameters
    ----------

    mask: torch.Tensor
        The final attention map

    mask_o: torch.Tensor
        The interior layer of output of the generator
        computed by discriminator

    mask_r: torch.Tensor
        The interior layer of clean image
        computed by discriminator

    """
    zeros_mask = torch.zeros(mask_r.shape)
    if torch.cuda.is_available():
        zeros_mask = zeros_mask.cuda()
    criterion = torch.nn.MSELoss()
    l_map = criterion(mask_o, mask) + \
        criterion(mask_r, zeros_mask)
    # print("Loss L_map: {}".format(l_map))
    return l_map

if __name__ == '__main__':
    config, unparsed = get_config()
    if len(unparsed) > 0:
        print_usage()
        exit(1)
    # Generate the image

    if config.enable_vgg16:
        vgg16 = VGG16()
    else:
        vgg16 = None

    # Create log directory and save directory if it does not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Create loggers and checkpoint saver
    g_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "generative_loss"))
    d_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "discriminative_loss"))

    model_gen = Generator(config)
    model_discriminator = Discriminator()
    if torch.cuda.is_available():
        model_gen = model_gen.cuda()
        model_discriminator = model_discriminator.cuda()

    # Setup optimizers
    optim_g = optim.Adam(
        model_gen.parameters(),
        lr=config.learning_rate
        )
    optim_d = optim.Adam(
        model_discriminator.parameters(),
        lr=config.learning_rate
        )

    # Load filenames in directories
    inp_list = sorted(os.listdir(config.data_dir))
    label_list = sorted(os.listdir(config.label_dir))

    # Init stuffs for training
    label_real = torch.ones(1, 1)
    label_fake = torch.zeros(1, 1)
    BCE_loss = torch.nn.BCELoss()
    if torch.cuda.is_available():
        label_real = label_real.cuda()
        label_fake = label_fake.cuda()

    starting_epoch = 0
    starting_index = 0
    # Load the model checkpoint if possible
    checkpoint_file = os.path.join(config.save_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_file):
        if config.resume:
            print("Checkpoint found! Resuming")
            # Read checkpoint file.
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            load_res = torch.load(checkpoint_file, map_location=device)
            # Resume iterations
            starting_epoch = load_res["epoch"] + 1
            starting_index = load_res["index"] + 1
            # Resume model
            model_gen.load_state_dict(load_res["model_gen"])
            model_discriminator.load_state_dict(load_res["model_discriminator"])
            # Resume optimizer
            optim_g.load_state_dict(load_res["optim_g"])
            optim_d.load_state_dict(load_res["optim_d"])
        else:
            os.remove(checkpoint_file)

    # Num epoch is around 100000 to get good results
    for epoch in range(config.num_epoch)[starting_epoch:]:
        prefix = "Training Epoch {:3d}: ".format(epoch)

        inp_list = inp_list[starting_index:]
        label_list = label_list[starting_index:]
        # Batch size is 1 by default.
        # It's not neccessary to use data loader
        # As I'm loading images one by one.
        for index in tqdm(range(len(inp_list))):
            img = mpimg.imread(config.data_dir + inp_list[index])
            img = utils.image2tensor(img)
            label_img = mpimg.imread(config.label_dir + label_list[index])
            label_img = utils.image2tensor(label_img)

            if torch.cuda.is_available():
                img = img.cuda()
                label_img = label_img.cuda()

            # Train Discriminator
            optim_d.zero_grad()
            mask_r, D_real = model_discriminator.forward(label_img)
            masks, f1, f2, x = model_gen.forward(img)
            mask_f, D_fake = model_discriminator.forward(x)

            # Eq9
            # L_map is the loss between the features extraced from
            # interior layers of the discriminator and the final attention map
            map_loss = d_map_loss(masks[-1], mask_f, mask_r)

            # -log(D(R))
            D_loss_real = BCE_loss(D_real, label_real)
            # -log(1-D(O)) where O = G(z)
            D_loss_fake = BCE_loss(D_fake, label_fake)
            # Eq8. Gamma default to 0.05
            D_loss = D_loss_real + D_loss_fake + config.gamma * map_loss

            D_loss.backward()
            optim_d.step()

            # Train generator
            optim_g.zero_grad()
            # These was computed in above
            # masks, f1, f2, x = model_gen.forward(img)
            # mask_f, D_fake = model_discriminator.forward(x)
            gen_loss = generator_loss(
                img, label_img, masks,
                f1, f2, x, config, vgg16=vgg16
            )
            # Eq7
            G_loss = 0.01 * -BCE_loss(D_fake, label_fake) + gen_loss
            G_loss.backward()
            optim_g.step()
            optim_g.zero_grad()

            # Logging
            print("Loss D: {}".format(D_loss))
            print("Loss G: {}".format(G_loss))
            d_writer.add_scalar("loss", loss, global_step=index)
            g_writer.add_scalar("loss", loss, global_step=index)

            # Save the checkpoint
            if index % config.rep_intv == 0:
                torch.save({
                    "epoch": epoch,
                    "index": index,
                    "model_gen": model_gen.state_dict(),
                    "model_discriminator": model_discriminator.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "optim_d": optim_d.state_dict()
                }, checkpoint_file)
