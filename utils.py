import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


def show_img(img, cmap=None):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.show()


def binary_mask(rain_img, clean_img):
    """ Compute binary mask from the degraded image and the clean image
    Threashold is 30 according to the paper
    """
    diff = torch.abs(rain_img - clean_img)
    threshold = 30.0
    diff[diff > threshold/255.0] = 1
    return diff


def resize(img, ratio):
    # Input img should be in shape (1, 3, 480, 720)
    # Transform to (480, 720, 3) and type uint8

    # This is frustrating.
    # Firstly, move to CPU to convert the image to PIL image.
    img = img.squeeze(0).cpu()
    pil_img = transforms.functional.to_pil_image(img)

    # Extract the image size
    width, height = pil_img.size

    # Compute new image size
    width = int(width * ratio)
    height = int(height * ratio)

    # Resize and convert back to tensor, then move to GPU
    scaled = transforms.functional.resize(pil_img, (height, width))
    scaled = transforms.functional.to_tensor(scaled)
    if torch.cuda.is_available():
        scaled = scaled.cuda()
    return scaled


def image2tensor(img):
    """Transform image array read from matplotlib to tensor with shape (1, x, h, w)
    Align to four because there are some images have dimension 481, or 721
    """
    algined_w = int(img.shape[0]/4)*4
    algined_h = int(img.shape[1]/4)*4
    img = img[0: algined_w, 0: algined_h]
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0)


def transform_to_mpimg(img):
    """
    Transform tensor to matplotlib image
    From (1, C, w, h) to (w, h, C)
    Also multiply 255 to get uint8 data
    """
    img = img.squeeze(0).permute(1, 2, 0) * 255
    return img.detach().numpy().astype(np.uint8)
