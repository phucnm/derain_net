import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


def show_img(img):
    img = img.squeeze(0).permute(1, 2, 0)
    plt.figure()
    plt.imshow(img)
    plt.show()


def binary_mask(rain_img, clean_img):
    diff = torch.abs(rain_img - clean_img)
    threshold = 30.0
    diff[diff > threshold/255.0] = 1
    return diff
    

def resize(img, ratio):
    # Input img should be in shape (1, 3, 480, 720)
    # Transform to (480, 720, 3) and type uint8
    img = img.squeeze(0).cpu()
    pil_img = transforms.functional.to_pil_image(img)
    width, height = pil_img.size
    width = int(width * ratio)
    height = int(height * ratio)
    scaled = transforms.functional.resize(pil_img, (height, width))
    scaled = transforms.functional.to_tensor(scaled)    
    if torch.cuda.is_available():
        scaled = scaled.cuda()
    return scaled


def image2tensor(img):
    """Transform image array read from matplotlib to tensor with shape (1, x, h, w)
    """
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0)


def transform_to_mpimg(img):
    '''
    Transform tensor to matplotlib image
    From (1, C, w, h) to (w, h, C)
    '''
    img = img.squeeze(0).squeeze(0)
    return img.detach().numpy().astype(np.uint8)
