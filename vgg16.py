import torch
import torch.nn as nn
import numpy as np
import torchvision
import matplotlib.image as mpimg


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.vgg_model = torchvision.models.vgg16(pretrained=True)
        self.vgg_model.train(False)
        self.vgg_model.eval()
        for param in self.vgg_model.features.parameters():
            param.requires_grad = False
        for param in self.vgg_model.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            self.vgg_model = self.vgg_model.cuda()
        self.vgg_layers = self.vgg_model.features

        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2",
            '6': "relu2_1",
            '8': "relu2_2"
        }

    def forward(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output

# For testing
if __name__ == "__main__":
    model = VGG16()
    img = mpimg.imread('../test_a/data/0_rain.png')
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    img2 = mpimg.imread('../test_a/gt/0_clean.png')
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0)
    if torch.cuda.is_available():
        img = img.cuda()
        img2 = img2.cuda()
    model.eval()
    with torch.no_grad():
        outputs = model.forward(img)
        outputs2 = model.forward(img2)
        losses = []
        for i, out in enumerate(outputs):
            l = torch.nn.functional.mse_loss(out, outputs2[i])
            losses.append(l)
        l_p = torch.FloatTensor(losses).mean()
        print(l_p)