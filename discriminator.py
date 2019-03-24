import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 5, 1, 2),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, 1, 2),
            nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 64, 5, 1, 2),
            nn.ReLU()
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU()
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 5, 1, 2),
            nn.ReLU()
            )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 5, 1, 2),
            nn.ReLU()
            )
        self.conv_mask = nn.Sequential(
            nn.Conv2d(128, 1, 5, 1, 2)
            )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 64, 5, 4, 1),
            nn.ReLU()
            )
        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 64, 5, 4, 1),
            nn.ReLU()
            )
        self.conv9 = nn.Sequential(
            nn.Conv2d(64, 32, 5, 4, 1),
            nn.ReLU()
            )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 11, 1024),
            nn.Linear(1024, 1),
            nn.Sigmoid()
            )
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        mask = self.conv_mask(x)
        x = self.conv7(x * mask)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.view(x.size(0), -1)
        
        return mask, self.fc(x)

if __name__ == '__main__':
    model = Discriminator()
    img = mpimg.imread('/Volumes/Data/MSc-UVic/DeepLearning/DeRainDrop/test_a/data/0_rain.png')
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    # x = torch.rand(1, 3, 480, 720)
    out = model.forward(img)
    mask = out.squeeze(0).detach().numpy()[0]
    mask = mask.transpose(1, 2, 0).squeeze(-1)
    plt.imshow(mask, cmap="gray")
    plt.show()
    # print(out)