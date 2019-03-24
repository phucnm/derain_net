import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, indim, config):
        super(ResBlock, self).__init__()

        # The first layer is from indim => 32 outdim
        self.num_res_layer = config.num_res_layer + 1
        self.ksize = 3
        self.outdim = 32
        setattr(
            self,
            "det_conv_0",
            nn.Sequential(
                nn.Conv2d(indim, 32, 3, 1, 1),
                nn.ReLU()
            )
        )
        indim = 32
        for i, _ in enumerate(range(self.num_res_layer), 1):
            setattr(
                self,
                "det_conv_{}".format(i),
                nn.Sequential(
                    nn.Conv2d(indim, self.outdim, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(indim, self.outdim, 3, 1, 1),
                    nn.ReLU()
                )
            )

    def forward(self, x):
        for i in range(self.num_res_layer):
            det_conv = getattr(
                self,
                "det_conv_{}".format(i)
            )
            if i == 0:
                x = det_conv(x)
            else:
                resx = x
                x = F.relu(det_conv(x) + resx)
        return x
