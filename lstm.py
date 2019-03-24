import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_shp):
        super(LSTM, self).__init__()
        batch_size = input_shp[0]
        h = input_shp[2]
        w = input_shp[3]
        self.cell_state = torch.zeros(batch_size, 32, h, w, requires_grad=True)
        self.lstm_feats = torch.zeros(batch_size, 32, h, w, requires_grad=True)
        if torch.cuda.is_available():
            self.cell_state = self.cell_state.cuda()
            self.lstm_feats = self.lstm_feats.cuda()

        # LSTM gates
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = torch.cat((x, self.lstm_feats), 1)
        i = self.conv_i(x)
        f = self.conv_f(x)
        g = self.conv_g(x)
        o = self.conv_o(x)
        self.cell_state = f * self.cell_state + i * g
        self.lstm_feats = o * torch.tanh(self.cell_state)
        return self.lstm_feats
