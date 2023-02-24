import torch.nn as nn


class ScoreFunction(nn.Module):

    def __init__(self, in_channels, middle_channels):
        super(ScoreFunction, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=middle_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=middle_channels, out_channels=1, kernel_size=1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        o = self.conv1(x)
        o = self.relu(o)
        o = self.conv2(o)
        o = self.softplus(o)
        return o

    def __repr__(self):
        tmpstr = super(ScoreFunction, self).__repr__()
        tmpstr += "a two-layer CNN"
        return tmpstr
