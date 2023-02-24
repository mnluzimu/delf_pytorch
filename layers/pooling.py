import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import layers.functional as LF


# --------------------------------------
# Pooling layers
# --------------------------------------


class SPoC(nn.Module):

    def __init__(self):
        super(SPoC, self).__init__()

    def forward(self, x):
        return LF.spoc(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


