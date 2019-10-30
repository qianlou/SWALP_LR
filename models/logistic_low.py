"""
    LogisticRegression model definition
"""

import torch.nn as nn
import math
from torch.nn import functional as F
from .quantizer import BlockQuantizer



class LogisticRegression(nn.Module):
    def __init__(self, quant, num_classes=2, depth=1, batch_norm=False,
            writer=None):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(131072, 2)#test inception-v3 features

    def forward(self, x):
        outputs = self.linear(x) #F.sigmoid() we don't need that.F.softmax(, dim=1) 
        return outputs

class LogisticLP:
    base = LogisticRegression
    args = list()
    kwargs = {'depth': 1}



