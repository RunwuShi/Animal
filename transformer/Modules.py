import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2)) # [2, 200, 200] 200: T (input: [1,80,200])
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf) # fill -np.inf to the True position

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn
