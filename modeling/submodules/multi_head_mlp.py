from copy import deepcopy
from gymnasium import spaces as sp
import numpy as np
from random import randint, choice, random
import torch
import torch.nn as nn
from gym_envs.base_card import BaseCard
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MultiHeadMLP(nn.Module):
    def __init__(
        self,
        layer_dims: List[int],  # [input, H1, H2, …, HL, output]
        num_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.activation = nn.GELU()
        self.num_layers = len(layer_dims) - 1

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i in range(self.num_layers):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            w = nn.Parameter(torch.empty(num_heads, out_dim, in_dim))
            b = nn.Parameter(torch.empty(num_heads, out_dim))
            bound = 1.0 / in_dim**0.5
            nn.init.uniform_(w, -bound, bound)
            nn.init.uniform_(b, -bound, bound)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(1).expand(-1, self.num_heads, -1)

        for i in range(self.num_layers):
            w, b = self.weights[i], self.biases[i]
            h = torch.einsum("bni,noi->bno", h, w) + b
            if i < self.num_layers - 1:
                h = self.activation(h)

        return h
