from copy import deepcopy
from gymnasium import spaces as sp
import numpy as np
from random import randint, choice, random
import torch
import torch.nn as nn
from gym_envs.base_card import BaseCard
import math


class LinearLogitHead(nn.Module):
    def __init__(self, input_size, num_cards=8):
        super().__init__()
        self.input_size = input_size
        # self.head = nn.Sequential(
        #     nn.Linear(input_size, int(input_size / 2)),
        #     nn.GELU(),
        #     nn.Linear(int(input_size / 2), 2 + num_cards * 2),
        # )
        self.head = nn.Linear(input_size, 2 + num_cards * 2)

    def forward(self, x):
        return self.head(x)

    @staticmethod
    def num_outputs():
        return 2 + 8 * 2
