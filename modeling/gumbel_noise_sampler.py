import torch
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import normc_initializer as normc_init_torch
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

# from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from ray.rllib.utils.framework import try_import_torch

# from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
)
from torch.distributions import Categorical, Bernoulli
from gymnasium.spaces import Box
import torch.nn as nn
from itertools import combinations
from ray.rllib.utils.torch_utils import FLOAT_MIN
import itertools
from ray.util.metrics import Counter
from modeling.distributions import ModeCountBinaryDistribution
from random import random


class GumbelNoiseSamplerDist(TorchCategorical):
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        self.inputs = inputs
        self.model = model

    def action_distribution(self):
        return Categorical(logits=self.inputs)

    def sample_gumbel(self, tau=1.0):
        eps = 1e-20
        U = torch.rand(self.inputs.shape, device=self.inputs.device)
        g_noise = -torch.log(-torch.log(U + eps) + eps)
        return g_noise * tau

    def deterministic_sample(self):
        return self.sample(deterministic=True)

    def sample(self, deterministic=False):
        # if random() < 0.25 and not deterministic:
        #     # Decide deterministically some of the time to explore with less variance
        #     return self.sample(deterministic=True)

        action_dist = self.action_distribution()
        if not deterministic:
            g_noise = self.sample_gumbel()
            action_dist = Categorical(logits=action_dist.logits + g_noise)
            # if random() < 0.1:
            #     action_dist = Categorical(logits=torch.zeros_like(action_dist.logits))
            # action_i = action_dist.sample()
        action_i = action_dist.probs.argmax(dim=1)

        self.last_sample = action_i

        return action_i
