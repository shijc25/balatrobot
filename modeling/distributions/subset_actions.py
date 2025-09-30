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
from .mode_count_multi_binary import ModeCountBinaryDistribution
from random import random


class SubsetActionsDistribution(TorchDistributionWrapper):
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        self.inputs = inputs
        self.model = model
        self.num_cards = 8
        self.action_masks = self.build_action_masks()

    def cannot_discard_flag(self):
        return self.inputs[:, -1]

    def build_action_masks(self):
        masks = []
        for n in range(1, 6):
            combos = list(combinations(range(self.num_cards), n))
            self.num_actions += len(combos)  # count total actions
            for combo in combos:
                mask = torch.zeros(self.num_cards, dtype=bool)
                mask[list(combo)] = True
                masks.append(mask)
        return torch.stack(masks, axis=0).astype(bool)

    def action_logits(self):
        return self.inputs[:, :-1]

    def sample_gumbel(self, tau=1.0):
        logits = self.action_logits()
        eps = 1e-20
        U = torch.rand(logits.shape, device=logits.device)
        g_noise = -torch.log(-torch.log(U + eps) + eps)
        return g_noise * tau

    def action_distribution(self):
        logits = self.action_logits()
        return Categorical(logits=logits)

    def index_to_mode(self, index):
        return index // self.action_masks.shape[0]

    def index_to_card_mask(self, index):
        card_mask = self.action_masks[index % self.action_masks.shape[0]]
        return card_mask

    def deterministic_sample(self):
        return self.sample(deterministic=True)

    def sample(self, deterministic=False):
        # if random() < 0.25 and not deterministic:
        #     # Decide deterministically some of the time to explore with less variance
        #     return self.sample(deterministic=True)

        use_gumbel = not deterministic
        action_dist = self.action_distribution()
        if use_gumbel:
            g_noise = self.sample_gumbel()
            action_dist = Categorical(logits=action_dist.logits + g_noise)
            action_i = action_dist.probs.argmax(dim=1)  # (B,)
        else:
            action_i = action_dist.sample()  # (B,)
        mode = self.index_to_mode(action_i)
        card_mask = self.index_to_card_mask(action_i)

        self._action_logp = action_dist.log_prob(action_i)  # (B,)

        action = torch.cat(
            [
                mode,
                card_mask.float(),  # Convert boolean mask to float
            ],
            dim=1,
        )

        return action  # (B, n + 1)

    def logp(self, actions):
        mode = actions[:, 0].long()
        card_mask = actions[:, 1:]
        action_i = mode * self.action_masks.shape[0]

        action_dist = self.action_distribution()
        action_logp = action_dist.log_prob(mode)

    def entropy(self):
        option_dist = self.option_distribution()
        option_entropy = option_dist.entropy()

        sub_entropy = torch.zeros_like(option_entropy)
        experts = []
        for i in range(self.num_experts):
            expert_dist = self.sub_distribution(i)
            sub_entropy += expert_dist.entropy() * option_dist.probs[:, i]
            experts.append(expert_dist)

        # Add an "entropy" bonus which is actually a regularization term
        # To encourage kl divergence between experts
        kl_entropy = torch.zeros_like(option_entropy)
        coeff = 0.00
        if self.num_experts > 1 and coeff > 0:
            for i, j in itertools.combinations(range(self.num_experts), 2):
                kl_entropy += experts[i].kl(experts[j])
                kl_entropy += experts[j].kl(experts[i])
            kl_entropy /= self.num_experts * (self.num_experts - 1) / 2.0
            kl_entropy = kl_entropy * coeff

        return option_entropy + sub_entropy + kl_entropy

    def kl(self, other):
        return 0

    def sampled_action_logp(self):
        return self._action_logp

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 28  # controls model output feature vector size

SubsetActionsDist = SubsetActionsDistribution
