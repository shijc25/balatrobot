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


class ExpertModeCountsDistribution(TorchDistributionWrapper):
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        self.num_experts = model.num_experts
        self.sub_distributions = []
        self.stride = 2 + 10 + 8 * 2  # mode + counts + play/discard cards
        self.inputs = inputs
        self.model = model

    def sub_inputs(self, options):
        options = options.long()  # ensure options are long tensor
        start = options * self.stride
        offsets = torch.arange(self.stride, device=self.inputs.device)
        pos = start.unsqueeze(1) + offsets.unsqueeze(0)  # (B, stride)
        sub_inputs = self.inputs.gather(1, pos)

        return sub_inputs

    def sub_distribution(self, options):
        if isinstance(options, int):
            options = torch.full(
                (self.inputs.shape[0],),
                options,
                dtype=torch.long,
                device=self.inputs.device,
            )
        return ModeCountBinaryDistribution(self.sub_inputs(options), self.model)

    def cannot_discard_flag(self):
        return self.inputs[:, -1]

    def option_logits(self):
        # subtract 1 for the cannot discard flag
        return self.inputs[:, -self.num_experts - 1 : -1]

    def option_distribution(self):
        option_logits = self.option_logits()
        tau = 0
        eps = 1e-20
        U = torch.rand(option_logits.shape, device=option_logits.device)
        g_noise = -torch.log(-torch.log(U + eps) + eps)  # Gumbel noise
        option_dist = Categorical(logits=option_logits + g_noise * tau)
        return option_dist

    def deterministic_sample(self):
        return self.sample(deterministic=True)

    def sample(self, deterministic=False):
        # if random() < 0.25 and not deterministic:
        #     # Decide deterministically some of the time to explore with less variance
        #     return self.sample(deterministic=True)

        option_dist = self.option_distribution()
        if deterministic:
            option_idx = option_dist.probs.argmax(dim=1)
        else:
            option_idx = option_dist.sample()  # (B,)
        expert_dist = self.sub_distribution(option_idx)  # (B, n)
        if deterministic:
            sub_action = expert_dist.deterministic_sample()  # (B, n)
        else:
            sub_action = expert_dist.sample()  # (B, n)

        self._action_logp = (
            option_dist.log_prob(option_idx) + expert_dist.sampled_action_logp()
        )

        return torch.cat(
            [option_idx.unsqueeze(1).float(), sub_action],  # (B, 1)
            dim=1,
        )  # (B, n + 1)

    def logp(self, actions):
        option = actions[:, 0]
        sub_action = actions[:, 1:]
        option_dist = self.option_distribution()
        option_logp = option_dist.log_prob(option)  # (B,)

        expert_dist = self.sub_distribution(option)  # (B, n)
        sub_logp = expert_dist.logp(sub_action)  # (B,)

        return option_logp + sub_logp

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

ExpertModeCountsDist = ExpertModeCountsDistribution
