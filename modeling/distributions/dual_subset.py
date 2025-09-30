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
from math import comb


class DualSubsetDistribution(TorchDistributionWrapper):
    """A distribution for dual subset actions, where each action consists of two subsets
    the sum of the count of which must be greater than 0 and less than or equal to 5.
    The first subset is the play subset, and the second is the discard subset.
    WARNING: this distribution masks illegal count combinations, but does not enforce that the
    subsets are disjoint."""

    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        self.inputs = inputs
        self.model = model
        self._action_logp = None
        self.max_hand_size = self.model.max_supported_hand_size

    cached_comb_count_indices = None

    def comb_count_indices(self):
        """Returns a list where indices[i] is the starting index of subsets of size i."""
        if self.cached_comb_count_indices is None:
            indices = [0]  # size 0 starts at 0
            for size in range(1, 7):
                indices.append(comb(self.max_hand_size, size - 1) + indices[-1])
            self.cached_comb_count_indices = torch.tensor(
                indices, device=self.inputs.device
            ).contiguous()
        return self.cached_comb_count_indices

    def indices_to_sizes(self, indices):
        """Convert a batch of indices to sizes based on the comb_count_indices."""
        return (
            torch.bucketize(
                indices,
                self.comb_count_indices(),
                right=True,
            )
            - 1
        )

    def deterministic_sample(self):
        return self.sample(deterministic=True)

    def play_subset_dist(self):
        return Categorical(logits=self.inputs[:, :, 0])

    def max_discard_index(self, max_discard_count):
        comb_count_indices = self.comb_count_indices()
        return comb_count_indices[max_discard_count + 1] - 1

    def discard_mask(self, min_discard_index, max_discard_index):
        all_indices = torch.arange(self.inputs.shape[1], device=self.inputs.device)
        mask = (all_indices >= min_discard_index) & (all_indices <= max_discard_index)
        return mask

    def discard_subset_dist(self, num_played):
        min_discard_index = torch.zeros_like(num_played)
        min_discard_index = torch.where(num_played == 0, 1, min_discard_index)
        max_discard_count = 5 - num_played
        max_discard_index = self.max_discard_index(max_discard_count)
        min_discard_index = min_discard_index.unsqueeze(1)
        max_discard_index = max_discard_index.unsqueeze(1)

        mask = self.discard_mask(min_discard_index, max_discard_index)
        discard_logits = torch.where(mask, self.inputs[:, :, 1], -1e9)
        return Categorical(logits=discard_logits)

    def sample(self, deterministic=False):
        # First, sample the play subset
        play_dist = self.play_subset_dist()
        if deterministic:
            play_subset_i = play_dist.mode
        else:
            play_subset_i = play_dist.sample()

        num_played = self.indices_to_sizes(play_subset_i)
        discard_dist = self.discard_subset_dist(num_played)
        if deterministic:
            discard_subset_i = discard_dist.mode
        else:
            discard_subset_i = discard_dist.sample()

        self._action_logp = play_dist.log_prob(play_subset_i) + discard_dist.log_prob(
            discard_subset_i
        )

        return play_subset_i, discard_subset_i

    def logp(self, actions):
        play_subset_i = actions[:, 0]
        discard_subset_i = actions[:, 1]
        logp_play = self.play_subset_dist().log_prob(play_subset_i)
        logp_discard = self.discard_subset_dist(
            self.indices_to_sizes(play_subset_i)
        ).log_prob(discard_subset_i)
        return logp_play + logp_discard

    def entropy(self):
        play_dist = self.play_subset_dist()
        play_probs = play_dist.probs
        play_sizes = self.indices_to_sizes(
            torch.arange(play_probs.shape[1], device=play_probs.device)
        )

        total_discard_entropy = torch.zeros(
            play_probs.shape[0], device=play_probs.device
        )

        for size in range(6):
            size_mask = play_sizes == size
            if size_mask.any():
                bucket_probs = play_probs[:, size_mask].sum(dim=1)
                discard_dist = self.discard_subset_dist(
                    torch.full((play_probs.shape[0],), size, device=play_probs.device)
                )
                discard_entropy = discard_dist.entropy()
                total_discard_entropy += bucket_probs * discard_entropy  # weighted sum

        return play_dist.entropy() + total_discard_entropy

    def kl(self, other):
        pass

    def sampled_action_logp(self):
        return self._action_logp

    # This is required but this distribution actually takes in a 2d tensor of shape (B, n, 2)
    # where n is the number of 0-5 combinations of the max hand size.
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 28

DualSubsetDist = DualSubsetDistribution
