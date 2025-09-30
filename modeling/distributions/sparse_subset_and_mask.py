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


class SparseSubsetAndMaskDistribution(TorchDistributionWrapper):
    """A distribution for subset actions, where the first action is a subset of the hand to play,
    and the second action is an intent vector used to determine the discard subset.
    The discard subset is currently not checked for size or disjointness with the play subset.
    """

    cached_comb_count_indices = None

    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        self.inputs = inputs
        self.logits = inputs[:, :, :-1]
        self.subset_indices = inputs[:, :, -1]
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

    def find_play_masks(self, play_subset_indices):
        return self.model.play_layer.mask_mat[play_subset_indices.long()]

    def deterministic_sample(self):
        return self.sample(deterministic=True)

    def play_logits(self):
        logits = self.logits[:, :, 0]
        # Uncomment this to bias towards the empty subset to make it more likely to explore discarding early on
        # Otherwise it rapidly learns to play non-empty subsets, and never explores the empty subset
        logits = logits.clone()
        logits[:, 0] += 1.0
        return logits

    def sample(self, deterministic=False):
        B, M, _ = self.logits.shape
        batch_idx = torch.arange(B, device=self.logits.device)
        # Handle the case when the logits are empty (likely during loss initialization)
        if M == 0:
            self._action_logp = torch.zeros(
                B, dtype=torch.float32, device=self.logits.device
            )
            return torch.zeros(
                B, dtype=torch.int64, device=self.logits.device
            ), torch.zeros(
                (B, self.max_hand_size), dtype=torch.float32, device=self.logits.device
            )

        # First, sample the play subset
        relative_idx_dist = Categorical(logits=self.play_logits())
        if deterministic:
            relative_play_subset_i = relative_idx_dist.mode
        else:
            relative_play_subset_i = relative_idx_dist.sample()

        # print(self.logits[:, :, 0])
        # print(batch_idx.shape, relative_play_subset_i.shape)
        # print(batch_idx, relative_play_subset_i)
        play_subset_i = self.subset_indices[batch_idx, relative_play_subset_i]
        num_played = self.indices_to_sizes(play_subset_i)

        # Now, determine the discard mask using the associated logits from the sampled play subset
        discard_logits = self.logits[batch_idx, relative_play_subset_i, 1:]
        # force discard_mask to be disjoint with play_subset_i
        play_masks = self.find_play_masks(play_subset_i)
        discard_logits = discard_logits.masked_fill(play_masks.bool(), -1e9)
        discard_dist = Bernoulli(logits=discard_logits)
        if deterministic:
            discard_mask = (discard_dist.probs > 0.5).float()
        else:
            discard_mask = discard_dist.sample()

        # Force 0 discard where 5 cards are played
        discard_mask = discard_mask * (num_played < 5).float().unsqueeze(1)
        # Force at least one discard if no cards are played
        forced_discard = Categorical(logits=discard_logits).sample()
        forced_discard_mask = torch.zeros_like(discard_mask)
        forced_discard_mask[batch_idx, forced_discard] = 1.0
        discard_mask = torch.where(
            num_played.unsqueeze(1) == 0, forced_discard_mask, discard_mask
        )

        self._action_logp = self.logp(
            torch.cat([play_subset_i.unsqueeze(1), discard_mask], dim=1)
        )
        return play_subset_i, discard_mask

    def logp(self, actions):
        """
        actions == (play_subset_ids: LongTensor (B,),
                    discard_mask: BoolTensor or FloatTensor (B, N))
        """
        play_subset, discard_mask = actions[:, 0], actions[:, 1:]
        B, M, C = self.logits.shape
        N = C - 1
        if M == 0:
            return torch.zeros(B, dtype=torch.float32, device=self.logits.device)

        batch_idx = torch.arange(B, device=self.logits.device)
        num_played = self.indices_to_sizes(play_subset)

        # 1) invert global→relative index
        # build mask of shape (B, M)
        rel_mask = self.subset_indices == play_subset.unsqueeze(1)
        rel_idx = rel_mask.float().argmax(dim=1)  # (B,)

        # 2) recompute the two dists
        cat_dist = Categorical(logits=self.play_logits())  # (B, M)
        mask_logits_all = self.logits[:, :, 1:]  # (B, M, N)
        # pick logits for chosen slot
        discard_logits = mask_logits_all[batch_idx, rel_idx]  # (B, N)
        play_masks = self.find_play_masks(play_subset)
        # force discard_mask to be disjoint with play_subset_i
        discard_logits = discard_logits.masked_fill(play_masks.bool(), -1e9)

        bern_dist = Bernoulli(logits=discard_logits)

        # 3) sum the log‐probs
        logp_cat = cat_dist.log_prob(rel_idx)  # (B,)

        # This is not mathematically correct, but we want to make a hard distinction between
        # rel_idx = 0 (no play) and rel_idx > 0 (play subset)
        # so we are going to add the log prob of choosing the first subset specifically
        # chose_no_play = rel_idx == 0
        # zero_probs = cat_dist.probs[:, 0]
        # non_zero_probs = cat_dist.probs[:, 1:].sum(dim=1)
        # logp_mode = torch.zeros_like(zero_probs, device=self.logits.device)
        # logp_cat[chose_no_play] = torch.log(zero_probs[chose_no_play])
        # logp_cat[~chose_no_play] = torch.log(non_zero_probs[~chose_no_play])

        # ensure mask is float
        dm = discard_mask.float()
        logp_mask = bern_dist.log_prob(dm).sum(dim=1)  # (B,)

        # Where num_played == 5, we ignore the discard mask
        logp_mask = torch.where(num_played == 5, torch.zeros_like(logp_mask), logp_mask)

        return logp_cat + logp_mask  # (B,)

    def entropy(self):
        """
        H = H(subset) + E_{subset}[ H(mask | subset) ]
        """
        B, M, C = self.logits.shape
        if M == 0:
            return torch.zeros(B, dtype=torch.float32, device=self.logits.device)

        probs0 = torch.softmax(self.play_logits(), dim=1)  # (B, M)
        cat_ent = Categorical(probs=probs0).entropy()  # (B,)

        mask_logits_all = self.logits[:, :, 1:]  # (B, M, N)
        # entropy of each mask‐distribution, summed over cards
        bern_ent_all = Bernoulli(logits=mask_logits_all).entropy().sum(dim=2)  # (B, M)
        # expectation under the subset‐softmax
        mask_ent = (probs0 * bern_ent_all).sum(dim=1)  # (B,)

        return cat_ent + mask_ent

    def kl(self, other):
        pass

    def sampled_action_logp(self):
        return self._action_logp

    # This is required but this distribution actually takes in a 2d tensor of shape (B, n, 2)
    # where n is the number of 0-5 combinations of the max hand size.
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 28

SparseSubsetAndMaskDist = SparseSubsetAndMaskDistribution
