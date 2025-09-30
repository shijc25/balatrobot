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


class ShopActionAndHandTargetsDistribution(TorchDistributionWrapper):
    def build_legal_masks(n_cards=8, k_min=0, k_max=3):
        rows = []
        for k in range(k_min, k_max + 1):
            for comb in itertools.combinations(range(n_cards), k):
                row = torch.zeros(n_cards)
                row[list(comb)] = 1.0
                rows.append(row)
        return torch.stack(rows)

    cached_legal_masks = None

    def __init__(self, inputs, model):
        hand_size = model.max_hand_size
        self.action_logits = inputs[:, :17]  # (B, 17) logits for actions
        self.hand_logits = inputs[:, 17:-5]  # (B, hand_size)
        self.target_counts = inputs[:, -5:]  # (B, 5) target counts for tarot cards
        if self.cached_legal_masks is None:
            with torch.no_grad():
                self.cached_legal_masks = (
                    ShopActionAndHandTargetsDistribution.build_legal_masks(
                        n_cards=hand_size
                    ).to(inputs.device)
                )
        self.legal_masks = self.cached_legal_masks

        self.masks_by_k = {
            k: self.legal_masks[
                (self.legal_masks.sum(dim=1) == k).nonzero(as_tuple=True)[0]
            ]
            for k in range(0, 4)
        }
        self.mask_sizes = self.legal_masks.sum(dim=1).long().to(inputs.device)  # (M,)
        # print(inputs.shape)
        # print(self.target_counts.shape)
        # print(self.target_counts)

    def action_dist(self):
        return torch.distributions.Categorical(logits=self.action_logits)

    def hand_exists_mask(self):
        return torch.any(self.hand_logits > -1e8, dim=1)

    # left pad with zeros to go from (B, hand_size) to (B, 17)
    def padded_target_counts(self):
        BATCH = self.target_counts.shape[0]
        padded = torch.zeros((BATCH, 17), device=self.target_counts.device)
        padded[:, -5:] = self.target_counts
        return padded

    def deterministic_sample(self):
        return self.sample(deterministic=True)

    def sample(self, deterministic=False):
        a_dist = self.action_dist()
        if not deterministic:
            action = a_dist.sample()
        else:
            action = a_dist.mode

        targets = self.sample_targets(action, deterministic=deterministic)

        actions = {"action": action, "hand_targets": targets}
        self._action_logp = self.logp(actions)
        # print(actions)

        return actions

    def targets_index_distribution(self, action):
        logits = self.hand_logits  # (B, n)
        count = self.padded_target_counts()[torch.arange(logits.shape[0]), action]
        # print(self.padded_target_counts())
        # print(action)
        # print(count)

        # 1) compute scores for all M masks against each batch row → (B, M)
        #    score[b,m] = sum_i legal_masks[m,i] * logits[b,i]
        scores = torch.matmul(logits, self.legal_masks.t())  # (B, M)
        # print(scores)

        # 2) invalidate masks whose size ≠ K[b]
        valid = self.mask_sizes.unsqueeze(0).eq(count.unsqueeze(1))  # (B, M)
        # print(valid)
        # print(valid.shape)
        # print(scores.shape)
        scores = scores.masked_fill(~valid, float("-1e9"))

        # 3) sample or argmax
        dist = Categorical(logits=scores)
        return dist

    def sample_targets(self, action, deterministic=False):
        dist = self.targets_index_distribution(action)
        idx = dist.sample() if not deterministic else dist.mode

        # print(idx)
        return self.legal_masks[idx]  # (B, n)

    def logp(self, actions):
        if type(actions) is not dict:
            action = actions[:, 0].long()
            hand_targets = actions[:, 1:]
        else:
            action = actions["action"]
            hand_targets = actions["hand_targets"]

        a_dist = self.action_dist()
        action_logp = a_dist.log_prob(action.long())

        action_logp += self.target_logp(action, hand_targets)
        return action_logp

    def target_logp(self, action, hand_targets):
        dist = self.targets_index_distribution(action)

        # Find the index of the mask that matches the hand_targets
        mask_index = (
            (self.legal_masks == hand_targets.unsqueeze(1))
            .all(dim=2)
            .nonzero(as_tuple=True)[1]
        )
        return dist.log_prob(mask_index)

    def entropy(self):
        a_dist = self.action_dist()
        action_entropy = a_dist.entropy()

        # Calculate the entropy for the target counts
        target_entropy = torch.zeros_like(action_entropy)
        for action in range(17):
            dist = self.targets_index_distribution(action)
            target_entropy += (
                dist.entropy()
                * a_dist.probs[torch.arange(action_entropy.shape[0]), action]
            )

        return action_entropy + target_entropy

    def kl(self, other):
        return 0

    def sampled_action_logp(self):
        return self._action_logp

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 18  # controls model output feature vector size

ShopActionAndHandTargetsDist = ShopActionAndHandTargetsDistribution
