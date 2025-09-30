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


class ARBinaryDistribution(TorchDistributionWrapper):
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        self.allow_illegal_actions = model.allow_illegal_actions

    def cannot_discard_flags(self):
        return self.inputs[:, -1]

    def hidden_state_logits(self):
        return self.inputs[:, :-1]

    def adjust_mode_logits(self, mode_logits, cannot_discard):
        if not self.allow_illegal_actions:
            adjusted_mode_logits = mode_logits.clone()
            adjusted_mode_logits[cannot_discard.bool(), 0] = -1e9
            return adjusted_mode_logits
        return mode_logits

    def sample(self, deterministic=False):
        B = self.inputs.shape[0]
        cannot_discard = self.cannot_discard_flags()
        h = self.hidden_state_logits()
        prev = torch.zeros(B, 1, device=self.inputs.device, dtype=torch.float32)
        samples = []
        card_counts = torch.zeros(B, device=self.inputs.device, dtype=torch.float32)

        # Sample mode
        mode_logits, h = self.model.ar_step(prev, h)
        mode_logits = self.adjust_mode_logits(mode_logits, cannot_discard)
        mode_dist = Categorical(logits=mode_logits)
        if deterministic:
            mode = mode_logits.argmax(dim=1)
        else:
            mode = mode_dist.sample()
        samples.append(mode)
        prev = mode.unsqueeze(1).float()  # (B, 1)

        # Sample cards
        for i in range(self.model.hand_size):
            card_logits, h = self.model.ar_step(prev, h)
            # Mask out the 1 logit whenever card_counts is already over 5
            if not self.allow_illegal_actions:
                card_logits = card_logits.clone()
                card_logits[card_counts >= 5, 1] = -1e9

            if deterministic:
                card = card_logits.argmax(dim=1)
            else:
                card_dist = Categorical(logits=card_logits)
                card = card_dist.sample()
            card_counts += card.float()  # Increment card count
            samples.append(card)
            prev = card.unsqueeze(1).float()

        action = torch.stack(samples, dim=1)  # (B, 1 + hand_size)

        # if any of the counts are 0 then randomly pick an index and choose that
        if not self.allow_illegal_actions:
            zero_counts = card_counts == 0
            if zero_counts.any():
                random_indices = torch.randint(
                    1, self.model.hand_size + 1, (zero_counts.sum(),)
                )
                action[zero_counts, random_indices] = 1

        self._action_logp = self.logp(action)  # Store logp for later use
        return action

    def deterministic_sample(self):
        return self.sample(deterministic=True)

    def logp(self, action):
        # action: (B, 1 + hand_size)
        B = action.shape[0]
        cannot_discard = self.cannot_discard_flags()
        h = self.hidden_state_logits()
        prev = torch.zeros(B, 1, device=self.inputs.device, dtype=torch.float32)
        card_counts = torch.zeros(B, device=self.inputs.device, dtype=torch.float32)
        logp = 0

        # Mode logp
        mode_logits, h = self.model.ar_step(prev, h)
        mode_logits = self.adjust_mode_logits(mode_logits, cannot_discard)
        mode_dist = Categorical(logits=mode_logits)
        mode_logp = mode_dist.log_prob(action[:, 0])
        logp += mode_logp
        prev = action[:, 0].unsqueeze(1).float()

        # Card logp
        for i in range(self.model.hand_size):
            card_logits, h = self.model.ar_step(prev, h)
            # Mask out the 1 logit whenever card_counts is already over 5
            if not self.allow_illegal_actions:
                card_logits = card_logits.clone()
                card_logits[card_counts >= 5, 1] = -1e9

            card_dist = Categorical(logits=card_logits)
            card_logp = card_dist.log_prob(action[:, i + 1])
            logp += card_logp
            prev = action[:, i + 1].unsqueeze(1).float()
            card_counts += action[:, i + 1].float()

        return logp

    # Calculate the entropy of the distribution
    # NOTE: This is not the exact entropy of the distribution,
    # but rather an approximation based on the sampled actions.
    # Full entropy calculation would require iterating over all possible sequences
    # Which would be 2 **(hand_size + 1) sequences (before legality trimming)
    def entropy(self):
        B = self.inputs.shape[0]
        cannot_discard = self.cannot_discard_flags()
        h = self.hidden_state_logits()
        prev = torch.zeros(B, 1, device=self.inputs.device, dtype=torch.float32)
        card_counts = torch.zeros(B, device=self.inputs.device, dtype=torch.float32)
        entropy = 0

        # Mode entropy
        mode_logits, h = self.model.ar_step(prev, h)
        mode_logits = self.adjust_mode_logits(mode_logits, cannot_discard)
        mode_dist = Categorical(logits=mode_logits)
        entropy += mode_dist.entropy()
        prev = mode_dist.sample().unsqueeze(1)

        # Card entropy
        for i in range(self.model.hand_size):
            card_logits, h = self.model.ar_step(prev.float(), h)
            # Mask out the 1 logit whenever card_counts is already over 5
            if not self.allow_illegal_actions:
                card_logits = card_logits.clone()
                card_logits[card_counts >= 5, 1] = -1e9

            card_dist = Categorical(logits=card_logits)
            entropy += card_dist.entropy()
            prev = card_dist.sample().unsqueeze(1)
            card_counts += prev.squeeze(1).float()

        return entropy

    def sampled_action_logp(self):
        return self._action_logp

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return model_config["custom_model_config"]["ar_head_hidden_size"]

ARBinaryDist = ARBinaryDistribution
