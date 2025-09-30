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

# this will automatically be picked up by RLlib’s metrics reporter
over_5_cards_counter = Counter("samples_over_5")
under_1_card_counter = Counter("samples_under_1")


class PlayDiscardBinaryDistribution(TorchDistributionWrapper):
    def build_legal_masks(n_cards=8, k_min=1, k_max=5):
        rows = []
        for k in range(k_min, k_max + 1):
            for comb in itertools.combinations(range(n_cards), k):
                row = torch.zeros(n_cards)
                row[list(comb)] = 1.0
                rows.append(row)
        return torch.stack(rows)  # (218, 8)

    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        legal = PlayDiscardBinaryDistribution.build_legal_masks()
        self.legal_masks = legal.to(self.inputs.device)  # (218, 8)
        self.allow_illegal_actions = model.allow_illegal_actions

    def deterministic_sample(self):
        print("PlayDiscardBinaryDistribution.deterministic_sample() called")
        self.sample_was_called = True
        mode_dist = self.mode_distribution()
        mode = mode_dist.probs.argmax(dim=1)  # (B,)

        cards = self.sample_cards(mode, deterministic=True)

        action = torch.cat([mode.unsqueeze(1), cards], dim=1)
        self._action_logp = self.logp(action)

        return action

    def sample(self):
        print("PlayDiscardBinaryDistribution.sample() called")
        self.sample_was_called = True
        mode_dist = self.mode_distribution()
        mode = mode_dist.sample()  # (B,)

        cards = self.sample_cards(mode, deterministic=False)

        action = torch.cat([mode.unsqueeze(1), cards], dim=1)
        self._action_logp = self.logp(action)

        return action

    def mode_logits(self):
        mode_logits = self.inputs[:, 16:18]
        if not self.allow_illegal_actions:
            cannot_discard = self.cannot_discard_flags()
            adjusted_mode_logits = mode_logits.clone()
            adjusted_mode_logits[cannot_discard.bool(), 0] = (
                -1e9  # lower the odds of discarding if we cannot discard
            )
            mode_logits = adjusted_mode_logits
        return mode_logits

    def mode_distribution(self):
        mode_logits = self.mode_logits()
        return torch.distributions.Categorical(logits=mode_logits)

    def cannot_discard_flags(self):
        return self.inputs[:, 18]  # (B,)

    def count_distribution(self, mode):
        play_count_logits = self.play_count_logits()
        discard_count_logits = self.discard_count_logits()
        count_logits = torch.where(
            mode.unsqueeze(1) == 1, play_count_logits, discard_count_logits
        )
        return TorchCategorical(count_logits)

    def card_logits(self, mode):
        return torch.where(
            mode.unsqueeze(1) == 1, self.play_card_logits(), self.discard_card_logits()
        )

    def play_card_logits(self):
        return self.inputs[:, :8]

    def discard_card_logits(self):
        return self.inputs[:, 8:16]

    def sample_cards(self, mode, deterministic=False):
        B = self.inputs.shape[0]
        # logits = self.card_logits(mode)  # shape (B,8)
        # inside sample_cards(), instead of independent Bernoulli + fixups:
        logits = self.card_logits(mode)  # (B, 8)
        log_p = torch.log(torch.sigmoid(logits))  # (B, 8)   == log(sigmoid)
        log_q = torch.log(torch.sigmoid(-logits))  # (B, 8)   == log(1−sigmoid)

        # compute per‐mask log-prob for each batch:
        #   legal_masks: (218,8), log_p/q need to be (8,B) for matmul convenience
        log_p_t = log_p.T  # (8, B)
        log_q_t = log_q.T  # (8, B)
        logits_per_mask = (self.legal_masks @ log_p_t) + (
            (1 - self.legal_masks) @ log_q_t
        )  # (218, B)

        # now sample one mask *per* batch row:
        dist = Categorical(logits=logits_per_mask.T)  # (B, 218)
        if deterministic:
            # For deterministic sampling, we can just take the argmax
            idx = logits_per_mask.argmax(dim=0)
        else:
            idx = dist.sample()  # (B,)

        # pick out the 8‐bit mask for each row:
        cards = self.legal_masks[idx]  # (B, 8)
        return cards

    def logp(self, actions):
        mode_dist = self.mode_distribution()

        mode = actions[:, 0]
        cards = actions[:, 1:]

        mode_logp = mode_dist.log_prob(mode.float())

        cards_logp = self.cards_logp(cards, mode)
        # cards_logp = cards_logp.sum(dim=1)

        return mode_logp + cards_logp

    def cards_logp(self, cards_mask, mode):
        """
        cards_mask : (B, 8)  binary float tensor AFTER your clamp (1-5 ones)
        returns    : (B,)    joint log-prob under the truncated Bernoulli product
        """
        logits = self.card_logits(mode)  # (B, 8)
        p = torch.sigmoid(logits)  # (B, 8)

        # ---- raw joint logp of this mask under independent Bernoulli ----
        logp_raw = (cards_mask * torch.log(p) + (1 - cards_mask) * torch.log1p(-p)).sum(
            dim=-1
        )  # (B,)

        if self.allow_illegal_actions:
            return logp_raw

        # ---- compute logZ (probability that 1 <= #ones <= 5) ----
        # legal_masks  : (218, 8)
        # log p, log(1-p): (B,8) -> (8,B) for matmul convenience
        log_p = torch.log(p).T  # (8, B)
        log_q = torch.log1p(-p).T  # (8, B)

        # each legal mask row m gives   m·log p + (1-m)·log q
        # result shape (218, B)
        log_all = (self.legal_masks @ log_p) + ((1 - self.legal_masks) @ log_q)

        logZ = torch.logsumexp(log_all, dim=0)  # (B,)

        return logp_raw - logZ  # (B,)

    def entropy(self):
        BATCH = self.inputs.shape[0]
        mode_dist = self.mode_distribution()
        mode_entropy = mode_dist.entropy()
        mode_probs = torch.softmax(self.mode_logits(), dim=1)
        # mode_prob = torch.sigmoid(self.mode_logits())
        cards_entropy = torch.zeros(
            (BATCH,), device=self.inputs.device, dtype=torch.float32
        )
        for mode in range(2):
            cards_entropy += (
                mode_probs[:, 1] if mode else mode_probs[:, 0]
            ) * self.cards_entropy(mode)

        return mode_entropy + cards_entropy

    def cards_entropy(self, mode):
        """
        Returns: (B,)  the entropy of the truncated‐Bernoulli over all subsets S with 1 <= |S| <= 5.
        Uses the same `self.legal_masks` that you built in `build_legal_masks()`.
        """

        # 1) get the raw logits and probabilities for each card
        if mode == 1:
            logits = self.play_card_logits()
        else:
            logits = self.discard_card_logits()

        if self.allow_illegal_actions:
            base_dist = torch.distributions.Bernoulli(logits=logits)
            return base_dist.entropy().sum(dim=1)

        p = torch.sigmoid(logits)  # (B, 8)

        # 2) build the per‐mask “raw log‐prob” matrix exactly as in cards_logp:
        #    - log_p: shape (8, B),  log probabilities of choosing each card
        #    - log_q: shape (8, B),  log probabilities of NOT choosing each card
        log_p = torch.log(torch.clamp(p, 1e-6, 1 - 1e-6)).T  # (8, B)
        log_q = torch.log1p(-torch.clamp(p, 1e-6, 1 - 1e-6)).T  # (8, B)

        # 3) legal_masks: (218, 8) float tensor with exactly one “1” for each chosen card in that subset.
        #    Make sure it’s on the same device and dtype as log_p/log_q:
        lm = self.legal_masks.to(log_p.device)  # (218, 8)

        # 4) compute `log_all_masks`: for each legal mask and each batch‐element,
        #    the unnormalized log‐prob under the independent Bernoullis:
        #      log_all_masks[k, b] = sum_i [ m[k,i] * log_p[i,b] + (1 - m[k,i]) * log_q[i,b] ]
        #
        #    We do it via matrix‐multiplication:
        #      (legal_masks @ log_p)   has shape (218, B)
        #      ((1 - legal_masks) @ log_q) also (218, B)
        #    Summing those gives log‐prob for each of the 218 subsets, for each of the B batch rows.
        log_all_masks = (lm @ log_p) + ((1.0 - lm) @ log_q)  # (218, B)

        # 5) normalize: compute logZ for each batch element (vector of length B):
        #      logZ[b] = logsumexp_{k=1..218} [ log_all_masks[k, b] ]
        logZ = torch.logsumexp(log_all_masks, dim=0)  # (B,)

        # 6) now obtain the “truncated” log‐prob logits:
        #      logP[k, b] = log_all_masks[k,b] - logZ[b]
        #    so if we do:
        logP = log_all_masks - logZ.unsqueeze(0)  # (218, B)

        # 7) convert to actual probability P:
        #    doing softmax over the 218 subsets (dim=0)
        P = torch.softmax(log_all_masks, dim=0)  # (218, B)
        #    (equivalently: P = torch.exp(logP) )

        # 8) compute entropy per batch row:
        #      H[b] = - sum_{k=1..218} P[k,b] * logP[k,b]
        H = -(P * logP).sum(dim=0)  # (B,)

        return H  # shape (B,)

    def kl(self, other):
        BATCH = self.inputs.shape[0]
        mode_dist = self.mode_distribution()
        mode_kl = torch.distributions.kl_divergence(
            mode_dist, other.mode_distribution()
        )
        # mode_kl = mode_dist.kl(other.mode_distribution())

        # mode_probs = torch.softmax(self.mode_logits(), dim=1)
        mode_prob = torch.sigmoid(self.mode_logits())
        cards_kl = torch.zeros_like(mode_kl)
        for mode in range(2):
            cards_kl += (mode_prob if mode else 1 - mode_prob) * self.cards_kl(
                other, mode
            )
        return mode_kl + cards_kl

    def cards_kl(self, other, mode):
        BATCH = self.inputs.shape[0]

        if mode == 1:
            self_card_logits = self.play_card_logits()
            other_card_logits = other.play_card_logits()
        else:
            self_card_logits = self.discard_card_logits()
            other_card_logits = other.discard_card_logits()

        self_card_probs = torch.sigmoid(self_card_logits)
        other_card_probs = torch.sigmoid(other_card_logits)

        self_card_dist = torch.distributions.Bernoulli(logits=self_card_logits)
        other_card_dist = torch.distributions.Bernoulli(logits=other_card_logits)

        return torch.distributions.kl_divergence(self_card_dist, other_card_dist).sum(
            -1
        )

    def sampled_action_logp(self):
        return self._action_logp

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 18  # controls model output feature vector size

PlayDiscardBinaryDist = PlayDiscardBinaryDistribution
