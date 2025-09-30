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


class ModeCountBinaryDistribution(TorchDistributionWrapper):
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
        legal = ModeCountBinaryDistribution.build_legal_masks()
        self.legal_masks = legal.to(self.inputs.device)  # (218, 8)
        self.allow_illegal_actions = model.allow_illegal_actions
        self.masks_by_k = {
            k: self.legal_masks[
                (self.legal_masks.sum(dim=1) == k).nonzero(as_tuple=True)[0]
            ]  # shape (C_k, n)
            for k in range(1, 6)
        }
        self.mask_sizes = self.legal_masks.sum(dim=1).long()  # (M,)

    def mode_logits(self):
        mode_logits = self.inputs[:, 0:2]
        # if not self.allow_illegal_actions:
        #     cannot_discard = self.cannot_discard_flags()
        #     adjusted_mode_logits = mode_logits.clone()
        #     adjusted_mode_logits[cannot_discard.bool(), 0] = (
        #         -1e9  # lower the odds of discarding if we cannot discard
        #     )
        #     mode_logits = adjusted_mode_logits
        return mode_logits

    def mode_distribution(self):
        mode_logits = self.mode_logits()
        return torch.distributions.Categorical(logits=mode_logits)

    def cannot_discard_flags(self):
        return self.inputs[:, -1]  # (B,)

    def play_count_logits(self):
        return self.inputs[:, 7:12]

    def discard_count_logits(self):
        return self.inputs[:, 2:7]

    def illegal_count_mask(self):
        # Prevents playing more cards than are in the hand
        # By looking for the number of cards with non-masked logits

        # Should be the same mask for play and discard, so just take play
        card_logits = self.play_card_logits()
        possible_cards = card_logits > -1e8  # (B, n_cards)
        possible_counts = possible_cards.sum(dim=1)  # (B,)
        # Create a mask for counts that are too high
        illegal_mask = torch.arange(1, 6, device=self.inputs.device).unsqueeze(
            0
        ) > possible_counts.unsqueeze(
            1
        )  # (B, 5)

        return illegal_mask

    def count_distribution(self, mode):
        play_count_logits = self.play_count_logits()
        discard_count_logits = self.discard_count_logits()
        if isinstance(mode, int):
            mode = torch.full(
                (self.inputs.shape[0],),
                mode,
                dtype=torch.long,
                device=self.inputs.device,
            )
        count_logits = torch.where(
            mode.unsqueeze(1) == 1, play_count_logits, discard_count_logits
        )

        illegal_counts = self.illegal_count_mask()
        count_logits = count_logits.masked_fill(illegal_counts, -1e9)
        return torch.distributions.Categorical(logits=count_logits)

    def play_card_logits(self):
        return self.inputs[:, 20:28]

    def discard_card_logits(self):
        return self.inputs[:, 12:20]

    def card_logits(self, mode):
        # print(
        #     mode.shape, self.play_card_logits().shape, self.discard_card_logits().shape
        # )
        return torch.where(
            mode.unsqueeze(1) == 1, self.play_card_logits(), self.discard_card_logits()
        )

    def deterministic_sample(self):
        return self.sample(deterministic=True)

    def sample(self, deterministic=False):
        # 1) mode
        mode_dist = self.mode_distribution()
        if deterministic:
            mode = mode_dist.probs.argmax(dim=1)
        else:
            mode = mode_dist.sample()  # (B,)

        # 2) count
        count_dist = self.count_distribution(mode)
        if deterministic:
            count = count_dist.probs.argmax(dim=1)
        else:
            count = count_dist.sample()  # (B,)

        # 3) mask of size K
        cards = self.sample_cards(
            mode, count + 1, deterministic=deterministic
        )  # see below

        # record total logp
        self._action_logp = (
            mode_dist.log_prob(mode)
            + count_dist.log_prob(count)
            + self.cards_logp(cards, mode, count + 1)
        )

        return torch.cat(
            [
                mode.unsqueeze(1),
                count.unsqueeze(1).float(),  # if you want to include count
                cards,
            ],
            dim=1,
        )

    def sample_cards(self, mode, count, deterministic=False):
        # B, n = logits.size()
        M = self.legal_masks.size(0)
        B = self.inputs.shape[0]  # batch size
        logits = self.card_logits(mode)

        # 1) compute scores for all M masks against each batch row → (B, M)
        #    score[b,m] = sum_i legal_masks[m,i] * logits[b,i]
        scores = torch.matmul(logits, self.legal_masks.t())  # (B, M)

        # 2) invalidate masks whose size ≠ K[b]
        valid = self.mask_sizes.unsqueeze(0).eq(count.unsqueeze(1))  # (B, M)
        scores = scores.masked_fill(~valid, float("-1e9"))

        # 3) sample or argmax
        dist = Categorical(logits=scores)
        idx = dist.sample() if not deterministic else scores.argmax(dim=1)  # (B,)

        # 4) gather the masks
        return self.legal_masks[idx]  # (B, n)

    def logp(self, actions):
        mode = actions[:, 0]
        count = actions[:, 1]
        cards = actions[:, 2:]
        mode_dist = self.mode_distribution()
        count_dist = self.count_distribution(mode)

        mode_logp = mode_dist.log_prob(mode.float())
        count_logp = count_dist.log_prob(count.float())
        cards_logp = self.cards_logp(cards, mode, count + 1)

        return mode_logp + count_logp + cards_logp

    def cards_logp(self, cards, mode, count):
        M = self.legal_masks.size(0)
        B = self.inputs.shape[0]  # batch size
        logits = self.card_logits(mode)

        # 1) recompute scores as above
        scores = torch.matmul(logits, self.legal_masks.t())  # (B, M)

        # 2) mask invalid sizes
        valid = self.mask_sizes.unsqueeze(0).eq(count.unsqueeze(1))  # (B, M)
        scores = scores.masked_fill(~valid, float("-1e9"))

        # 3) find index of each chosen mask
        #    matching[b,m] = True iff legal_masks[m] == cards[b]
        matching = (cards.unsqueeze(1) == self.legal_masks.unsqueeze(0)).all(
            dim=2
        )  # (B,M)
        idx = matching.float().argmax(dim=1)  # (B,)

        # 4) log‐prob = score[b,idx] − logsumexp(scores[b,:])
        logsumexp = torch.logsumexp(scores, dim=1)  # (B,)
        chosen = scores[torch.arange(B), idx]  # (B,)

        return chosen - logsumexp

    def entropy(self):
        """
        Returns: Tensor of shape (B,) giving the entropy of
        P(mode, K, mask) = P(mode)·P(K|mode)·P(mask|mode,K).
        """
        B, device = self.inputs.shape[0], self.inputs.device

        # --- 1) Mode entropy ---
        mode_logits = self.mode_logits()  # (B,2)
        mode_probs = F.softmax(mode_logits, dim=1)  # (B,2)
        H_mode = Categorical(probs=mode_probs).entropy()  # (B,)

        # --- 2) Count entropy, H(K|mode) ---
        # get the two 5‐logit heads
        play_count_logits = self.play_count_logits()  # (B,5)
        discard_count_logits = self.discard_count_logits()  # (B,5)

        play_count_probs = F.softmax(play_count_logits, dim=1)  # (B,5)
        discard_count_probs = F.softmax(discard_count_logits, dim=1)  # (B,5)

        H_play = Categorical(logits=play_count_logits).entropy()  # (B,)
        H_discard = Categorical(logits=discard_count_logits).entropy()  # (B,)

        # weight by mode_probs: mode=1 → play, mode=0 → discard
        H_count = mode_probs[:, 1] * H_play + mode_probs[:, 0] * H_discard  # (B,)

        H_mask = torch.zeros(B, device=device)  # (B,)
        for mode in range(2):
            for count in range(5):
                card_entropy = self.cards_entropy(mode, count)
                card_entropy *= mode_probs[:, mode]  # (B,)
                if mode == 0:
                    card_entropy *= discard_count_probs[:, count]  # (B,)
                else:
                    card_entropy *= play_count_probs[:, count]  # (B,)

        # --- total entropy ---
        return H_mode + H_count + H_mask

    def cards_entropy(self, mode, count):
        if mode == 1:
            logits = self.play_card_logits()
        else:
            logits = self.discard_card_logits()

        # 1) score every legal mask m against each row b → (B, M)
        scores = torch.matmul(logits, self.legal_masks.t())  # (B, M)

        # 2) invalidate masks whose size != K[b]
        valid = self.mask_sizes.unsqueeze(0).eq(count)  # (B, M)
        scores = scores.masked_fill(~valid, float("-1e9"))  # (B, M)

        # 3) compute P over the (remaining) masks
        P = torch.softmax(scores, dim=1)  # (B, M)
        logP = torch.log(P + 1e-20)  # (B, M)

        # 4) H = - sum_m P[b,m] * logP[b,m]
        H_mask = -(P * logP).sum(dim=1)  # (B,)

        return H_mask

    def kl(self, other):
        mode = self.mode_distribution()
        other_mode = other.mode_distribution()

        mode_kl = torch.distributions.kl_divergence(mode, other_mode)

        cards_kl = torch.zeros(self.inputs.shape[0], device=self.inputs.device)
        count_kl = torch.zeros(self.inputs.shape[0], device=self.inputs.device)
        for m in range(2):
            count_kl += self.count_kl(other, m) * mode.probs[:, m]
            for count in range(1, 5):
                cards_kl += (
                    self.cards_kl(other, m, count)
                    * mode.probs[:, m]
                    * self.count_distribution(m).probs[:, count - 1]
                )

        return mode_kl + count_kl + cards_kl

    def count_kl(self, other, mode):
        BATCH = self.inputs.shape[0]

        if mode == 1:
            self_count_logits = self.play_count_logits()
            other_count_logits = other.play_count_logits()
        else:
            self_count_logits = self.discard_count_logits()
            other_count_logits = other.discard_count_logits()

        self_count_dist = torch.distributions.Categorical(logits=self_count_logits)
        other_count_dist = torch.distributions.Categorical(logits=other_count_logits)

        return torch.distributions.kl_divergence(self_count_dist, other_count_dist)

    def cards_kl(self, other, mode, count):
        BATCH = self.inputs.shape[0]

        if mode == 1:
            self_card_logits = self.play_card_logits()
            other_card_logits = other.play_card_logits()
        else:
            self_card_logits = self.discard_card_logits()
            other_card_logits = other.discard_card_logits()

        # 1) score every legal mask m against each row b → (B, M)
        self_scores = torch.matmul(self_card_logits, self.legal_masks.t())
        other_scores = torch.matmul(other_card_logits, self.legal_masks.t())
        # (B, M)
        # 2) invalidate masks whose size != K[b]
        valid = self.mask_sizes.unsqueeze(0).eq(count)
        self_scores = self_scores.masked_fill(~valid, FLOAT_MIN)
        other_scores = other_scores.masked_fill(~valid, FLOAT_MIN)
        # (B, M)
        # 3) compute P over the (remaining) masks
        self_P = torch.softmax(self_scores, dim=1)
        other_P = torch.softmax(other_scores, dim=1)
        # (B, M)
        self_logP = torch.log(self_P + 1e-20)
        other_logP = torch.log(other_P + 1e-20)
        # (B, M)
        # 4) H = - sum_m P[b,m] * logP[b,m]
        self_H_mask = -(self_P * self_logP).sum(dim=1)
        other_H_mask = -(other_P * other_logP).sum(dim=1)
        # (B,)
        # 5) KL = sum_m P[b,m] * (logP[b,m] - logQ[b,m])
        kl = (self_P * (self_logP - other_logP)).sum(dim=1)
        # (B,)
        return kl - self_H_mask + other_H_mask

    def sampled_action_logp(self):
        return self._action_logp

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 28  # controls model output feature vector size

ModeCountBinaryDist = ModeCountBinaryDistribution
