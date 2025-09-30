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
from modeling.submodules.linear_logit_head import LinearLogitHead


class ExpertOptionsDistribution(TorchDistributionWrapper):
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
        legal = ExpertOptionsDistribution.build_legal_masks()
        self.legal_masks = legal.to(self.inputs.device)  # (218, 8)

        self.allow_illegal_actions = model.allow_illegal_actions
        self.expert_sample_separate = model.expert_sample_separate
        self._action_logp = None
        self._cards_logp = None
        self._entropy = None
        self.num_experts = model.num_experts
        self.hand_size = model.hand_size

    def cannot_discard_flags(self):
        return self.inputs[:, -1]

    def expert_locked_flags(self):
        return self.inputs[:, -self.num_experts - 1 : -1]

    def option_logits(self):
        return self.inputs[:, : self.num_experts]

    def option_distribution(self):
        option_logits = self.option_logits()
        option_dist = Categorical(logits=option_logits)
        # Blend the options with a uniform distribution
        epsilon = 0.00
        option_dist = Categorical(
            probs=option_dist.probs * (1 - epsilon) + epsilon / self.num_experts
        )

        return option_dist

    def expert_logits(self, selected_expert):
        B, total_dims = self.inputs.shape
        E = self.num_experts
        S = LinearLogitHead.num_outputs()

        # 1) Split off controller logits (first E dims) and extra logit (last dim)
        #    and keep just the middle block [E : E + E*S]
        expert_block = self.inputs[:, E : E + E * S]  # → [B, E*S]

        # 2) Reshape into [batch, expert, expert_size]
        expert_block = expert_block.view(B, E, S)  # → [B, E, S]

        # 3) Index into that with a LongTensor of expert IDs per batch-row
        batch_idx = torch.arange(B, device=self.inputs.device)
        logits = expert_block[batch_idx, selected_expert]  # → [B, S]

        return logits

    def mode_logits(self, selected_expert):
        return self.expert_logits(selected_expert)[:, :2]

    def play_logits(self, selected_expert):
        return self.expert_logits(selected_expert)[:, 2 : 2 + self.hand_size]

    def discard_logits(self, selected_expert):
        return self.expert_logits(selected_expert)[
            :, 2 + self.hand_size : 2 + 2 * self.hand_size
        ]

    def card_logits(self, selected_expert, mode):
        play_logits = self.play_logits(selected_expert)
        discard_logits = self.discard_logits(selected_expert)

        logits = torch.where(
            mode.unsqueeze(-1).bool(),
            play_logits,
            discard_logits,
        )
        return logits

    def card_logits_i(self, selected_expert, mode):
        play_logits = self.play_logits(selected_expert)
        discard_logits = self.discard_logits(selected_expert)

        if mode == 1:
            logits = play_logits
        elif mode == 0:
            logits = discard_logits

        return logits

    def adjust_mode_logits(self, mode_logits, cannot_discard):
        if not self.allow_illegal_actions:
            adjusted_mode_logits = mode_logits.clone()
            adjusted_mode_logits[cannot_discard.bool(), 0] = -1e9
            return adjusted_mode_logits
        return mode_logits

    def sample(self, deterministic=False):
        B = self.inputs.shape[0]

        option_dist = self.option_distribution()
        if deterministic:
            selected_expert = option_dist.probs.argmax(dim=1)
        else:
            selected_expert = option_dist.sample()

        cannot_discard = self.cannot_discard_flags()
        mode_logits = self.mode_logits(selected_expert)
        mode_logits = self.adjust_mode_logits(mode_logits, cannot_discard)
        mode_dist = Categorical(logits=mode_logits)

        if deterministic:
            mode = mode_logits.argmax(dim=1)
        else:
            mode = mode_dist.sample()

        # Sample cards based on the selected expert and mode
        cards = self.sample_cards(selected_expert, mode, deterministic)

        action = torch.cat(
            [
                selected_expert.unsqueeze(-1),  # (B, 1)
                mode.unsqueeze(-1),  # (B, 1)
                cards,  # (B, 8)
            ],
            dim=-1,
        )

        # Need to recalculate to cover all the possible experts
        self._action_logp = self.logp(action)

        return action

    def sample_cards(self, expert, mode, deterministic=False):
        B = self.inputs.shape[0]
        self._cards_logp = torch.zeros(
            B, device=self.inputs.device, dtype=torch.float32
        )
        # logits = self.card_logits(mode)  # shape (B,8)
        # inside sample_cards(), instead of independent Bernoulli + fixups:
        logits = self.card_logits(expert, mode)  # (B, 8)
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
        self._cards_logp += dist.log_prob(idx)

        # pick out the 8‐bit mask for each row:
        cards = self.legal_masks[idx]  # (B, 8)
        return cards

    def deterministic_sample(self):
        return self.sample(deterministic=True)

    def logp(self, samples):
        """Compute the true log-prob of `samples` under the expert-mixture policy."""
        # 1) get the expert-choice distribution and its log-probs
        option_logits = self.option_logits()  # (B, E)
        option_dist = Categorical(logits=option_logits)

        # Disconnecting the option sampling and mode/action selection, as if it was two independent actions
        if self.expert_sample_separate:
            chosen_expert = samples[:, 0].long()  # (B,)
            option_logp = option_dist.log_prob(chosen_expert)  # (B,)
            action_logp = self.logp_expert(chosen_expert, samples[:, 1:])
            return option_logp + action_logp  # (B,)
        else:
            log_pi = torch.log(option_dist.probs + 1e-9)  # (B, E)
            # 2) compute log-prob under each expert i:  log p(x | i)
            logp_experts = []
            for i in range(self.num_experts):
                # make logp_expert return only the conditional log-prob,
                # *not* including log π_i
                logp_experts.append(self.logp_expert(i, samples))
            logp_expert_tensor = torch.stack(logp_experts, dim=1)  # (B, E)

            # 3) add in log π_i to get joint log(π_i · p(x|i))
            joint = log_pi + logp_expert_tensor  # (B, E)

            # 4) marginalize i via logsumexp:
            return torch.logsumexp(joint, dim=1)  # (B,)

    def logp_expert(self, expert, action):
        B = action.size(0)
        logp = torch.zeros(B, device=action.device)

        # 1) mode term
        mode = action[:, 0].long()
        mode_logits = self.adjust_mode_logits(
            self.mode_logits(expert), self.cannot_discard_flags()
        )
        mode_dist = Categorical(logits=mode_logits)
        logp += mode_dist.log_prob(mode)

        # 2) cards term
        cards = action[:, 1:]
        logp += self.cards_logp_expert(cards, expert, mode)

        return logp  # (B,) conditional on this expert

    def cards_logp_expert(self, cards_mask, expert, mode):
        """
        cards_mask : (B, 8)  binary float tensor AFTER your clamp (1-5 ones)
        returns    : (B,)    joint log-prob under the truncated Bernoulli product
        """
        logits = self.card_logits(expert, mode)  # (B, 8)
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

    def sampled_action_logp(self):
        return self._action_logp

    def entropy(self):
        B = self.inputs.shape[0]
        self._entropy = torch.zeros(B, device=self.inputs.device, dtype=torch.float32)

        option_dist = self.option_distribution()
        self._entropy += option_dist.entropy()

        if self.expert_sample_separate:
            # If we sample experts separately, we only take the entropy of the argmax expert
            selected_expert = option_dist.probs.argmax(dim=1)
            self._entropy += self.expert_entropy(selected_expert)
            return self._entropy
        else:
            for i in range(self.num_experts):
                expert_prob = option_dist.probs[:, i]
                self._entropy += expert_prob * self.expert_entropy(i)

            return self._entropy

    def expert_entropy(self, expert):
        entropy = torch.zeros(
            self.inputs.shape[0], device=self.inputs.device, dtype=torch.float32
        )
        mode_logits = self.mode_logits(expert)
        mode_logits = self.adjust_mode_logits(mode_logits, self.cannot_discard_flags())
        mode_dist = Categorical(logits=mode_logits)
        entropy += mode_dist.entropy()

        # Add the entropy of the cards
        for mode in range(2):
            cards_entropy = self.cards_entropy(expert, mode)
            entropy += cards_entropy * mode_dist.probs[:, mode]
        return entropy  # (B,)

    def cards_entropy(self, expert, mode):
        """
        Returns: (B,)  the entropy of the truncated‐Bernoulli over all subsets S with 1 <= |S| <= 5.
        Uses the same `self.legal_masks` that you built in `build_legal_masks()`.
        """

        # 1) get the raw logits and probabilities for each card
        card_logits = self.card_logits_i(expert, mode)  # (B, 8)

        if self.allow_illegal_actions:
            base_dist = torch.distributions.Bernoulli(logits=card_logits)
            return base_dist.entropy().sum(dim=1)

        p = torch.sigmoid(card_logits)  # (B, 8)

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

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return model_config["custom_model_config"]["ar_head_hidden_size"]

ExpertOptionsDist = ExpertOptionsDistribution
