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


class ARChooseOrStopDistribution(TorchDistributionWrapper):
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        self.allow_illegal_actions = model.allow_illegal_actions
        self._action_logp = None
        self._entropy = None

    def cannot_discard_flags(self):
        return self.inputs[:, -1]

    def mode_logits(self):
        return self.inputs[:, :2]

    def hidden_state_logits(self):
        return self.inputs[:, 2:-1]

    def adjust_mode_logits(self, mode_logits, cannot_discard):
        if not self.allow_illegal_actions:
            adjusted_mode_logits = mode_logits.clone()
            adjusted_mode_logits[cannot_discard.bool(), 0] = -1e9
            return adjusted_mode_logits
        return mode_logits

    def sample(self, deterministic=False):
        return self.sample_return_all(deterministic=deterministic)[0]

    def sample_return_all(self, deterministic=False):
        B = self.inputs.shape[0]
        self._action_logp = torch.zeros(
            B, device=self.inputs.device, dtype=torch.float32
        )
        self._entropy = torch.zeros(B, device=self.inputs.device, dtype=torch.float32)

        cannot_discard = self.cannot_discard_flags()
        mode_logits = self.mode_logits()
        mode_logits = self.adjust_mode_logits(mode_logits, cannot_discard)
        mode_dist = Categorical(logits=mode_logits)

        h = self.hidden_state_logits()
        selected_so_far = torch.zeros(
            B, 10, device=self.inputs.device, dtype=torch.float32
        )  # mode + hand + stop
        sampled_mode = mode_dist.sample()
        self._action_logp += mode_dist.log_prob(sampled_mode)
        self._entropy += mode_dist.entropy()
        selected_so_far[:, 0] = sampled_mode

        for i in range(5):
            stopped = selected_so_far[:, -1] == 1
            if stopped.all():
                break
            logits = self.model.ar_step(selected_so_far, h)

            # Force selecting at least 1 card
            if not self.allow_illegal_actions and i == 0:
                logits = logits.clone()
                logits[:, -1] = (
                    -1e9
                )  # Mask out the "stop" action if no cards are selected yet

            # Mask out any logits for already selected actions
            logits = logits.clone()
            logits[selected_so_far[:, 1:] > 0] = -1e9

            next_slot_dist = Categorical(logits=logits)
            next_slot_sample = next_slot_dist.sample()

            # set the select_so_far to 1 IFF the "stop" action has not been selected yet
            rows_to_update = (~stopped).nonzero(as_tuple=False)[:, 0]
            selected_so_far[rows_to_update, next_slot_sample[rows_to_update] + 1] = 1
            self._action_logp[rows_to_update] += next_slot_dist.log_prob(
                next_slot_sample
            )[rows_to_update]
            self._entropy[rows_to_update] += next_slot_dist.entropy()[rows_to_update]

        return selected_so_far[:, :-1], self._action_logp, self._entropy

    def deterministic_sample(self):
        return self.sample(deterministic=True)

    def logp(self, actual_action):
        B, device = actual_action.shape[0], actual_action.device

        # 1) Mode
        cannot_discard = self.cannot_discard_flags()
        mode_logits = self.adjust_mode_logits(self.mode_logits(), cannot_discard)
        mode_act = actual_action[:, 0].long()
        dist_mode = Categorical(logits=mode_logits)
        logp = dist_mode.log_prob(mode_act)  # (B,)

        # 2) Prepare for AR picks
        h = self.hidden_state_logits()
        selected_so_far = torch.zeros(B, 10, device=device)
        selected_so_far[:, 0] = mode_act.float()
        target_mask = actual_action[:, 1:9].bool().clone()  # (B,8)

        # Count how many cards were actually picked
        pick_counts = target_mask.sum(dim=1)  # (B,)

        # 3) Greedy teacher-forcing along the MAP path
        still_active = pick_counts > 0
        for i in range(5):
            if not still_active.any():
                break

            logits = self.model.ar_step(selected_so_far, h)  # (B,9)
            picked = selected_so_far[:, 1:9].bool()

            # build mask of legal actions
            allowed = torch.ones_like(logits, dtype=torch.bool)
            allowed[:, :8] &= ~picked
            # mask STOP on first step
            if i == 0:
                allowed[:, 8] = False
            logits = logits.masked_fill(~allowed, -1e9)

            card_logits = logits[:, :8]
            card_probs = torch.softmax(card_logits, dim=1)

            # pick the most probable **true** card
            masked_logits = card_logits.masked_fill(~target_mask, -1e9)
            jstar = masked_logits.argmax(dim=1)

            idx = still_active.nonzero(as_tuple=True)[0]
            logp[idx] += torch.log(card_probs[idx, jstar[idx]].clamp(min=1e-9))

            # update
            selected_so_far[idx, jstar[idx] + 1] = 1.0
            target_mask[idx, jstar[idx]] = False
            still_active = target_mask.sum(dim=1) > 0

        # 4) STOP token only for those who picked <5
        # mask out cards so that the only legal action is STOP (index 8)
        stop_logits = self.model.ar_step(selected_so_far, h)
        stop_logits[:, :8] = -1e9
        dist_final = Categorical(logits=stop_logits)
        stop_token = torch.full((B,), 8, device=device, dtype=torch.long)

        # find which trajectories actually stopped early
        stopped_early = pick_counts < 5
        idx_stop = stopped_early.nonzero(as_tuple=True)[0]
        if len(idx_stop) > 0:
            logp[idx_stop] += dist_final.log_prob(stop_token)[idx_stop]

        # 2) form a scalar loss (e.g. negative log-prob weighted by a dummy positive advantage)
        # #    Here we just test with all-ones advantage so that loss = -(1 * logp).mean()
        # loss = -logp.mean()
        # loss.backward(retain_graph=True)

        # # 3) check grad norms on your AR head
        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(f"{name}: NO GRAD!")
        #     else:
        #         print(f"{name}: grad norm = {param.grad.norm().item():.3e}")
        return logp  # (B,)

    def entropy(self):
        return self.sample_return_all(deterministic=False)[2]

    def sampled_action_logp(self):
        return self._action_logp

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return model_config["custom_model_config"]["ar_head_hidden_size"]

ARChooseOrStop = ARChooseOrStopDistribution
