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


class ARChooseOrStopStackedDistribution(TorchDistributionWrapper):
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        self.allow_illegal_actions = model.allow_illegal_actions
        self._action_logp = None
        self._entropy = None
        self.hand_size = model.max_supported_hand_size

        self._perms = {m: list(itertools.permutations(range(m))) for m in range(1, 6)}
        self.perm_cap = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
        self.min_played_cards = 1

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

    def sample(self, deterministic=False, get_entropy=False):
        if not get_entropy:
            with torch.no_grad():
                return self._sample(deterministic, get_entropy)
        else:
            return self._sample(deterministic, get_entropy)

    def _sample(self, deterministic=False, get_entropy=False):
        B, device = self.inputs.shape[0], self.inputs.device

        cannot_discard = self.cannot_discard_flags()
        mode_logits = self.adjust_mode_logits(self.mode_logits(), cannot_discard)
        if deterministic:
            sampled_mode = mode_logits.argmax(dim=1)
            mode_logp = torch.zeros(B, device=device)
            mode_ent = torch.zeros(B, device=device)
        else:
            mode_dist = Categorical(logits=mode_logits)
            sampled_mode = mode_dist.sample()
            mode_logp = mode_dist.log_prob(sampled_mode)
            mode_ent = mode_dist.entropy()

        # self._action_logp = torch.zeros(B, device=device, dtype=torch.float32)
        pick_logp_sum = torch.zeros(B, device=device)
        stop_logp_sum = torch.zeros(B, device=device)
        entropy = mode_ent.clone()

        samples = torch.zeros(B, 6, self.hand_size + 2, device=device)
        selected_so_far = torch.zeros(B, self.hand_size + 2, device=device)
        selected_so_far[:, 0] = sampled_mode
        samples[:, 0, 0] = sampled_mode

        h = self.hidden_state_logits()

        for i in range(5):
            stopped = selected_so_far[:, -1] == 1
            if stopped.all():
                break

            logits = self.model.ar_step(selected_so_far, h)  # (B, cur_len+1)

            # enforce at least 1 card on first pick
            if not self.allow_illegal_actions and i < self.min_played_cards:
                logits = logits.clone()
                logits[:, -1] = -1e9

            # mask selected cards
            sel_mask = (selected_so_far[:, 1:] > 0)[:, : logits.size(1)]
            logits = logits.masked_fill(sel_mask, float("-inf"))

            active = ~stopped
            if not active.any():
                break

            act_logits = logits[active]
            # If any active row has all logits highly negative or -inf,
            # make STOP selectable for that row to avoid creating a Categorical
            # with only -inf entries (which leads to NaNs/errors).
            bad_rows = act_logits.max(dim=1).values < -1e8
            if bad_rows.any():
                # clone before modifying so we don't inadvertently change other rows
                act_logits = act_logits.clone()
                # ensure STOP (last column) is selectable
                act_logits[bad_rows, -1] = 0.0
                # write back to logits so downstream bookkeeping sees the change
                logits[active] = act_logits
            dist = Categorical(logits=act_logits)
            if deterministic:
                act_choice = act_logits.argmax(dim=1)
                # optional: entropy += 0 on deterministic
            else:
                act_choice = dist.sample()
                if i == 0:
                    entropy[active] += dist.entropy()

            lp_all = dist.log_prob(act_choice)
            is_stop = act_choice == (act_logits.size(1) - 1)

            # write back (note +1 shift because col 0 is mode)
            selected_so_far[active, act_choice + 1] = 1.0
            samples[active, i + 1, act_choice + 1] = 1.0

            rows = active.nonzero(as_tuple=False).squeeze(1)
            pick_rows = rows[~is_stop]
            stop_rows = rows[is_stop]
            if pick_rows.numel():
                pick_logp_sum[pick_rows] += lp_all[~is_stop]
            if stop_rows.numel():
                stop_logp_sum[stop_rows] += lp_all[is_stop]

        # m = (selected_so_far[:, 1:-1] > 0).sum(dim=1).clamp(min=1)
        # self._action_logp = (
        #     mode_logp + pick_logp_sum + stop_logp_sum + torch.lgamma(m + 1)
        # )
        if not get_entropy:
            self._action_logp = self.logp(samples)

        return entropy if get_entropy else samples

    def deterministic_sample(self):
        return self.sample(deterministic=True)

    def logp(self, samples):
        # Surrogate set logp: at each step, credit the sum of probs over any remaining target cards.

        if samples.sum() == 0:
            return torch.zeros(self.inputs.shape[0], device=self.inputs.device)

        B, T, A = samples.shape
        device = self.inputs.device
        NEG_INF = float("-inf")

        h = self.hidden_state_logits()
        cannot_discard = self.cannot_discard_flags()

        selected_so_far = torch.zeros(B, A, device=device)

        # --- mode term ---
        sampled_mode = samples[:, 0, 0].long()
        mode_logits = self.adjust_mode_logits(self.mode_logits(), cannot_discard)
        mode_dist = Categorical(logits=mode_logits)
        total_logp = mode_dist.log_prob(sampled_mode)
        selected_so_far[:, 0] = sampled_mode.float()

        # Which cards are in the final chosen set? (aggregate across steps; exclude STOP col)
        final_set_mask = samples[:, :, 1:-1].amax(dim=1).bool()  # [B, A-2]

        pick_mass_sum = torch.zeros(B, device=device)
        stop_logp_sum = torch.zeros(B, device=device)

        for i in range(1, T):
            stopped = selected_so_far[:, -1] == 1
            if stopped.all():
                break

            logits = self.model.ar_step(
                selected_so_far, h
            )  # [B, L], where L = (#avail cards) + 1 (STOP)

            # ---- legality/gating identical to sampling ----
            picks = (selected_so_far[:, 1:-1] > 0).sum(
                dim=1
            )  # how many cards already chosen
            logits = logits.clone()

            # forbid STOP until at least `min_played_cards` picked (usually 1)
            if not self.allow_illegal_actions:
                need_one = picks < self.min_played_cards
                logits[need_one, -1] = NEG_INF

            # force STOP if already picked as many cards as there are available
            # options (covers hands smaller than 5). logits.size(1)-1 == #card cols
            available_cards = logits.size(1) - 1
            force_stop = picks >= available_cards
            if force_stop.any():
                rows = force_stop.nonzero(as_tuple=False).squeeze(1)
                logits[rows, :-1] = NEG_INF  # only STOP legal

            # Mask already-picked cards (never mask STOP)
            card_mask = (selected_so_far[:, 1:-1] > 0)[:, : logits.size(1) - 1]
            logits[:, : logits.size(1) - 1] = logits[
                :, : logits.size(1) - 1
            ].masked_fill(card_mask, NEG_INF)

            active = ~stopped
            if not active.any():
                break

            active_idx = active.nonzero(as_tuple=False).squeeze(1)  # [b_act]
            logits_act = logits[active]  # [b_act, L]
            L = logits_act.size(1)
            # if any row in logits_act is all -inf, make STOP the only available option
            all_bad = logits_act.max(dim=1).values < -1e8
            if all_bad.any():
                logits_act = logits_act.clone()
                logits_act[all_bad, :] = NEG_INF
                logits_act[all_bad, -1] = 0.0
            probs_act = torch.softmax(logits_act, dim=1)

            # One-hots for this step (include STOP column; shift by +1)
            step_onehots = samples[active][:, i, 1 : L + 1]  # [b_act, L]
            has_choice = step_onehots.sum(dim=1) > 0
            if not has_choice.any():
                continue

            probs_sel = probs_act[has_choice]  # [b_sel, L]
            onehots_sel = step_onehots[has_choice]  # [b_sel, L]
            idx_in_logits = onehots_sel.argmax(dim=1)  # [b_sel], 0..L-1
            is_stop = idx_in_logits == (L - 1)  # [b_sel]
            rows_sel = active_idx[has_choice]  # [b_sel] original-B rows

            # --- surrogate credit: mass on any remaining target card (exclude STOP col) ---
            # Remaining = cards in final set but not yet picked in selected_so_far.
            remaining_mask = final_set_mask[rows_sel, : L - 1] & (
                ~selected_so_far[rows_sel, 1:L].bool()
            )  # [b_sel, L-1]

            if (~is_stop).any():
                rows_pick = rows_sel[~is_stop]
                mass = (
                    probs_sel[~is_stop, : L - 1] * remaining_mask[~is_stop].float()
                ).sum(
                    dim=1
                )  # [b_pick]
                if (mass < 1e-12).any():
                    print("Some pick masses are very small or zero.")
                pick_mass_sum[rows_pick] += torch.log(mass.clamp_min(1e-12))

            if is_stop.any():
                rows_stop = rows_sel[is_stop]
                stop_mass = probs_sel[is_stop, -1]  # STOP column
                stop_logp_sum[rows_stop] += torch.log(stop_mass.clamp_min(1e-12))

            # advance selected_so_far using the actual chosen index (including STOP)
            idx_in_sel_full = idx_in_logits + 1
            selected_so_far[rows_sel, idx_in_sel_full] = 1.0

        total_logp = total_logp + pick_mass_sum + stop_logp_sum
        if (total_logp < -1e9).any():
            print("Some total_logp values are very small or zero.")
        if torch.isnan(total_logp).any():
            print("NaNs detected in total_logp.")
            print(
                "selected_so_far for NaN rows:",
                selected_so_far[torch.isnan(total_logp)],
            )

        return total_logp

    def entropy(self):
        return self.sample(get_entropy=True)

    def sampled_action_logp(self):
        return self._action_logp

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return model_config["custom_model_config"]["card_embedding_size"]

ARChooseOrStopStacked = ARChooseOrStopStackedDistribution
