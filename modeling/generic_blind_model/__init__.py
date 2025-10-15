from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations as combos
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.modelv2 import ModelConfigDict
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from gym_envs.base_card import BaseCard

from .builders import BlindModelBuilder
from .config import BlindModelConfig
from .losses import AuxiliaryLossMixin
from .observation import ObservationEncoder, ObservationEncoding


@dataclass
class ActionOutput:
    logits: torch.Tensor
    extra_state: List[torch.Tensor]
    append_cannot_discard: bool = True


class BalatroBlindModel(AuxiliaryLossMixin, TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config: ModelConfigDict,
        name: str,
        *args,
        **kwargs,
    ) -> None:
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        custom_config = kwargs
        self.config = BlindModelConfig.from_kwargs(**custom_config)
        for key, value in self.config.as_kwargs().items():
            setattr(self, key, value)
        self.subset_hand_types_mode = self.config.subset_hand_types

        self.custom_losses_have_been_calced = False
        self.invisible_obs_features = ["subset_hand_types", "scoring_cards_masks"]

        self.hand_size = self.max_supported_hand_size
        self.deck_size = 52
        self.num_subsets = self._calculate_num_subsets(self.hand_size)
        self.activation = nn.GELU()

        self._builder = BlindModelBuilder(self, obs_space, num_outputs)
        self._builder.build()

        self._observation_encoder = ObservationEncoder(self)
        self._init_internal_state()

    @staticmethod
    def _calculate_num_subsets(hand_size: int, max_subset_size: int = 5) -> int:
        total = 0
        for size in range(1, max_subset_size + 1):
            total += len(list(combos(range(hand_size), size)))
        return total

    def _init_internal_state(self) -> None:
        self._last_context: Optional[torch.Tensor] = None
        self._hand: Optional[torch.Tensor] = None
        self._hand_padding: Optional[torch.Tensor] = None
        self._first_ar_h: Optional[torch.Tensor] = None
        self._has_jokers: Optional[torch.Tensor] = None
        self._last_hand_obs = None

        self._last_aux_outputs: Optional[torch.Tensor] = None
        self._last_suit_loss: Optional[float] = None
        self._last_joker_aux_loss: Optional[float] = None
        self._last_suit_entropy_loss: Optional[float] = None
        self._last_rank_entropy_loss: Optional[float] = None
        self._last_suit_matching_loss: Optional[float] = None
        self._last_rank_matching_loss: Optional[float] = None
        self._last_valid_card_count_loss: Optional[float] = None
        self._last_available_hand_types_loss: Optional[float] = None
        self._last_option_variation_loss: Optional[float] = None
        self._last_intent_similarity_loss: Optional[float] = None
        self._last_weight_decay_loss: Optional[float] = None
        self._last_joker_spread_loss: Optional[float] = None
        self._last_hand_score_loss: Optional[float] = None
        self._last_curiosity_bonus: Optional[float] = None
        self._last_joker_pca_evs: List[float] = []
        self.play_aux: Optional[torch.Tensor] = None

    @staticmethod
    def index_to_suit_rank_index(index: int) -> tuple[int, int]:
        suit = index // 13
        rank = index % 13
        return suit, rank

    def mean_jokers(
        self, jokers: Optional[torch.Tensor], padding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if jokers is None or jokers.shape[1] == 0:
            return torch.zeros(
                (jokers.shape[0], self.card_embedding_size),
                dtype=jokers.dtype,
                device=jokers.device,
            )

        if padding is None:
            padding = torch.zeros(
                (jokers.shape[0], jokers.shape[1]),
                dtype=torch.bool,
                device=jokers.device,
            )

        valid_mask = ~padding.unsqueeze(-1)
        jokers_sum = (jokers * valid_mask).sum(dim=1, keepdim=True)
        jokers_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)

        jokers_mean = jokers_sum / jokers_count
        return jokers_mean.squeeze(1)

    def prepend_special_token(self, cards_obs, u_idx):
        device = cards_obs["indices"].device
        batch_size = cards_obs["indices"].shape[0]

        special_card = BaseCard(
            segment=BaseCard.Segments.SPECIAL_TOKEN,
            u_index=u_idx + BaseCard.FIRST_SPECIAL_TOKEN_INDEX,
        )
        special_card_obs = BaseCard.observe_list([special_card], 1)
        special_card_obs = {
            key: torch.tensor(value, device=device).expand(*((batch_size,) + value.shape))
            for key, value in special_card_obs.items()
        }

        combined_obs = {
            key: torch.cat([tensor, cards_obs[key]], dim=1)
            for key, tensor in special_card_obs.items()
        }
        return combined_obs

    def forward(self, input_dict, state, seq_lens):
        encoding = self._observation_encoder.encode(input_dict)

        hidden_output = self.hidden_layers(encoding.hidden_inputs)
        if self.FiLM_mode == "hidden" and encoding.film_scale is not None:
            hidden_output = hidden_output * (1 + encoding.film_scale) + encoding.film_shift
        self._last_context = hidden_output

        if self.aux_outputs > 0:
            self._last_aux_outputs = self.aux_layer(self._last_context)

        action_output = self._compute_action_logits(encoding)
        logits = action_output.logits
        if action_output.append_cannot_discard:
            cannot_discard = input_dict["obs"]["cannot_discard"].to(torch.float32)
            logits = torch.cat([logits, cannot_discard], dim=1)

        self.action_logits = logits
        return self.action_logits, action_output.extra_state

    def _compute_action_logits(self, encoding: ObservationEncoding) -> ActionOutput:
        obs = encoding.obs

        if self.action_method == "linear_logits":
            logits = self.action_layer(self._last_context)
            return ActionOutput(logits=logits, extra_state=[])

        if self.action_method == "intent_vectors":
            if encoding.expert_contexts is None:
                raise ValueError(
                    "Intent vectors require expert contexts in the observation encoding."
                )

            expert_outputs = []
            for i in range(self.num_experts):
                expert_context = encoding.expert_contexts[:, i, :]
                expert_inputs = torch.cat(
                    [encoding.hidden_inputs, expert_context],
                    dim=1,
                )
                expert_outputs.append(self.expert_heads[i](expert_inputs))
            expert_outputs = torch.stack(expert_outputs, dim=1)

            self.option_logits = expert_outputs[:, :, -1]
            self.value_predictions = expert_outputs[:, :, -2]
            self.play_intent = expert_outputs[:, :, : self.card_embedding_size]
            self.discard_intent = expert_outputs[
                :, :, self.card_embedding_size : 2 * self.card_embedding_size
            ]
            other_action_logits = expert_outputs[
                :, :, 2 * self.card_embedding_size : -2
            ]

            option_mask = obs["option_mask"]
            self.option_logits = torch.where(
                option_mask.bool(),
                torch.tensor(-1e9, device=self.option_logits.device),
                self.option_logits,
            )

            play_scores = torch.einsum("bne, bpe -> bpn", self._hand, self.play_intent)
            discard_scores = torch.einsum(
                "bne, bpe -> bpn", self._hand, self.discard_intent
            )

            expert_hand_padding = encoding.hand_padding[:, self.num_experts + 1 :]
            expert_hand_padding = expert_hand_padding.unsqueeze(1).expand(
                -1,
                self.num_experts,
                -1,
            )
            play_scores = torch.where(
                expert_hand_padding.bool(),
                torch.tensor(-1e9, device=play_scores.device),
                play_scores,
            )
            discard_scores = torch.where(
                expert_hand_padding.bool(),
                torch.tensor(-1e9, device=discard_scores.device),
                discard_scores,
            )

            expert_logits = torch.cat(
                [
                    other_action_logits,
                    discard_scores,
                    play_scores,
                ],
                dim=2,
            )
            sub_logits = expert_logits.flatten(start_dim=1)
            logits = torch.cat([sub_logits, self.option_logits], dim=1)
            return ActionOutput(logits=logits, extra_state=[])

        if self.action_method in {"subset_convolution", "dual_subset"}:
            if encoding.without_hand_for_actions is None:
                raise ValueError(
                    "Subset-based heads require joker context outside the shared attention path."
                )

            per_subset_info = None
            if self.subset_hand_types_mode == "obs":
                per_subset_info = obs["subset_hand_types"]
            scoring_masks = None
            if self.scoring_cards_masks == "obs":
                scoring_masks = obs["scoring_cards_masks"]

            play_logits, action_subset_idx = self.play_layer(
                self._hand,
                encoding.without_hand_for_actions,
                per_subset_info,
                scoring_masks,
                cards=obs["hand"],
                hand_embeddings=self._hand,
                hand_padding=encoding.hand_padding,
            )
            self.play_aux = play_logits

            if self.action_method == "dual_subset":
                logits = torch.cat(
                    [play_logits, action_subset_idx.unsqueeze(2)],
                    dim=2,
                )
                return ActionOutput(
                    logits=logits,
                    extra_state=[],
                    append_cannot_discard=False,
                )

            discard_logits, _ = self.discard_layer(
                self._hand,
                encoding.without_hand_for_actions,
                per_subset_info,
                cards=obs["hand"],
            )

            if self.allow_illegal_actions and self.forced_play_head:
                forced_play_logits, _ = self.forced_play_layer(
                    self._hand,
                    encoding.without_hand_for_actions,
                    per_subset_info,
                    cards=obs["hand"],
                )
                discard_logits = torch.where(
                    obs["cannot_discard"].bool(),
                    forced_play_logits,
                    discard_logits,
                )
            else:
                discard_logits = torch.where(
                    obs["cannot_discard"].bool(),
                    torch.tensor(-1e9, device=discard_logits.device),
                    discard_logits,
                )

            if play_logits.dim() == 3 and play_logits.size(-1) == 1:
                play_logits = play_logits.squeeze(-1)

            logits = torch.cat([play_logits, discard_logits], dim=1)
            return ActionOutput(
                logits=logits,
                extra_state=[],
                append_cannot_discard=False,
            )

        if self.action_method == "autoregressive":
            self.mode_logits = self.mode_layer(self._last_context)
            self._first_ar_h = self.first_ar_h_head(self._last_context)
            logits = torch.cat([self.mode_logits, self._first_ar_h], dim=1)
            return ActionOutput(logits=logits, extra_state=[])

        if self.action_method == "convolutional":
            self.mode_logits = self.play_discard_logit_layer(self._last_context)
            hand_with_non_seq = torch.cat(
                [
                    self._hand,
                    encoding.non_sequence_tensor.unsqueeze(1).expand(
                        -1,
                        self._hand.shape[1],
                        -1,
                    ),
                ],
                dim=2,
            )
            logits = self.card_action_head(hand_with_non_seq)
            logits = logits.permute(0, 2, 1)
            logits = logits.flatten(start_dim=1)
            logits = torch.cat([self.mode_logits, logits], dim=1)
            return ActionOutput(logits=logits, extra_state=[])

        if self.action_method == "linear_experts":
            self.option_logits = self.option_layer(encoding.non_sequence_tensor)
            option_mask = obs["option_mask"]
            self.option_logits = torch.where(
                option_mask.bool(),
                torch.tensor(-1e9, device=self.option_logits.device),
                self.option_logits,
            )
            logits = torch.cat(
                [self.option_logits]
                + [expert(self._last_context) for expert in self.experts],
                dim=-1,
            )
            return ActionOutput(logits=logits, extra_state=[])

        raise ValueError(f"Unsupported action method: {self.action_method}")

    def value_function(self):
        if self.action_method == "intent_vectors":
            option_probs = F.softmax(self.option_logits, dim=1)
            values = self.value_predictions
            return (option_probs * values).sum(dim=1) + self.value_layer(
                self._last_context
            ).squeeze(1)
        return self.value_layer(self._last_context).squeeze(1)

    def ar_step(self, already_selected, h, hand_wo_context=None, hand_padding=None):
        mode = already_selected[:, 0]
        selected_cards_mask = already_selected[:, 1:]

        if hand_wo_context is None:
            hand_wo_context = self._hand
        if hand_padding is None:
            hand_padding = self._hand_padding[:, 1:]

        ar_logits = self.ar_head(
            mode,
            hand_wo_context,
            hand_padding,
            selected_cards_mask,
            h_tokens=h.unsqueeze(1),
            card_obs=self._last_hand_obs,
        )
        return ar_logits

        if hand_wo_context is None:
            hand_wo_context = self._hand
        if hand_padding is None:
            hand_padding = self._hand_padding

        ar_logits = self.ar_head(
            mode,
            selected_cards_mask,
            h,
            hand_embeddings=hand_wo_context,
            hand_padding=hand_padding,
        )
        return ar_logits


__all__ = [
    "BalatroBlindModel",
    "BlindModelConfig",
    "BlindModelBuilder",
    "ObservationEncoder",
    "ObservationEncoding",
    "AuxiliaryLossMixin",
]