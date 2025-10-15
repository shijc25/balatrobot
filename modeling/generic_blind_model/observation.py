from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from gym_envs.base_card import BaseCard


@dataclass
class ObservationEncoding:
    obs: Dict[str, torch.Tensor]
    non_sequence_tensor: torch.Tensor
    hidden_inputs: torch.Tensor
    without_hand_for_actions: Optional[torch.Tensor]
    hand: torch.Tensor
    hand_context: torch.Tensor
    hand_padding: torch.Tensor
    expert_contexts: Optional[torch.Tensor]
    joker_context: Optional[torch.Tensor]
    film_scale: Optional[torch.Tensor]
    film_shift: Optional[torch.Tensor]


class ObservationEncoder:
    """Encapsulates observation preprocessing for ``BalatroBlindModel``."""

    def __init__(self, model) -> None:
        self.model = model

    def encode(self, input_dict) -> ObservationEncoding:
        model = self.model
        obs = input_dict["obs"]

        non_sequence_features: List[str] = sorted(obs.keys())
        for feature in model.invisible_obs_features:
            if feature in non_sequence_features:
                non_sequence_features.remove(feature)

        model.hand_card_indices = obs["hand"]["suit"] * 13 + obs["hand"]["rank"]
        model._last_hand_obs = obs["hand"]

        non_sequence_features.remove("hand")
        non_sequence_features.remove("jokers")

        if "blind_index" in obs:
            non_sequence_features.remove("blind_index")
            obs["blind_embedding"] = model.blind_embedding(
                obs["blind_index"].to(torch.int64)
            ).squeeze(1)
            non_sequence_features.append("blind_embedding")

        if "hand_stats" in obs:
            obs["hand_stats"] = torch.concat(
                [
                    obs["hand_stats"]["level"],
                    obs["hand_stats"]["chips"],
                    obs["hand_stats"]["mult"],
                    obs["hand_stats"]["played_count"],
                    obs["hand_stats"]["played_this_blind"],
                ],
                dim=1,
            )

        hand_with_context = model.prepend_special_token(
            obs["hand"],
            BaseCard.SpecialTokens.HAND_CONTEXT,
        )

        if model.action_method == "intent_vectors":
            for idx in range(model.num_experts):
                hand_with_context = model.prepend_special_token(
                    hand_with_context, BaseCard.SpecialTokens.expert_context(idx)
                )

        hand, hand_padding = model.universal_card_encoder(hand_with_context)
        model._hand_padding = hand_padding

        jokers, joker_padding = model.universal_card_encoder(
            model.prepend_special_token(
                obs["jokers"], BaseCard.SpecialTokens.JOKER_CONTEXT
            )
            if not model.jokers_in_hand_attention
            else obs["jokers"]
        )

        expert_contexts: Optional[torch.Tensor] = None
        joker_context: Optional[torch.Tensor] = None
        film_scale: Optional[torch.Tensor] = None
        film_shift: Optional[torch.Tensor] = None

        if model.jokers_in_hand_attention:
            combined_embeddings = torch.cat([hand, jokers], dim=1)
            combined_padding = torch.cat([hand_padding, joker_padding], dim=1)
            for layer in range(model.self_attention_layers):
                hand = model.self_attention[layer](combined_embeddings, combined_padding)
            if model.action_method == "intent_vectors":
                expert_contexts = hand[:, : model.num_experts, :]
                hand = hand[:, model.num_experts :, :]
            hand_context = hand[:, 0, :]
            hand = hand[:, 1 : 1 + model.hand_size, :]
        else:
            joker_context = jokers[:, 0, :]
            jokers = jokers[:, 1:, :]
            joker_context = model.joker_projector(jokers)
            joker_context = torch.where(
                joker_padding[:, 1:].unsqueeze(-1),
                torch.tensor(0.0, device=joker_context.device),
                joker_context,
            )
            num_jokers = (~joker_padding[:, 1:]).sum(dim=1)
            joker_context = joker_context.sum(dim=1)
            mean_context = joker_context / (num_jokers.unsqueeze(-1) + 1e-6)
            joker_context = model.joker_summary_projector(joker_context)
            joker_context += mean_context

            model._has_jokers = num_jokers > 0
            joker_context = torch.where(
                model._has_jokers.unsqueeze(-1),
                joker_context,
                torch.zeros_like(joker_context),
            )

            if model.FiLM_mode is not None:
                film_params = model.FiLM_layer(
                    joker_context.reshape(joker_context.shape[0], -1)
                )
                film_params = film_params.view(joker_context.shape[0], -1, 2)
                film_scale = film_params[:, :, 0]
                film_shift = film_params[:, :, 1]
                if model.FiLM_mode == "pre-hand":
                    hand = hand * (1 + film_scale.unsqueeze(1)) + film_shift.unsqueeze(1)

            for layer in range(model.self_attention_layers):
                hand = model.self_attention[layer](hand, hand_padding)

            if model.FiLM_mode == "post-hand":
                hand = hand * (1 + film_scale.unsqueeze(1)) + film_shift.unsqueeze(1)

            hand_context = hand[:, 0, :]
            hand = hand[:, 1:, :]

        model._hand = hand

        non_sequence_tensor = torch.concat(
            [input_dict["obs"][feature] for feature in non_sequence_features],
            dim=1,
        )

        hidden_inputs = non_sequence_tensor
        without_hand_for_actions: Optional[torch.Tensor] = None

        if not model.jokers_in_hand_attention:
            without_hand_for_actions = torch.cat(
                [hidden_inputs, joker_context],
                dim=1,
            )
            hidden_inputs = torch.cat(
                [hidden_inputs, joker_context.detach()],
                dim=1,
            )

        if model.hand_representation_method == "concat":
            hidden_inputs = torch.cat(
                [hidden_inputs, hand.reshape(hand.shape[0], -1)],
                dim=1,
            )
        elif model.hand_representation_method == "context_token":
            hidden_inputs = torch.cat([hidden_inputs, hand_context], dim=1)

        return ObservationEncoding(
            obs=obs,
            non_sequence_tensor=non_sequence_tensor,
            hidden_inputs=hidden_inputs,
            without_hand_for_actions=without_hand_for_actions,
            hand=hand,
            hand_context=hand_context,
            hand_padding=hand_padding,
            expert_contexts=expert_contexts,
            joker_context=joker_context,
            film_scale=film_scale,
            film_shift=film_shift,
        )
