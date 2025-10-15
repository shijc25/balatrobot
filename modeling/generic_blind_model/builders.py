from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from gymnasium.spaces import flatten_space

from modeling.submodules.card_self_attention import CardSelfAttention
from modeling.submodules.linear_logit_head import LinearLogitHead
from modeling.submodules.masked_subset_convolution import (
    MaskedSubsetConvolutionModel,
)
from modeling.submodules.next_card_factored_head import NextCardFactoredHead
from modeling.shared_parameters import SharedParameters
from gym_envs.base_card import BaseCard
from gym_envs.universal_card_encoder import UniversalCardEncoder
from modeling.optional_torchrl import noisy_linear


@dataclass
class BackboneDimensions:
    hidden_input: int
    flat_context: int


class BlindModelBuilder:
    """Centralizes the heavy-weight module construction for ``BalatroBlindModel``."""

    def __init__(self, model: nn.Module, obs_space, num_outputs: int) -> None:
        self.model = model
        self.obs_space = obs_space
        self.num_outputs = num_outputs
        self._dimensions: Optional[BackboneDimensions] = None

    def build(self) -> BackboneDimensions:
        self._configure_encoders()
        self._configure_attention()
        self._configure_joker_projectors()
        dims = self._compute_backbone_dimensions()
        self._dimensions = dims
        self._configure_film()
        self._build_hidden_layers(dims.hidden_input)
        self._build_value_head()
        self._build_action_heads(dims)
        self._configure_auxiliary_outputs()
        return dims

    # ------------------------------------------------------------------
    # Encoder + attention setup
    # ------------------------------------------------------------------
    def _configure_encoders(self) -> None:
        model = self.model
        if model.shared_encoder:
            if SharedParameters.encoder is None:
                raise ValueError("SharedParameters.encoder has not been initialized")
            model.universal_card_encoder = SharedParameters.encoder
        else:
            model.universal_card_encoder = UniversalCardEncoder(
                embedding_size=model.card_embedding_size
            )

        if model.blind_obs:
            model.blind_embedding = nn.Embedding(
                model.num_blind_types, model.blind_embedding_size
            )

    def _configure_attention(self) -> None:
        model = self.model
        if model.self_attention_layers <= 0:
            model.self_attention = None
            return
        if not model.embed_cards:
            raise ValueError("Self-attention requires card embeddings to be enabled.")
        model.self_attention = nn.ModuleList(
            [
                CardSelfAttention(
                    d_model=model.card_embedding_size,
                    n_heads=model.self_attention_heads,
                    dropout=0.05,
                    noisy="attention" in model.noisy_layers,
                )
                for _ in range(model.self_attention_layers)
            ]
        )

    def _configure_joker_projectors(self) -> None:
        model = self.model
        activation = model.activation
        model.joker_projector = nn.Sequential(
            nn.Linear(model.card_embedding_size, model.card_embedding_size),
            activation,
            nn.Linear(model.card_embedding_size, model.card_embedding_size),
        )
        model.joker_summary_projector = nn.Sequential(
            nn.Linear(model.card_embedding_size, model.card_embedding_size),
            activation,
            nn.Linear(model.card_embedding_size, model.card_embedding_size),
        )

    # ------------------------------------------------------------------
    # Backbone sizing + FiLM
    # ------------------------------------------------------------------
    def _compute_backbone_dimensions(self) -> BackboneDimensions:
        model = self.model
        input_size = self.obs_space.shape[0]

        if model.embed_cards:
            hand_obs = flatten_space(BaseCard.observation_space(model.hand_size)).shape[0]
            joker_obs = flatten_space(BaseCard.observation_space(model.max_jokers)).shape[0]
            input_size -= hand_obs
            input_size -= joker_obs

            if model.hand_representation_method == "concat":
                input_size += model.card_embedding_size * model.hand_size
            elif model.hand_representation_method == "context_token":
                input_size += model.card_embedding_size

            if not model.jokers_in_hand_attention:
                input_size += model.card_embedding_size

            if model.deck_obs:
                input_size -= model.deck_size
                input_size += model.card_embedding_size

            if model.blind_obs:
                input_size -= 1
                input_size += model.blind_embedding_size

            if model.subset_hand_types_mode is not None:
                input_size -= 8 * model.num_subsets

            if model.scoring_cards_masks is not None:
                input_size -= model.hand_size * model.num_subsets

        if model.hand_representation_method == "concat":
            flat_size = input_size - model.card_embedding_size * model.hand_size
        elif model.hand_representation_method == "context_token":
            flat_size = input_size - model.card_embedding_size
        else:
            flat_size = input_size

        return BackboneDimensions(hidden_input=input_size, flat_context=flat_size)

    def _configure_film(self) -> None:
        model = self.model
        if model.FiLM_mode is None:
            return
        if model.FiLM_mode in ("pre-hand", "post-hand"):
            output_size = model.card_embedding_size * 2
        elif model.FiLM_mode == "hidden":
            output_size = model.hidden_size * 2
        else:
            raise ValueError(f"Unsupported FiLM mode: {model.FiLM_mode}")

        model.FiLM_layer = nn.Sequential(
            nn.Linear(model.card_embedding_size, int(model.hidden_size / 2)),
            nn.LayerNorm(int(model.hidden_size / 2)),
            model.activation,
            nn.Linear(int(model.hidden_size / 2), output_size),
        )

    # ------------------------------------------------------------------
    # Backbone + heads
    # ------------------------------------------------------------------
    def _build_hidden_layers(self, input_size: int) -> None:
        model = self.model
        layers = [nn.Linear(input_size, model.hidden_size)]
        for _ in range(model.hidden_layer_count - 1):
            layers.append(model.activation)
            layers.append(nn.Linear(model.hidden_size, model.hidden_size))
        layers.append(model.activation)
        model.hidden_layers = nn.Sequential(*layers)

    def _build_value_head(self) -> None:
        model = self.model
        model.value_layer = nn.Sequential(
            nn.Linear(model.hidden_size, int(model.hidden_size / 2)),
            model.activation,
            nn.Linear(int(model.hidden_size / 2), 1),
        )

    def _build_action_heads(self, dims: BackboneDimensions) -> None:
        model = self.model
        flat_size = dims.flat_context
        input_size = dims.hidden_input

        if model.action_method == "linear_logits":
            model.action_layer = nn.Linear(model.hidden_size, self.num_outputs)
            return

        if model.action_method == "intent_vectors":
            output_per_expert = (
                model.card_embedding_size * 2
                + (self.num_outputs - model.hand_size * 2)
                + 1
                + 1
            )

            def linear_factory(x: int, y: int) -> nn.Module:
                if "experts" in model.noisy_layers:
                    return noisy_linear(x, y, std_init=0.017)
                return nn.Linear(x, y)

            heads = []
            for _ in range(model.num_experts):
                layers = [
                    linear_factory(input_size + model.card_embedding_size, model.hidden_size),
                    nn.LayerNorm(model.hidden_size),
                    model.activation,
                ]
                for _ in range(model.hidden_layer_count - 1):
                    layers.extend(
                        [
                            linear_factory(model.hidden_size, model.hidden_size),
                            nn.LayerNorm(model.hidden_size),
                            model.activation,
                        ]
                    )
                layers.append(linear_factory(model.hidden_size, output_per_expert))
                heads.append(nn.Sequential(*layers))
            model.expert_heads = nn.ModuleList(heads)
            return

        if model.action_method in {"subset_convolution", "dual_subset"}:
            per_subset_info_size = 0
            if model.subset_hand_types_mode == "obs":
                per_subset_info_size += 8

            num_aux = 4
            model.register_buffer("aux_mean", torch.zeros(num_aux))
            model.register_buffer("aux_var", torch.ones(num_aux))
            model.ema_decay = 0.99

            model.play_layer = MaskedSubsetConvolutionModel(
                d_model=model.card_embedding_size,
                flat_size=flat_size,
                per_subset_info_size=per_subset_info_size,
                expect_scoring_mask=model.scoring_cards_masks == "obs",
                aux_outputs=num_aux,
                max_num_cards=model.hand_size,
                invalidate_non_minimal=model.invalidate_non_minimal,
                dual_action_logits=model.action_method == "dual_subset",
                include_zero_subset=model.action_method == "dual_subset",
                discard_as_intent=model.discard_as_intent,
                intent_size=model.card_embedding_size,
            )

            if model.action_method == "subset_convolution":
                model.discard_layer = MaskedSubsetConvolutionModel(
                    d_model=model.card_embedding_size,
                    flat_size=flat_size,
                    per_subset_info_size=per_subset_info_size,
                    max_num_cards=model.hand_size,
                )
                if model.allow_illegal_actions and model.forced_play_head:
                    model.forced_play_layer = MaskedSubsetConvolutionModel(
                        d_model=model.card_embedding_size,
                        flat_size=flat_size,
                        per_subset_info_size=per_subset_info_size,
                        max_num_cards=model.hand_size,
                    )
            return

        if model.action_method == "autoregressive":
            model.first_ar_h_head = nn.Linear(model.hidden_size, model.card_embedding_size)
            model.mode_layer = nn.Linear(model.hidden_size, 2)
            model.ar_head = NextCardFactoredHead(
                card_embedding_size=model.card_embedding_size,
                num_attention_layers=2,
                card_encoder=model.universal_card_encoder,
            )
            return

        if model.action_method == "convolutional":
            model.card_action_head = nn.Sequential(
                nn.Linear(model.card_embedding_size + flat_size, model.card_embedding_size // 2),
                nn.LayerNorm(model.card_embedding_size // 2),
                model.activation,
                nn.Linear(model.card_embedding_size // 2, 2),
            )
            model.play_discard_logit_layer = nn.Sequential(
                nn.Linear(model.hidden_size, model.hidden_size // 2),
                nn.LayerNorm(model.hidden_size // 2),
                model.activation,
                nn.Linear(model.hidden_size // 2, 2),
            )
            return

        if model.action_method == "linear_experts":
            model.option_layer = nn.Linear(flat_size, model.num_experts)
            model.experts = nn.ModuleList(
                [LinearLogitHead(model.hidden_size, num_cards=model.hand_size) for _ in range(model.num_experts)]
            )
            return

        raise ValueError(f"Unsupported action method: {model.action_method}")

    def _configure_auxiliary_outputs(self) -> None:
        model = self.model
        aux_outputs = 0
        if model.suit_count_aux_coeff > 0:
            aux_outputs += 4
        if model.joker_identity_coeff > 0:
            aux_outputs += model.max_jokers * model.joker_types
        if model.suit_matching_aux_coeff > 0:
            pairs = model.hand_size * (model.hand_size - 1) // 2
            aux_outputs += pairs
        if model.rank_matching_aux_coeff > 0:
            pairs = model.hand_size * (model.hand_size - 1) // 2
            aux_outputs += pairs
        if model.available_hand_types_coeff > 0:
            aux_outputs += 8

        model.aux_outputs = aux_outputs
        if aux_outputs <= 0:
            return
        model.aux_layer = nn.Sequential(
            nn.Linear(model.hidden_size, model.hidden_size),
            model.activation,
            nn.Linear(model.hidden_size, aux_outputs),
        )
