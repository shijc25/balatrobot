import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelConfigDict
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet
from modeling.submodules.card_self_attention import CardSelfAttention
from gym_envs.components.card import Card
import math
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
import torch.nn.functional as F
from gym_envs.joker import Joker
from gym_envs.base_card import BaseCard
from gym_envs.universal_card_encoder import UniversalCardEncoder
from modeling.shared_parameters import SharedParameters


class BalatroShopModel(TorchModelV2, nn.Module):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, *args, **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        custom_config = kwargs
        self.sequence_features = [
            "shop_joker_indices",
            "owned_joker_indices",
            "shop_joker_costs",
            "owned_joker_costs",
        ]

        self.card_embedding_size = custom_config.get("card_embedding_size", 64)

        # self.joker_embedding_size = custom_config.get("joker_embedding_size", 64)
        # self.card_segment_embedding_size = custom_config.get(
        #     "card_segment_embedding_size", 8
        # )
        self.attention_layers = custom_config.get("attention_layers", 1)
        self.attention_heads = custom_config.get("attention_heads", 4)
        self.max_jokers = custom_config.get("max_jokers", 5)
        self.max_shop_jokers = custom_config.get("max_shop_jokers", 2)
        self.action_head_depth = custom_config.get("action_head_depth", 2)
        self.value_head_depth = custom_config.get("value_head_depth", 2)
        self.max_hand_size = custom_config.get("max_hand_size", 8)
        self.action_method = custom_config.get("action_method", "convolutional")
        self.shared_encoder = custom_config.get("shared_encoder", False)
        self.activation = nn.GELU

        if self.shared_encoder:
            self.universal_card_encoder = SharedParameters.encoder
        else:
            self.universal_card_encoder = UniversalCardEncoder(
                embedding_size=self.card_embedding_size
            )

        self.blind_embedding = nn.Embedding(30, 16)

        self.state_size = (
            1
            + 6
            + 4
            + 13
            + 16
            + 3
            + BaseCard.num_enhancements
            + BaseCard.num_editions
            + BaseCard.num_seals
            + 1
            + 5 * 12
        )  # dollars, round_info, suits in deck, ranks in deck, blind index, joker count, max owned jokers, jokers at capacity, deck enh/edition/seal, deck size
        self.state_projector = nn.Sequential(
            # nn.LayerNorm(self.state_size),
            nn.Linear(self.state_size, self.card_embedding_size),
            # nn.LayerNorm(self.card_embedding_size),
            self.activation(),
            nn.Linear(self.card_embedding_size, self.card_embedding_size),
        )

        self.self_attention_layer = nn.ModuleList(
            [
                CardSelfAttention(
                    d_model=self.card_embedding_size,
                    n_heads=self.attention_heads,
                    dropout=0.00,
                )
                for _ in range(self.attention_layers)
            ]
        )

        if self.action_method == "convolutional":
            action_layers = []
            for i in range(self.action_head_depth):
                if i == 0:
                    action_layers.append(
                        nn.Linear(
                            self.card_embedding_size * 2 + self.state_size,
                            self.card_embedding_size,
                        )
                    )
                else:
                    action_layers.append(
                        nn.Linear(
                            self.card_embedding_size,
                            self.card_embedding_size,
                        )
                    )
                # action_layers.append(nn.LayerNorm(self.card_embedding_size))
                action_layers.append(self.activation())
            action_layers.append(nn.Linear(self.card_embedding_size, 1))
            self.action_head = nn.Sequential(*action_layers)
        elif self.action_method == "intent_vectors":
            action_layers = []
            for i in range(self.action_head_depth - 1):
                if i == 0:
                    action_layers.append(
                        nn.Linear(
                            self.card_embedding_size + self.state_size,
                            self.card_embedding_size,
                        )
                    )
                else:
                    action_layers.append(
                        nn.Linear(
                            self.card_embedding_size,
                            self.card_embedding_size,
                        )
                    )
                # action_layers.append(nn.LayerNorm(self.card_embedding_size))
                action_layers.append(self.activation())
            action_layers.append(
                nn.Linear(self.card_embedding_size, self.card_embedding_size)
            )
            self.action_head = nn.Sequential(*action_layers)

        value_layers = []
        for _ in range(self.value_head_depth):
            value_layers.append(
                nn.Linear(self.card_embedding_size, self.card_embedding_size)
            )
            # value_layers.append(nn.LayerNorm(self.card_embedding_size))
            value_layers.append(self.activation())
        value_layers.append(nn.Linear(self.card_embedding_size, 1))
        self.value_head = nn.Sequential(*value_layers)

        # self.value_layer = nn.Linear(self.card_embedding_size, 1)

    def scale_costs(self, costs):
        costs -= 3.0
        costs /= 5.0
        return costs

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        dollars = (obs["dollars"].float() / 50.0) - 1
        round = obs["round_info"].float()  # Normalize round to [0, 1]
        deck_stats = obs["deck_stats"].float()
        blind_emb = self.blind_embedding(obs["blind_index"].long()).squeeze(
            1
        )  # Shape: (BATCH_SIZE, 16)
        owned_joker_count = obs["owned_joker_count"]
        max_owned_jokers = obs["max_owned_jokers"]
        jokers_at_capacity = obs["jokers_at_capacity"]
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

        # "Card" here includes special tokens for end shop, end booster, reroll, and context
        card_embeddings, padding_mask = self.universal_card_encoder(obs["all_cards"])

        # In case we are in the initial test observation we need to check for padding mask being 1 everywhere
        if padding_mask.sum() == padding_mask.numel():
            # If all are padded, we create a dummy padding instead
            padding_mask = torch.zeros_like(padding_mask, dtype=torch.bool)

        state_obs = torch.cat(
            [
                dollars.unsqueeze(-1),
                round,
                deck_stats,
                blind_emb,
                owned_joker_count.unsqueeze(-1),
                max_owned_jokers.unsqueeze(-1),
                jokers_at_capacity.unsqueeze(-1),
                obs["hand_stats"],
            ],
            dim=-1,
        )  # Shape: (BATCH_SIZE, 2)
        state_projection = self.state_projector(state_obs)
        state_projection = state_projection.unsqueeze(1)

        all_actions = card_embeddings + state_projection

        for layer in self.self_attention_layer:
            all_actions = layer(all_actions, padding=padding_mask)
        hidden_actions = all_actions

        self._context_token = hidden_actions[:, 0, :]
        hidden_actions = hidden_actions[:, 1:, :]

        action_mask = obs["action_mask"]
        if self.action_method == "convolutional":
            action_input = torch.cat(
                [
                    self._context_token,
                    state_obs,
                ],
                dim=-1,
            )
            action_input = action_input.unsqueeze(1).expand(
                -1, hidden_actions.size(1), -1
            )
            action_input = torch.cat([hidden_actions, action_input], dim=-1)

            action_logits = self.action_head(action_input).squeeze(-1)
        if self.action_method == "intent_vectors":
            intent_input = torch.cat(
                [
                    self._context_token,
                    state_obs,
                ],
                dim=-1,
            )
            intent_vector = self.action_head(intent_input)
            action_logits = torch.einsum("bd, bnd -> bn", intent_vector, hidden_actions)

        action_logits = torch.where(
            action_mask.bool(),
            torch.tensor(-1e9, device=action_logits.device),
            action_logits,
        )

        action_logits = torch.cat([action_logits, obs["num_targets"]], dim=-1)

        return action_logits, state

    def value_function(self):
        return self.value_head(self._context_token).squeeze(1)
