from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelConfigDict
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet
from modeling.submodules.card_self_attention import CardSelfAttention
from gym_envs.pseudo.card import Card
import math
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
import torch.nn.functional as F
from gym_envs.joker import Joker
from gym_envs.universal_card_encoder import UniversalCardEncoder
from gymnasium.spaces import flatten_space
from gym_envs.base_card import BaseCard
from modeling.submodules.linear_logit_head import LinearLogitHead
from modeling.submodules.next_card_attention_head import NextCardAttentionHead
from modeling.submodules.next_card_factored_head import NextCardFactoredHead
from modeling.shared_parameters import SharedParameters
from modeling.submodules.multi_head_mlp import MultiHeadMLP
from modeling.submodules.masked_subset_convolution import (
    MaskedSubsetConvolutionModel,
)
import torchrl
import torchrl.modules
from ray.rllib.policy.sample_batch import SampleBatch
from itertools import combinations as combos
from torch.quantization import quantize_dynamic
import gc
from sklearn.decomposition import PCA


class BalatroBlindModel(TorchModelV2, nn.Module):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, *args, **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # custom_config = model_config.get("custom_model_config", {})
        custom_config = kwargs
        self.embed_cards = custom_config.get("embed_cards", True)
        self.card_embedding_size = custom_config.get("card_embedding_size", 32)
        self.embed_suits_ranks = custom_config.get(
            "embed_suits_ranks", None
        )  # None, "learned", "manual", "manual_learned"
        self.hidden_size = custom_config.get("hidden_size", 1024)
        self.hidden_layer_count = custom_config.get("hidden_layer_count", 1)
        # self.context_size = custom_config.get("context_size", 256)
        self.action_method = custom_config.get("action_method", "linear_logits")
        self.ar_head_hidden_size = custom_config.get("ar_head_hidden_size", 64)
        # self.num_intents = custom_config.get("num_intents", 1)
        self.hand_representation_method = custom_config.get(
            "hand_representation_method", "concat"
        )

        self.self_attention_layers = custom_config.get("self_attention_layers", 1)
        self.self_attention_heads = custom_config.get("self_attention_heads", 4)
        self.max_jokers = custom_config.get("max_jokers", 0)
        self.deck_obs = custom_config.get("deck_obs", True)
        self.joker_identity_coeff = custom_config.get("joker_identity_coeff", False)
        self.joker_types = custom_config.get("joker_types", 151)  # including null
        self.suit_rank_entropy_coeff = custom_config.get("suit_rank_entropy_coeff", 0.0)
        # self.pre_FiLM = custom_config.get("pre_FiLM", False)
        # self.post_FiLM = custom_config.get("post_FiLM", False)
        self.FiLM_mode = custom_config.get(
            "FiLM_mode", None
        )  # "pre-hand", "post-hand", "hidden", None
        # self.FiLM_jokers = self.pre_FiLM or self.post_FiLM
        self.suit_matching_aux_coeff = custom_config.get("suit_matching_aux_coeff", 0.0)
        self.rank_matching_aux_coeff = custom_config.get("rank_matching_aux_coeff", 0.0)
        self.allow_illegal_actions = custom_config.get("allow_illegal_actions", False)
        self.valid_card_count_coeff = custom_config.get("valid_card_count_coeff", 0.0)
        self.joker_spread_loss_coeff = custom_config.get("joker_spread_loss_coeff", 0.0)
        self.weight_decay_coeff = custom_config.get("weight_decay_coeff", 0.0)
        self.blind_obs = custom_config.get("blind_obs", True)
        self.num_blind_types = custom_config.get("num_blind_types", 30)
        self.blind_embedding_size = custom_config.get("blind_embedding_size", 16)
        self.num_experts = custom_config.get("num_experts", 3)
        self.available_hand_types_coeff = custom_config.get(
            "available_hand_types_coeff", 0.0
        )
        self.expert_sample_separate = custom_config.get("expert_sample_separate", True)
        self.custom_losses_have_been_calced = False
        self.invisible_obs_features = [
            # "available_hand_types",
            "subset_hand_types",
            "scoring_cards_masks",
        ]
        self.jokers_in_hand_attention = custom_config.get(
            "jokers_in_hand_attention", False
        )
        self.option_variation_coeff = custom_config.get("option_variation_coeff", 0.0)
        self.suit_count_aux_coeff = custom_config.get("suit_count_aux_coeff", 0.0)
        self.intent_similarity_coeff = custom_config.get("intent_similarity_coeff", 0.0)
        self.forced_play_head = custom_config.get("forced_play_head", False)
        self.noisy_layers = custom_config.get(
            "noisy_layers", []
        )  # ["attention", "experts"]
        self.subset_hand_types_mode = custom_config.get("subset_hand_types", None)
        self.scoring_cards_masks = custom_config.get("scoring_cards_masks", None)
        self.hand_score_aux_coeff = custom_config.get("hand_score_aux_coeff", 0.0)
        self.max_supported_hand_size = custom_config.get("max_supported_hand_size", 8)
        self.invalidate_non_minimal = custom_config.get("invalidate_non_minimal", True)
        self.shared_encoder = custom_config.get("shared_encoder", False)
        self.discard_as_intent = custom_config.get("discard_as_intent", False)

        self.hand_size = self.max_supported_hand_size
        self.deck_size = 52
        hand_indices_obs_size = self.hand_size
        joker_indices_obs_size = 0

        self.num_subsets = 0
        for size in range(1, 6):
            self.num_subsets += len(list(combos(range(self.hand_size), size)))

        # self.activation = nn.Tanh()
        # self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.activation = nn.GELU()

        print(self.card_embedding_size, self.shared_encoder)
        if self.shared_encoder:
            self.universal_card_encoder = SharedParameters.encoder
        else:
            self.universal_card_encoder = UniversalCardEncoder(
                embedding_size=self.card_embedding_size
            )

        if self.blind_obs:
            self.blind_embedding = nn.Embedding(
                self.num_blind_types, self.blind_embedding_size
            )

        if self.self_attention_layers > 0:
            if not self.embed_cards:
                raise ValueError(
                    "Self-attention requires card embeddings to be enabled."
                )
            self.self_attention = nn.ModuleList(
                [
                    CardSelfAttention(
                        d_model=self.card_embedding_size,
                        n_heads=self.self_attention_heads,
                        dropout=0.05,
                        noisy="attention" in self.noisy_layers,
                    )
                    for _ in range(self.self_attention_layers)
                ]
            )

            # self.joker_attention = CardSelfAttention(
            #     d_model=self.card_embedding_size,
            #     n_heads=self.self_attention_heads,
            #     dropout=0.00,
            #     noisy="attention" in self.noisy_layers,
            # )

        self.joker_projector = nn.Sequential(
            nn.Linear(self.card_embedding_size, self.card_embedding_size),
            self.activation,
            nn.Linear(self.card_embedding_size, self.card_embedding_size),
        )
        self.joker_summary_projector = nn.Sequential(
            nn.Linear(self.card_embedding_size, self.card_embedding_size),
            self.activation,
            nn.Linear(self.card_embedding_size, self.card_embedding_size),
        )

        input_size = obs_space.shape[0]
        # input_size -= 8  # available hand types
        # print(input_size)
        if self.embed_cards:
            input_size -= flatten_space(
                BaseCard.observation_space(self.hand_size)
            ).shape[0]
            input_size -= flatten_space(
                BaseCard.observation_space(self.max_jokers)
            ).shape[0]

            if self.hand_representation_method == "concat":
                input_size += (
                    self.card_embedding_size * self.hand_size
                )  # Add the embedded hand back in
            elif self.hand_representation_method == "context_token":
                input_size += self.card_embedding_size
            if not self.jokers_in_hand_attention:
                input_size += self.card_embedding_size  # Add the embedded joker context
            if self.deck_obs:
                input_size -= 52  # Remove deck indices from obs space
                input_size += self.card_embedding_size
            if self.blind_obs:
                input_size -= 1
                input_size += self.blind_embedding_size
            if self.subset_hand_types_mode is not None:
                input_size -= (
                    8 * self.num_subsets
                )  # Remove the subset hand types from obs space because it is either getting special handling or being used as a target instead of a feature
            if self.scoring_cards_masks is not None:
                input_size -= self.hand_size * self.num_subsets
        if self.hand_representation_method == "concat":
            flat_size = input_size - self.card_embedding_size * self.hand_size
        elif self.hand_representation_method == "context_token":
            flat_size = input_size - self.card_embedding_size

        if self.FiLM_mode != None:
            if self.FiLM_mode in ["pre-hand", "post-hand"]:
                output_size = self.card_embedding_size * 2
            elif self.FiLM_mode == "hidden":
                output_size = self.hidden_size * 2
            self.FiLM_layer = nn.Sequential(
                nn.Linear(
                    self.card_embedding_size,
                    int(self.hidden_size / 2),
                ),
                nn.LayerNorm(int(self.hidden_size / 2)),
                self.activation,
                nn.Linear(int(self.hidden_size / 2), output_size),
            )

        self.hidden_layers = []
        # self.hidden_layers.append(nn.LayerNorm(input_size))
        self.hidden_layers.append(
            nn.Linear(
                input_size,
                self.hidden_size,
            )
        )
        for _ in range(self.hidden_layer_count - 1):
            # self.hidden_layers.append(nn.LayerNorm(int(self.hidden_size)))
            self.hidden_layers.append(self.activation)
            self.hidden_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        # self.hidden_layers.append(nn.LayerNorm(int(self.hidden_size)))
        self.hidden_layers.append(self.activation)
        self.hidden_layers = nn.Sequential(*self.hidden_layers)

        # self.context_layer = nn.Linear(self.hidden_size, self.context_size)
        num_value_heads = 1
        if self.action_method == "intent_vectors":
            num_value_heads = self.num_experts
        self.value_layer = nn.Sequential(
            nn.Linear(self.hidden_size, int(self.hidden_size / 2)),
            # nn.LayerNorm(int(self.hidden_size / 2)),
            self.activation,
            nn.Linear(int(self.hidden_size / 2), 1),
        )
        if self.action_method == "linear_logits":
            self.action_layer = nn.Linear(self.hidden_size, num_outputs)
        elif self.action_method == "intent_vectors":
            output_per_expert = (
                (self.card_embedding_size * 2)  # play and discard intents
                + (num_outputs - self.hand_size * 2)  # other action logits
                + 1  # option logits
                + 1  # value head
            )
            # self.expert_heads = MultiHeadMLP(
            #     layer_dims=[input_size]
            #     + [self.hidden_size] * self.hidden_layer_count
            #     + [output_per_expert],
            #     num_heads=self.num_experts,
            # )

            noise = lambda x, y: torchrl.modules.NoisyLinear(x, y, std_init=0.017)
            lin_cls = noise if "experts" in self.noisy_layers else nn.Linear
            self.expert_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        lin_cls(
                            input_size + self.card_embedding_size, self.hidden_size
                        ),  # extra input size for the individual expert context embeddings
                        nn.LayerNorm(self.hidden_size),
                        self.activation,
                        *[
                            nn.Sequential(
                                lin_cls(self.hidden_size, self.hidden_size),
                                nn.LayerNorm(self.hidden_size),
                                self.activation,
                            )
                            for _ in range(self.hidden_layer_count - 1)
                        ],
                        lin_cls(self.hidden_size, output_per_expert),
                    )
                    for _ in range(self.num_experts)
                ]
            )
        elif self.action_method in ["subset_convolution", "dual_subset"]:
            per_subset_info_size = 0
            if self.subset_hand_types_mode == "obs":
                per_subset_info_size += 8  # num hand types

            num_aux = 4
            self.register_buffer("aux_mean", torch.zeros(num_aux))
            self.register_buffer("aux_var", torch.ones(num_aux))
            self.ema_decay = 0.99

            self.play_layer = MaskedSubsetConvolutionModel(
                d_model=self.card_embedding_size,
                flat_size=flat_size,
                per_subset_info_size=per_subset_info_size,
                expect_scoring_mask=self.scoring_cards_masks == "obs",
                aux_outputs=num_aux,
                max_num_cards=self.hand_size,
                invalidate_non_minimal=self.invalidate_non_minimal,
                dual_action_logits=self.action_method == "dual_subset",
                include_zero_subset=self.action_method == "dual_subset",
                discard_as_intent=self.discard_as_intent,
                intent_size=self.card_embedding_size,
            )

            if self.action_method == "subset_convolution":
                self.discard_layer = MaskedSubsetConvolutionModel(
                    d_model=self.card_embedding_size,
                    flat_size=flat_size,
                    per_subset_info_size=per_subset_info_size,
                    max_num_cards=self.hand_size,
                )
                if self.allow_illegal_actions and self.forced_play_head:
                    self.forced_play_layer = MaskedSubsetConvolutionModel(
                        d_model=self.card_embedding_size,
                        flat_size=flat_size,
                        per_subset_info_size=per_subset_info_size,
                        max_num_cards=self.hand_size,
                    )
        elif self.action_method == "autoregressive":
            # h = (
            #     self.ar_head_hidden_size - self.hand_size - 2
            # )  # -1 for stop selecting -1 for mode
            self.first_ar_h_head = nn.Linear(
                self.hidden_size, self.card_embedding_size
            )  # putting state into attention, needs same size
            # self.first_ar_h_head = nn.Linear(self.hidden_size, self.ar_head_hidden_size)
            self.mode_layer = nn.Linear(self.hidden_size, 2)
            # self.ar_head = nn.Sequential(
            #     nn.Linear(self.ar_head_hidden_size, h),
            #     # nn.LayerNorm(h),
            #     self.activation,
            #     nn.Linear(h, self.hand_size + 1),  # +1 for stop selecting
            # )
            # self.ar_head = NextCardAttentionHead(
            #     self.card_embedding_size,
            #     num_attention_layers=2,
            #     card_encoder=self.universal_card_encoder,
            # )
            self.ar_head = NextCardFactoredHead(
                card_embedding_size=self.card_embedding_size,
                num_attention_layers=2,
                card_encoder=self.universal_card_encoder,
            )

        elif self.action_method == "convolutional":
            self.card_action_head = nn.Sequential(
                nn.Linear(
                    self.card_embedding_size + flat_size,
                    int(self.card_embedding_size / 2),
                ),
                nn.LayerNorm(int(self.card_embedding_size / 2)),
                self.activation,
                nn.Linear(int(self.card_embedding_size / 2), 2),
            )

            self.play_discard_logit_layer = nn.Sequential(
                nn.Linear(self.hidden_size, int(self.hidden_size / 2)),
                nn.LayerNorm(int(self.hidden_size / 2)),
                self.activation,
                nn.Linear(int(self.hidden_size / 2), 2),
            )
        elif self.action_method == "linear_experts":
            self.option_layer = nn.Linear(flat_size, self.num_experts)
            self.experts = nn.ModuleList(
                [
                    LinearLogitHead(
                        self.hidden_size,
                        num_cards=self.hand_size,
                    )
                    for _ in range(self.num_experts)
                ]
            )

        self.aux_outputs = 0
        if self.suit_count_aux_coeff > 0:
            self.aux_outputs += 4
        if self.joker_identity_coeff > 0:
            self.aux_outputs += self.max_jokers * self.joker_types
        if self.suit_matching_aux_coeff > 0:
            unordered_pairs = self.hand_size * (self.hand_size - 1) // 2
            self.aux_outputs += unordered_pairs
        if self.rank_matching_aux_coeff > 0:
            unordered_pairs = self.hand_size * (self.hand_size - 1) // 2
            self.aux_outputs += unordered_pairs
        if self.available_hand_types_coeff > 0:
            self.aux_outputs += 8

        if self.aux_outputs > 0:
            # self.aux_layer = nn.Linear(self.hidden_size, self.aux_outputs)
            self.aux_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                # nn.LayerNorm(self.hidden_size),
                self.activation,
                nn.Linear(self.hidden_size, self.aux_outputs),
            )

        self._last_context = None
        self._hand = None
        self._last_suit_loss = None
        self._last_joker_aux_loss = None
        self._last_suit_entropy_loss = None
        self._last_rank_entropy_loss = None
        self._last_suit_matching_loss = None
        self._last_rank_matching_loss = None
        self._last_valid_card_count_loss = None
        self._last_available_hand_types_loss = None
        self._last_option_variation_loss = None
        self._last_intent_similarity_loss = None
        self._last_weight_decay_loss = None
        self._last_joker_spread_loss = None
        self._last_hand_score_loss = None
        self._last_curiosity_bonus = None
        self._last_joker_pca_evs = []

    @staticmethod
    def index_to_suit_rank_index(index):
        suit = index // 13
        rank = index % 13
        return suit, rank

    def mean_jokers(self, jokers, padding=None):
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

        # Safely compute the average of jokers, excluding masked slots
        valid_mask = ~padding.unsqueeze(-1)

        jokers_sum = (jokers * valid_mask).sum(dim=1, keepdim=True)
        jokers_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)

        jokers_mean = jokers_sum / jokers_count
        return jokers_mean.squeeze(1)

    def prepend_special_token(self, cards_obs, u_idx):
        device = cards_obs["indices"].device
        B = cards_obs["indices"].shape[0]

        special_card = BaseCard(
            segment=BaseCard.Segments.SPECIAL_TOKEN,
            u_index=u_idx + BaseCard.FIRST_SPECIAL_TOKEN_INDEX,
        )
        special_card_obs = BaseCard.observe_list([special_card], 1)
        # expand the special card observation to match the batch size
        special_card_obs = {
            k: torch.tensor(v, device=device).expand(*((B,) + v.shape))
            for k, v in special_card_obs.items()
        }

        combined_obs = {
            k: torch.cat(
                [v, cards_obs[k]],
                dim=1,
            )
            for k, v in special_card_obs.items()
        }
        return combined_obs

    def forward(self, input_dict, state, seq_lens):
        non_sequence = input_dict["obs"].keys()
        # Enforce consistent order of non-sequence features
        non_sequence = list(sorted(non_sequence))

        for x in self.invisible_obs_features:
            if x in non_sequence:
                non_sequence.remove(x)

        obs = input_dict["obs"]
        self.hand_card_indices = obs["hand"]["suit"] * 13 + obs["hand"]["rank"]
        self._last_hand_obs = obs["hand"]

        non_sequence.remove("hand")
        non_sequence.remove("jokers")

        if "blind_index" in obs:
            non_sequence.remove("blind_index")
            non_sequence.append("blind_embedding")
            obs["blind_embedding"] = self.blind_embedding(
                obs["blind_index"].to(torch.int64)
            ).squeeze(1)

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

        hand_w_context = self.prepend_special_token(
            obs["hand"],
            BaseCard.SpecialTokens.HAND_CONTEXT,
        )
        if self.action_method == "intent_vectors":
            for i in range(self.num_experts):
                hand_w_context = self.prepend_special_token(
                    hand_w_context, BaseCard.SpecialTokens.expert_context(i)
                )

        hand, hand_padding = self.universal_card_encoder(hand_w_context)
        self._hand_padding = hand_padding

        # Super hacky, but just zero the hand_padding so we don't choke on the test obs that rllib sends before training
        # hand_padding = torch.zeros_like(hand_padding, dtype=torch.bool)

        jokers, joker_padding = self.universal_card_encoder(
            self.prepend_special_token(
                obs["jokers"],
                BaseCard.SpecialTokens.JOKER_CONTEXT,
            )
            if not self.jokers_in_hand_attention
            else obs["jokers"]
        )

        if self.jokers_in_hand_attention:
            combined_embeddings = torch.cat(
                [hand, jokers],
                dim=1,
            )
            combined_padding = torch.cat(
                [hand_padding, joker_padding],
                dim=1,
            )
            for i in range(self.self_attention_layers):
                hand = self.self_attention[i](combined_embeddings, combined_padding)
            # Remove the context token and the jokers
            if self.action_method == "intent_vectors":
                expert_contexts = hand[:, : self.num_experts, :]
                hand = hand[:, self.num_experts :, :]
            hand_context = hand[:, 0, :]
            hand = hand[:, 1 : 1 + self.hand_size, :]
        else:
            # jokers = self.joker_attention(jokers, joker_padding)
            joker_context = jokers[:, 0, :]
            jokers = jokers[:, 1:, :]  # Remove the context token
            joker_context = self.joker_projector(jokers)
            # remove the padding jokers
            joker_context = torch.where(
                joker_padding[:, 1:].unsqueeze(-1),
                torch.tensor(0.0, device=joker_context.device),
                joker_context,
            )
            num_jokers = (~joker_padding[:, 1:]).sum(dim=1)
            joker_context = joker_context.sum(dim=1)
            mean_context = joker_context / (num_jokers.unsqueeze(-1) + 1e-6)
            joker_context = self.joker_summary_projector(joker_context)
            joker_context += mean_context

            # Try to avoid destroying the summary model while learning round 1
            self._has_jokers = num_jokers > 0
            joker_context = torch.where(
                self._has_jokers.unsqueeze(-1),
                joker_context,
                torch.zeros_like(joker_context),
            )

            if self.FiLM_mode is not None:
                film_params = self.FiLM_layer(
                    joker_context.reshape(joker_context.shape[0], -1)
                )
                film_params = film_params.view(
                    joker_context.shape[0],
                    -1,
                    2,
                )
                film_scale = film_params[:, :, 0]
                film_shift = film_params[:, :, 1]

            if self.FiLM_mode == "pre-hand":
                hand = hand * (1 + film_scale.unsqueeze(1)) + film_shift.unsqueeze(1)

            for i in range(self.self_attention_layers):
                hand = self.self_attention[i](hand, hand_padding)

            if self.FiLM_mode == "post-hand":
                hand = hand * (1 + film_scale.unsqueeze(1)) + film_shift.unsqueeze(1)

            hand_context = hand[:, 0, :]
            hand = hand[:, 1:, :]  # Remove the context token

        self._hand = hand
        non_sequence = torch.concat(
            [input_dict["obs"][feature] for feature in non_sequence],
            dim=1,
        )

        hidden_inputs = non_sequence
        if not self.jokers_in_hand_attention:
            without_hand_for_actions = torch.cat(
                [hidden_inputs, joker_context],
                dim=1,
            )

            # Only policy/aux heads get to grad the joker context
            # Value head must use the joker definitions the policy develops
            hidden_inputs = torch.cat(
                [hidden_inputs, joker_context.detach()],
                dim=1,
            )
        if self.hand_representation_method == "concat":
            hidden_inputs = torch.cat(
                [hidden_inputs, hand.reshape(hand.shape[0], -1)], dim=1
            )
        elif self.hand_representation_method == "context_token":
            hidden_inputs = torch.cat([hidden_inputs, hand_context], dim=1)

        hidden_output = self.hidden_layers(hidden_inputs)
        if self.FiLM_mode == "hidden":
            hidden_output = hidden_output * (1 + film_scale) + film_shift
        self._last_context = hidden_output

        if self.aux_outputs > 0:
            self._last_aux_outputs = self.aux_layer(self._last_context)

        if self.action_method == "linear_logits":
            self.action_logits = self.action_layer(self._last_context)
        elif self.action_method == "intent_vectors":
            expert_outputs = []
            for i in range(self.num_experts):
                expert_context = expert_contexts[:, i, :]
                expert_inputs = torch.cat(
                    [hidden_inputs, expert_context],
                    dim=1,
                )
                expert_outputs.append(self.expert_heads[i](expert_inputs))
            expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, D)

            # expert_outputs = self.expert_heads(hidden_inputs)
            self.option_logits = expert_outputs[:, :, -1]
            self.value_predictions = expert_outputs[:, :, -2]
            self.play_intent = expert_outputs[:, :, : self.card_embedding_size]
            self.discard_intent = expert_outputs[
                :, :, self.card_embedding_size : 2 * self.card_embedding_size
            ]
            other_action_logits = expert_outputs[
                :, :, 2 * self.card_embedding_size : -2
            ]

            # self.option_logits = self.option_layer(without_hand)
            option_mask = obs["option_mask"]
            self.option_logits = torch.where(
                option_mask.bool(),
                torch.tensor(-1e9, device=self.option_logits.device),
                self.option_logits,
            )

            # dot product the attended hand embeddings with the play and discard intents
            play_scores = torch.einsum("bne, bpe -> bpn", self._hand, self.play_intent)
            discard_scores = torch.einsum(
                "bne, bpe -> bpn", self._hand, self.discard_intent
            )

            # Mask play and discard scores with the hand padding so we don't try to play or discard padding slots
            expert_hand_padding = hand_padding[:, self.num_experts + 1 :]
            num_padded = torch.sum(expert_hand_padding)
            if num_padded > 0:
                print(num_padded)
            expert_hand_padding = expert_hand_padding.unsqueeze(1).expand(
                -1,
                self.num_experts,
                -1,
            )  # (B, num_experts, hand_size)
            play_scores = torch.where(
                expert_hand_padding.bool(),
                torch.tensor(-1e9, device=play_scores.device),
                play_scores,
            )  # (B, num_experts, hand_size)
            discard_scores = torch.where(
                expert_hand_padding.bool(),
                torch.tensor(-1e9, device=discard_scores.device),
                discard_scores,
            )  # (B, num_experts, hand_size)

            expert_logits = torch.cat(
                [
                    other_action_logits,
                    discard_scores,
                    play_scores,
                ],
                dim=2,
            )  # (B, num_experts, n)
            sub_logits = expert_logits.flatten(start_dim=1)  # (B, num_experts * n)
            self.action_logits = torch.cat([sub_logits, self.option_logits], dim=1)
        elif self.action_method in ["subset_convolution", "dual_subset"]:
            per_subset_info = None
            if self.subset_hand_types_mode == "obs":
                per_subset_info = obs["subset_hand_types"]
            scoring_masks = None
            if self.scoring_cards_masks == "obs":
                scoring_masks = obs["scoring_cards_masks"]
            action_logits, action_subset_idx = self.play_layer(
                self._hand,
                without_hand_for_actions,
                per_subset_info,
                scoring_masks,
                cards=obs["hand"],
                hand_embeddings=self._hand,
                hand_padding=hand_padding,
            )
            # play_logits, play_aux = self.play_layer(
            #     self._hand,
            #     without_hand_for_actions,
            #     per_subset_info,
            #     scoring_masks,
            #     cards=obs["hand"],
            # )

            if self.action_method == "dual_subset":
                self.action_logits = torch.cat(
                    [action_logits, action_subset_idx.unsqueeze(2)], dim=2
                )
            else:
                discard_logits, discard_aux = self.discard_layer(
                    self._hand,
                    without_hand_for_actions,
                    per_subset_info,
                    cards=obs["hand"],
                )

                if self.allow_illegal_actions and self.forced_play_head:
                    forced_play_logits, fp_aux = self.forced_play_layer(
                        self._hand,
                        without_hand_for_actions,
                        per_subset_info,
                        cards=obs["hand"],
                    )
                    discard_logits = torch.where(
                        obs["cannot_discard"].bool(),
                        forced_play_logits,
                        discard_logits,
                    )  # (B, hand_size)
                else:
                    discard_logits = torch.where(
                        obs["cannot_discard"].bool(),
                        torch.tensor(-1e9, device=discard_logits.device),
                        discard_logits,
                    )  # (B, hand_size)

                self.action_logits = torch.cat(
                    [
                        play_logits,
                        discard_logits,
                    ],
                    dim=1,
                )

            return self.action_logits, []
        elif self.action_method == "autoregressive":
            # self.action_logits = self.ar_hidden_init(self._last_context)
            # self._first_ar_h = self.action_logits
            self.mode_logits = self.mode_layer(self._last_context)
            self._first_ar_h = self.first_ar_h_head(self._last_context)
            self.action_logits = torch.cat(
                [
                    self.mode_logits,
                    self._first_ar_h,
                ],
                dim=1,
            )
            # self.ar_head.invalidate_cache()
        elif self.action_method == "convolutional":
            self.mode_logits = self.play_discard_logit_layer(self._last_context)
            hand_with_non_seq = torch.cat(
                [
                    self._hand,
                    non_sequence.unsqueeze(1).expand(
                        -1,
                        self._hand.shape[1],
                        -1,
                    ),
                ],
                dim=2,
            )

            self.action_logits = self.card_action_head(hand_with_non_seq)
            self.action_logits = self.action_logits.permute(0, 2, 1)  # (B, 2, SLOTS)
            self.action_logits = self.action_logits.flatten(
                start_dim=1
            )  # (B, 2 * SLOTS)
            self.action_logits = torch.cat(
                [
                    self.mode_logits,
                    self.action_logits,
                ],
                dim=1,
            )  # (B, 2 + 2 * SLOTS)
        elif self.action_method == "linear_experts":
            self.option_logits = self.option_layer(non_sequence)

            # If option_mask is provided, use it to disable options
            option_mask = obs["option_mask"]
            self.option_logits = torch.where(
                option_mask.bool(),
                torch.tensor(-1e9, device=self.option_logits.device),
                self.option_logits,
            )
            # print(torch.isnan(self._last_context).any())
            self.action_logits = torch.cat(
                [self.option_logits]
                + [expert(self._last_context) for expert in self.experts],
                dim=-1,
            )

            # print(torch.isnan(self.action_logits).any())

        cannot_discard = input_dict["obs"]["cannot_discard"].to(torch.float32)
        self.action_logits = torch.cat(
            [
                self.action_logits,
                cannot_discard,
            ],
            dim=1,
        )
        return self.action_logits, []

    def value_function(self):
        if self.action_method == "intent_vectors":
            # For intent vectors, we have to weight the value function based on the expert softmax
            option_probs = F.softmax(self.option_logits, dim=1)
            # values = self.value_layer(self._last_context)
            values = self.value_predictions
            return (option_probs * values).sum(dim=1) + self.value_layer(
                self._last_context
            ).squeeze(1)
        # elif self.action_method == "subset_attention":
        #     action_probs = F.softmax(self.action_logits, dim=1)
        #     values = self.subset_values
        #     return (action_probs * values).sum(dim=1) + self.value_layer(
        #         self._last_context
        #     ).squeeze(1)

        return self.value_layer(self._last_context).squeeze(1)

    # In your outer model
    def ar_step(self, already_selected, h, hand_wo_context=None, hand_padding=None):
        mode = already_selected[:, 0]
        selected_cards_mask = already_selected[:, 1:]

        # Use provided slices if given; otherwise fall back to full batch caches
        if hand_wo_context is None:
            hand_wo_context = self._hand  # [B, N, D]
        if hand_padding is None:
            hand_padding = self._hand_padding[:, 1:]  # [B, N]  (drop context)

        logits = self.ar_head(
            mode,
            hand_wo_context,
            hand_padding,
            selected_cards_mask,
            h_tokens=h.unsqueeze(1),  # [B, 1, D]
            card_obs=self._last_hand_obs,
        )
        return logits  # [b, N+1]  (cards + stop)

    @property
    def aux_logits(self):
        if self.aux_outputs > 0:
            return self._last_aux_outputs
        else:
            return None

    def joker_aux_loss(self, policy_loss, loss_inputs, aux_logits):
        # try to predict the jokers that were in the input
        # reshape to be a 3D tensor of shape [batch_size, max_jokers, 151]
        batch_size = aux_logits.shape[0]
        max_jokers = self.max_jokers
        aux_logits = aux_logits.view(batch_size, self.joker_types, max_jokers)
        mask = Joker.implemented_mask()
        mask = (
            mask.to(aux_logits.device)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(batch_size, -1, max_jokers)
        )
        aux_logits = torch.where(
            mask, aux_logits, torch.tensor(-1e9, device=aux_logits.device)
        )

        # Need to get the joker indices from the observation, but first we undo the flattening
        # of the joker indices in the input_dict
        original_obs = restore_original_dimensions(
            loss_inputs["obs"].cpu(), self.obs_space, "torch"
        )
        actual_jokers = original_obs["jokers"]["indices"].to(aux_logits.device)
        actual_jokers = actual_jokers.to(torch.int64)
        actual_jokers = actual_jokers - BaseCard.FIRST_JOKER_INDEX + 1
        actual_jokers = torch.clamp(actual_jokers, min=0)

        ce_loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
        aux_loss = ce_loss(aux_logits, actual_jokers)
        self._last_joker_aux_loss = aux_loss.detach().cpu().item()
        return aux_loss

    def suit_rank_entropy_loss(self, loss_inputs):
        """
        Computes the *global* rank‐ and suit‐entropy for all “done” & valid hands in the batch,
        and returns (H_rank_global + H_suit_global).  Concretely:

        1) Filter to only those timesteps where done==True and reward>=0 (valid plays).
        2) From each such obs, extract the first SLOTS entries of `hand_indices` so that
        SLOTS = self.action_logits.shape[1].  (This ensures we only look at exactly
        the card‐slots your policy is choosing between.)
        3) For each hand b in that filtered set, compute
            q_{b,r} = 1 - ∏_{i : rank[b,i] = r} (1 - p_{b,i}),
        where p_{b,i} = sigmoid(logit_{b,i}), and rank[b,i] = (hand_indices[b,i] % 13).
        4) Average q_{b,r} over b to get \overline{q}_r.  Normalize p_r = \overline{q}_r / ∑_r \overline{q}_r.
        5) H_rank_global = -∑_r p_r log(p_r + eps).  (Similarly for suits.)
        6) Return (H_rank_global + H_suit_global).

        This exactly matches “entropy of the aggregated rank probabilities,” so if your logging
        callback also computes
        p_r_agg = scored_ranks_flat[r] / ∑_r scored_ranks_flat[r]
        H_metric = -∑_r p_r_agg log(p_r_agg + eps),
        they will be proportional (in fact, numerically identical).
        """

        # 1) Build the “valid play” mask: done & non-negative reward
        done_mask = loss_inputs["dones"].bool()  # shape (TOTAL_STEPS,)
        done_mask &= loss_inputs["rewards"] >= 0.0  # filter out discards if reward<0

        if not done_mask.any():
            # No valid “done & play” steps in this batch → zero entropy
            return torch.tensor(0.0, device=self.action_logits.device)

        # 2) Extract only those observations & logits
        obs_done = loss_inputs["obs"][
            done_mask
        ]  # a dict; e.g. obs_done["hand_indices"] is (B, 18)
        logits_done = self.action_logits[
            done_mask, : self.hand_size
        ]  # shape (B, SLOTS)

        # 3) Restore dims to get the original “hand_indices” as LongTensor (B, 18)
        original_obs = restore_original_dimensions(
            obs_done.cpu(), self.obs_space, "torch"
        )
        hand_indices_full = original_obs["hand_indices"].to(
            self.action_logits.device
        )  # (B, 18)

        # 4) Slice exactly the first SLOTS columns so that we match logits_done.shape[1]
        SLOTS = logits_done.shape[1]  # e.g. 8
        hand_indices = hand_indices_full[:, :SLOTS]  # shape (B, SLOTS)

        # 5) Compute p_{b,i} = sigmoid(logit_{b,i})  for each slot
        probs = torch.sigmoid(logits_done)  # shape (B, SLOTS)

        # 6) ---- RANK side: build mask and q_{b,r} ----

        # a) Decode rank[b,i] = hand_indices[b,i] % 13, shape (B, SLOTS)
        hand_ranks = (hand_indices % 13).long()  # dtype=long

        # b) Build a boolean mask of shape (B, 13, SLOTS):
        #    mask_rank[b, r, i] = 1 if hand_ranks[b,i] == r, else 0
        device = probs.device
        rank_range = torch.arange(13, device=device).view(1, 13, 1)  # shape (1,13,1)
        mask_rank = (
            hand_ranks.unsqueeze(1) == rank_range
        ).float()  # shape (B,13,SLOTS)

        # c) Compute for each (b,r):  prod_{i: rank[b,i]==r} (1 - p[b,i])
        one_minus = (1.0 - probs).unsqueeze(1)  # shape (B, 1, SLOTS)
        # (one_minus * mask_rank) picks out (1-p_{b,i}) only for i with rank r
        # (1 - mask_rank) is 1 for i not in that rank, so those factors contribute a “1” to the product.
        prod_terms_r = (one_minus * mask_rank + (1.0 - mask_rank)).prod(
            dim=2
        )  # shape (B,13)

        # d) q_ranks[b,r] = 1 - product_term
        q_ranks = 1.0 - prod_terms_r  # shape (B,13), each entry in [0,1]

        # e) Average across batch to get \overline{q}_r
        mean_q_ranks = q_ranks.mean(dim=0)  # shape (13,)

        # f) Normalize into a proper distribution p_r = mean_q_ranks / (sum + eps)
        eps = 1e-8
        sum_qr = mean_q_ranks.sum()
        p_r = mean_q_ranks / (sum_qr + eps)  # shape (13,), sums ≈ 1

        # g) Compute global rank-entropy: H_rank_global = -∑_r p_r log(p_r + eps)
        H_rank_global = -torch.sum(p_r * torch.log(p_r + eps))  # scalar

        # Optionally store for debugging
        self._last_rank_entropy_loss = H_rank_global.detach().cpu().item()

        # 7) ---- SUIT side: build mask and q_{b,s} ----

        # a) Decode suit[b,i] = hand_indices[b,i] // 13, shape (B, SLOTS)
        hand_suits = (hand_indices // 13).long()

        # b) Build a boolean mask of shape (B, 4, SLOTS):
        suit_range = torch.arange(4, device=device).view(1, 4, 1)
        mask_suit = (hand_suits.unsqueeze(1) == suit_range).float()  # (B,4,SLOTS)

        # c) Compute prod_{i: suit[b,i]==s} (1 - p[b,i]) same as above
        prod_terms_s = (one_minus * mask_suit + (1.0 - mask_suit)).prod(
            dim=2
        )  # shape (B,4)

        # d) q_suits[b,s] = 1 - product
        q_suits = 1.0 - prod_terms_s  # shape (B,4)

        # e) Average across batch to get \overline{q}_s
        mean_q_suits = q_suits.mean(dim=0)  # shape (4,)

        # f) Normalize into p_s = mean_q_suits / (sum + eps)
        sum_qs = mean_q_suits.sum()
        p_s = mean_q_suits / (sum_qs + eps)  # shape (4,)

        # g) Compute global suit‐entropy: H_suit_global = -∑_s p_s log(p_s + eps)
        H_suit_global = -torch.sum(p_s * torch.log(p_s + eps))  # scalar

        self._last_suit_entropy_loss = H_suit_global.detach().cpu().item()

        # 8) Return the sum of both global entropies
        return H_rank_global  # + H_suit_global

    def suit_aux_loss(self, policy_loss, loss_inputs, aux_logits):
        original_obs = restore_original_dimensions(
            loss_inputs["obs"].cpu(), self.obs_space, "torch"
        )
        hand_suits = original_obs["hand"]["suit"].to(self.action_logits.device) - 1
        hand_suits = hand_suits.long()
        # Count of each suit in the hand
        suit_counts = torch.zeros(
            (hand_suits.shape[0], 4), device=hand_suits.device, dtype=torch.float32
        )
        for suit in range(4):
            suit_counts[:, suit] = (hand_suits == suit).sum(dim=1)

        # Normalize the counts to get probabilities
        num_suits = suit_counts.sum(dim=1, keepdim=True)
        # Avoid division by zero
        num_suits = torch.clamp(num_suits, min=1.0)
        suit_probs = suit_counts.float() / num_suits.float()

        pred_probs = torch.sigmoid(aux_logits)
        suit_count_loss = F.mse_loss(pred_probs, suit_probs, reduction="mean")

        # suit_count_loss = F.binary_cross_entropy_with_logits(
        #     aux_logits, suit_probs, reduction="mean"
        # )
        self._last_suit_loss = suit_count_loss.detach().cpu().item()

        return suit_count_loss

    # Auxiliary loss for recognizing which cards in hands have matching suits
    def suit_matching_aux_loss(self, policy_loss, loss_inputs, aux_logits):
        original_obs = restore_original_dimensions(
            loss_inputs["obs"].cpu(), self.obs_space, "torch"
        )
        hand_suits = original_obs["hand"]["suit"].to(self.action_logits.device) - 1
        s1 = hand_suits.unsqueeze(1)  # (B, 1, 8)
        s2 = hand_suits.unsqueeze(2)  # (B, 8, 1)
        suit_matching_mask = (s1 == s2).float()

        H = hand_suits.shape[1]  # Hand size, e.g., 8
        tri_mask = torch.triu(torch.ones(H, H, device=hand_suits.device), diagonal=1)
        target = suit_matching_mask[:, tri_mask.bool()].float()  # (B, unordered_pairs)

        num_positive = target.sum()
        num_negative = target.numel() - num_positive
        num_positive = torch.clamp(num_positive, min=1.0)  # Avoid division by zero
        pos_weight = num_negative / num_positive

        suit_matching_loss = F.binary_cross_entropy_with_logits(
            aux_logits, target, reduction="mean", pos_weight=pos_weight
        )
        self._last_suit_matching_loss = suit_matching_loss.detach().cpu().item()
        return suit_matching_loss

    # Auxiliary loss for recognizing which cards in hands have matching ranks
    def rank_matching_aux_loss(self, policy_loss, loss_inputs, aux_logits):
        original_obs = restore_original_dimensions(
            loss_inputs["obs"].cpu(), self.obs_space, "torch"
        )
        # hand_indices = original_obs["hand_indices"].to(
        #     self.action_logits.device
        # )  # (B, 8)
        # hand_ranks = hand_indices % 13  # (B, 8) -> rank indices in [0, 3]
        hand_ranks = (
            original_obs["hand"]["rank"].to(self.action_logits.device) - 1
        )  # Go back to 0 index since there are no null ranks being considered
        s1 = hand_ranks.unsqueeze(1)  # (B, 1, 8)
        s2 = hand_ranks.unsqueeze(2)  # (B, 8, 1)
        rank_matching_mask = (s1 == s2).float()

        H = hand_ranks.shape[1]  # Hand size, e.g., 8
        tri_mask = torch.triu(torch.ones(H, H, device=hand_ranks.device), diagonal=1)
        target = rank_matching_mask[:, tri_mask.bool()].float()  # (B, unordered_pairs)

        num_positive = target.sum()
        num_negative = target.numel() - num_positive
        pos_weight = num_negative / (num_positive + 1e-8)
        # pos_weight = torch.clamp(pos_weight, min=1.0, max=10.0)

        rank_matching_loss = F.binary_cross_entropy_with_logits(
            aux_logits, target, reduction="mean", pos_weight=pos_weight
        )
        self._last_rank_matching_loss = rank_matching_loss.detach().cpu().item()
        return rank_matching_loss

    def valid_card_count_aux_loss(self, policy_loss, loss_inputs):
        # Penalize the model for selecting too many or too few cards
        play_logits = self.action_logits[:, : self.hand_size]
        discard_logits = self.action_logits[:, self.hand_size : self.hand_size * 2]

        play_probs = torch.sigmoid(play_logits)
        discard_probs = torch.sigmoid(discard_logits)

        # Penalize E[play_count] over 5 or under 1 and the same for discards
        play_count = play_probs.sum(dim=1)  # shape (B,)
        discard_count = discard_probs.sum(dim=1)  # shape (B,)
        play_loss = F.relu(play_count - 5.5).mean() + F.relu(0.5 - play_count).mean()
        discard_loss = (
            F.relu(discard_count - 5.5).mean() + F.relu(0.5 - discard_count).mean()
        )
        valid_card_count_loss = play_loss + discard_loss
        self._last_valid_card_count_loss = valid_card_count_loss.detach().cpu().item()
        return valid_card_count_loss

    def weight_decay_loss(self):
        weight_decay_loss = 0.0
        for param in self.parameters():
            if param.requires_grad:
                weight_decay_loss += param.pow(2).sum()
        weight_decay_loss *= self.weight_decay_coeff
        self._last_weight_decay_loss = weight_decay_loss.detach().cpu().item()
        return weight_decay_loss

    def option_variation_loss(self):
        if self.action_method != "intent_vectors":
            return 0.0
        # gating_probs:    [B, K]  (softmax over experts)
        dist = torch.distributions.Categorical(logits=self.option_logits)
        H_cond = dist.entropy().mean()

        # marginal: you still have to do it manually,
        # but you can clamp that one:
        eps = 1e-8
        marginal = torch.softmax(self.option_logits, dim=-1).mean(dim=0).clamp(min=eps)
        H_marg = -(marginal * marginal.log()).sum()
        info_bonus = H_marg - H_cond
        info_loss = -info_bonus  # we want to minimize loss, so we negate it
        # info_loss = info_bonus

        self._last_option_variation_loss = info_loss.detach().cpu().item()

        return info_loss

        # # Convert the logits to probabilities
        # option_probs = torch.softmax(self.option_logits, dim=1)

        # # get the average probability of each option across the batch
        # avg_option_probs = option_probs.mean(dim=0)

        # # Compute the entropy of the average option probabilities
        # entropy = -torch.sum(avg_option_probs * torch.log(avg_option_probs + 1e-8))
        # self._last_option_variation_loss = entropy.detach().cpu().item()
        # return -entropy

    def intent_cosine_loss(self, play_intents, discard_intents):
        """
        play_intents: Tensor of shape (B, E, D)
        discard_intents: Tensor of shape (B, E, D)
        Returns: scalar loss = mean_pairwise_cos(play) + mean_pairwise_cos(discard)
        """

        def pairwise_cos_loss(intents):
            # 1) normalize each vector to unit length
            normed = F.normalize(intents, dim=-1)  # (B, E, D)

            # 2) compute pairwise cosine sims per batch
            sims = torch.einsum("bnd,bmd->bnm", normed, normed)  # (B, E, E)

            # 3) average over batch
            avg_sims = sims.mean(dim=0)  # (E, E)

            # 4) zero out the diagonal
            E = avg_sims.size(0)
            off_diag = avg_sims.flatten()[~torch.eye(E, dtype=bool).flatten()]

            # 5) return mean of squared cosines
            return torch.mean(off_diag**2)

        play_loss = pairwise_cos_loss(play_intents)
        discard_loss = pairwise_cos_loss(discard_intents)

        # total loss; positive when experts are similar, and minimizing
        # drives their mean cosine toward -1 (maximally dissimilar)
        return play_loss + discard_loss

    def intent_similarity_loss(self):
        if self.action_method != "intent_vectors" or self.num_experts <= 1:
            return 0.0

        # Compute the cosine similarity between play and discard intents of the pairs of experts
        play_intents = self.play_intents
        discard_intents = self.discard_intents

        intent_similarity_loss = self.intent_cosine_loss(play_intents, discard_intents)
        self._last_intent_similarity_loss = intent_similarity_loss.detach().cpu().item()
        return intent_similarity_loss

    def available_hand_types_aux_loss(
        self, policy_loss, loss_inputs, available_hand_types_logits
    ):
        d = available_hand_types_logits.device
        original_obs = restore_original_dimensions(
            loss_inputs["obs"], self.obs_space, "torch"
        )
        target_hand_types = original_obs["available_hand_types"].to(d).float()
        pos_counts = target_hand_types.sum(dim=0)
        neg_counts = target_hand_types.shape[0] - pos_counts
        pos_weight = neg_counts.clamp_min(1.0) / pos_counts.clamp_min(1.0)
        pos_weight = pos_weight.to(d)

        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # available_hand_types_loss = criterion(
        #     available_hand_types_logits,
        #     target_hand_types,
        # )
        available_hand_types_loss = F.binary_cross_entropy_with_logits(
            available_hand_types_logits,
            target_hand_types,
            pos_weight=pos_weight,
            reduction="mean",
        )
        self._last_available_hand_types_loss = (
            available_hand_types_loss.detach().cpu().item()
        )
        return available_hand_types_loss

    def hand_score_loss(self, loss_inputs):
        aux_preds = self.play_aux
        num_aux = aux_preds.size(1)
        if self.action_method == "subset_attention":
            actions = loss_inputs[SampleBatch.ACTIONS]
            is_play = actions < num_aux
        elif self.action_method == "dual_subset":
            actions = loss_inputs[SampleBatch.ACTIONS][:, 0]
            is_play = actions > 0
        true_scores = loss_inputs["joker_scores"]
        is_play_w_joker = is_play & (self._has_jokers)
        # Print the count of non-minimum entries in each column of true_scores
        min_vals = true_scores.min(dim=0, keepdim=True)[0]
        non_min_mask = true_scores > min_vals + 1e-6  # Avoid numerical issues
        non_min_counts = non_min_mask.sum(dim=0)
        nans = torch.isnan(true_scores[is_play_w_joker])
        nans_count = nans.sum(dim=0)
        # print("NaN counts per column in true_scores:", nans_count.tolist())

        is_play_w_joker = is_play_w_joker & (~torch.isnan(true_scores).any(dim=1))
        if (actions >= aux_preds.size(1)).any():
            print("Warning: Some actions are out of bounds!")
        if torch.isnan(aux_preds).any().item():
            print("Warning: aux_preds contains NaNs!")
        if is_play_w_joker.any():
            valid_actions = actions[is_play_w_joker]  # [N]
            valid_true_scores = true_scores[is_play_w_joker]  # [N, 4]
            valid_aux_preds = aux_preds[is_play_w_joker]  # [N, 1586, 4]

            # gather the predicted scores at the chosen action index for each valid row
            pred_scores = valid_aux_preds.gather(
                1,
                valid_actions.view(-1, 1, 1).expand(
                    -1, 1, aux_preds.size(2)
                ),  # [N, 1, 4]
            ).squeeze(1)
            aux_loss = F.mse_loss(pred_scores, valid_true_scores)
        else:
            aux_loss = torch.tensor(0.0, device=aux_preds.device)

        self._last_hand_score_loss = aux_loss.detach().cpu().item()
        return aux_loss

    def joker_spread_loss(self, loss_inputs):
        joker_embeddings = self.universal_card_encoder.general_index_embedding(
            torch.arange(
                BaseCard.FIRST_JOKER_INDEX,
                BaseCard.FIRST_JOKER_INDEX + 150,
                device=self.action_logits.device,
            )
        )
        diffs = joker_embeddings.unsqueeze(0) - joker_embeddings.unsqueeze(1)
        dists = torch.norm(diffs, dim=-1)
        mask = ~torch.eye(dists.size(0), dtype=torch.bool, device=dists.device)
        loss = torch.exp(-0.5 * dists[mask]).mean()
        self._last_joker_spread_loss = loss.detach().cpu().item()

        pca = PCA()
        joker_np = joker_embeddings.detach().cpu().numpy()
        pca.fit(joker_np)
        explained = pca.explained_variance_ratio_
        self._last_joker_pca_evs = explained.tolist()

        return loss

    def custom_loss(self, policy_loss, loss_inputs):
        if self.aux_outputs > 0:
            aux_logits = self._last_aux_outputs
            if self.suit_count_aux_coeff:
                suit_logits = aux_logits[:, :4]
                aux_logits = aux_logits[:, 4:]
                suit_loss = self.suit_aux_loss(policy_loss, loss_inputs, suit_logits)
                policy_loss = [x + suit_loss for x in policy_loss]

            if self.joker_identity_coeff > 0:
                joker_logits = aux_logits[:, : self.max_jokers * self.joker_types]
                aux_logits = aux_logits[:, self.max_jokers * self.joker_types :]
                joker_loss = self.joker_aux_loss(policy_loss, loss_inputs, joker_logits)
                policy_loss = [
                    x + joker_loss * self.joker_identity_coeff for x in policy_loss
                ]

            if self.suit_matching_aux_coeff > 0:
                unordered_pairs = self.hand_size * (self.hand_size - 1) // 2
                suit_matching_logits = aux_logits[:, :unordered_pairs]
                aux_logits = aux_logits[:, unordered_pairs:]
                suit_matching_loss = self.suit_matching_aux_loss(
                    policy_loss, loss_inputs, suit_matching_logits
                )
                policy_loss = [
                    x + suit_matching_loss * self.suit_matching_aux_coeff
                    for x in policy_loss
                ]

            if self.rank_matching_aux_coeff > 0:
                unordered_pairs = self.hand_size * (self.hand_size - 1) // 2
                rank_matching_logits = aux_logits[:, :unordered_pairs]
                aux_logits = aux_logits[:, unordered_pairs:]
                rank_matching_loss = self.rank_matching_aux_loss(
                    policy_loss, loss_inputs, rank_matching_logits
                )
                policy_loss = [
                    x + rank_matching_loss * self.rank_matching_aux_coeff
                    for x in policy_loss
                ]

            if self.available_hand_types_coeff > 0:
                available_hand_types_logits = aux_logits[:, :8]
                aux_logits = aux_logits[:, 8:]
                available_hand_types_loss = self.available_hand_types_aux_loss(
                    policy_loss, loss_inputs, available_hand_types_logits
                )
                policy_loss = [
                    x + available_hand_types_loss * self.available_hand_types_coeff
                    for x in policy_loss
                ]

        if self.suit_rank_entropy_coeff > 0:
            entropy_loss = self.suit_rank_entropy_loss(loss_inputs)
            policy_loss = [
                x - entropy_loss * self.suit_rank_entropy_coeff for x in policy_loss
            ]

        if self.valid_card_count_coeff > 0:
            valid_card_count_loss = self.valid_card_count_aux_loss(
                policy_loss, loss_inputs
            )
            policy_loss = [
                x + valid_card_count_loss * self.valid_card_count_coeff
                for x in policy_loss
            ]

        if self.num_experts > 1:
            option_variation_loss = self.option_variation_loss()
            policy_loss = [
                x + option_variation_loss * self.option_variation_coeff
                for x in policy_loss
            ]

        if self.intent_similarity_coeff > 0:
            intent_similarity_loss = self.intent_similarity_loss()
            policy_loss = [
                x + intent_similarity_loss * self.intent_similarity_coeff
                for x in policy_loss
            ]

        if self.weight_decay_coeff > 0:
            weight_decay_loss = self.weight_decay_loss()
            policy_loss = [x + weight_decay_loss for x in policy_loss]

        if self.hand_score_aux_coeff > 0:
            assert self.action_method in ["subset_attention", "dual_subset"]
            hand_score_loss = self.hand_score_loss(loss_inputs)
            policy_loss = [
                x + hand_score_loss * self.hand_score_aux_coeff for x in policy_loss
            ]

        if self.joker_spread_loss_coeff > 0:
            joker_spread_loss = self.joker_spread_loss(loss_inputs)
            policy_loss = [
                x + joker_spread_loss * self.joker_spread_loss_coeff
                for x in policy_loss
            ]

        self.custom_losses_have_been_calced = True
        return policy_loss

    def metrics(self):
        m = {}

        if self._last_suit_loss is not None:
            m["suit_aux_loss"] = self._last_suit_loss
        if self._last_joker_aux_loss is not None:
            m["joker_aux_loss"] = self._last_joker_aux_loss
        if self._last_suit_entropy_loss is not None:
            m["suit_entropy_loss"] = self._last_suit_entropy_loss
            m["rank_entropy_loss"] = self._last_rank_entropy_loss
        if self._last_suit_matching_loss is not None:
            m["suit_matching_loss"] = self._last_suit_matching_loss
        if self._last_rank_matching_loss is not None:
            m["rank_matching_loss"] = self._last_rank_matching_loss
        if self._last_valid_card_count_loss is not None:
            m["valid_card_count_loss"] = self._last_valid_card_count_loss
        if self._last_weight_decay_loss is not None:
            m["weight_decay_loss"] = self._last_weight_decay_loss
        if self._last_available_hand_types_loss is not None:
            m["available_hand_types_loss"] = self._last_available_hand_types_loss
        if self._last_option_variation_loss is not None:
            m["option_variation_loss"] = self._last_option_variation_loss
        if self._last_intent_similarity_loss is not None:
            m["intent_similarity_loss"] = self._last_intent_similarity_loss
        if self._last_hand_score_loss is not None:
            m["hand_score_loss"] = self._last_hand_score_loss
        if self._last_joker_spread_loss is not None:
            m["joker_spread_loss"] = self._last_joker_spread_loss
        if len(self._last_joker_pca_evs) > 0:
            for i in range(len(self._last_joker_pca_evs)):
                m[f"joker_pca_ev_top_{i}"] = sum(self._last_joker_pca_evs[: i + 1])
        if self._last_curiosity_bonus is not None:
            m["blind_curiosity_bonus"] = self._last_curiosity_bonus
        # for i, param in enumerate(self.parameters()):
        #     if param.requires_grad:
        #         m[f"param_{i}_{str(param.shape)}_norm"] = param.norm().item()

        return m
