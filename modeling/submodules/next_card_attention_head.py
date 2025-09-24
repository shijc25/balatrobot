from copy import deepcopy
from gymnasium import spaces as sp
import numpy as np
from random import randint, choice, random
import torch
import torch.nn as nn
from gym_envs.base_card import BaseCard
import math

from gym_envs.universal_card_encoder import UniversalCardEncoder
from modeling.submodules.card_self_attention import CardSelfAttention
import torch.nn.functional as F
from modeling.bitwise_hand_types import BitwiseHandTypes


class NextCardAttentionHead(nn.Module):
    def __init__(self, card_embedding_size, num_attention_layers=2, card_encoder=None):
        super().__init__()
        self.card_embedding_size = card_embedding_size
        self.num_attention_layers = num_attention_layers
        self.self_attention_layers = nn.ModuleList(
            [
                CardSelfAttention(
                    d_model=card_embedding_size, n_heads=4, dropout=0.00, noisy=False
                )
                for _ in range(num_attention_layers)
            ]
        )
        self.action_method = "per_card_head"
        # self.reachable_hands_mode = "mlp_inputs"
        self.reachable_hands_mode = None

        # self.already_selected_tokens = nn.Parameter(torch.randn(2, card_embedding_size))
        self.already_selected_tokens = nn.Embedding(
            2, card_embedding_size, padding_idx=0
        )

        if self.action_method == "per_card_head":
            pch_inputs = card_embedding_size * 3
            if self.reachable_hands_mode == "mlp_inputs":
                pch_inputs += 8
            self.per_card_head = nn.Sequential(
                nn.Linear(pch_inputs, int(pch_inputs / 2)),
                nn.GELU(),
                nn.Linear(int(pch_inputs / 2), 1),  # One logit per card
            )
        self.tau = nn.Parameter(torch.tensor(10.0))

        stop_card = BaseCard(
            segment=BaseCard.Segments.SPECIAL_TOKEN,
            u_index=BaseCard.SpecialTokens.STOP_CONTEXT
            + BaseCard.FIRST_SPECIAL_TOKEN_INDEX,
        )
        play_card = BaseCard(
            segment=BaseCard.Segments.SPECIAL_TOKEN,
            u_index=BaseCard.SpecialTokens.PLAY_CONTEXT
            + BaseCard.FIRST_SPECIAL_TOKEN_INDEX,
        )
        discard_card = BaseCard(
            segment=BaseCard.Segments.SPECIAL_TOKEN,
            u_index=BaseCard.SpecialTokens.DISCARD_CONTEXT
            + BaseCard.FIRST_SPECIAL_TOKEN_INDEX,
        )
        action_cards = [stop_card, play_card, discard_card]
        self.action_cards_obs = BaseCard.observe_list(action_cards, 3)

        if card_encoder is None:
            card_encoder = UniversalCardEncoder(embedding_size=card_embedding_size)
        self.card_encoder = card_encoder

        self.cached_action_embeddings = None

    def invalidate_cache(self):
        self.cached_action_embeddings = None

    def build_action_cards_embeddings(self, B, device):
        if self.cached_action_embeddings is not None:
            action_embeddings, action_padding = self.cached_action_embeddings
            action_embeddings = action_embeddings.repeat(B, 1, 1)
            action_padding = action_padding.repeat(B, 1)
            return action_embeddings, action_padding
        action_cards_obs = {
            k: torch.tensor(v, device=device).unsqueeze(0)
            for k, v in self.action_cards_obs.items()
        }
        action_embeddings, action_padding = self.card_encoder(action_cards_obs)

        self.cached_action_embeddings = (action_embeddings, action_padding)
        action_embeddings = action_embeddings.repeat(B, 1, 1)
        action_padding = action_padding.repeat(B, 1)
        return action_embeddings, action_padding

    def forward(
        self,
        mode,
        hand_embeddings,
        hand_padding,
        already_selected_cards_mask,
        h_tokens,
        card_obs,
    ):
        B = hand_embeddings.shape[0]
        device = hand_embeddings.device

        action_embeddings, action_padding = self.build_action_cards_embeddings(
            B, device
        )

        already_selected_cards_embeddings = self.already_selected_tokens(
            already_selected_cards_mask.long()
        )

        # ignore the last token (stop) for now
        hand_embeddings = hand_embeddings + already_selected_cards_embeddings[:, :-1, :]

        original_x = torch.cat(
            [hand_embeddings, action_embeddings, h_tokens],
            dim=1,
        )
        x = original_x

        padding = torch.cat(
            [
                hand_padding,
                action_padding,
                torch.zeros(
                    (B, h_tokens.shape[1]), dtype=torch.bool, device=device
                ),  # padding for h_tokens
            ],
            dim=1,
        )

        for layer in self.self_attention_layers:
            x = layer(x, padding=padding)

        x = F.layer_norm(x, (x.size(-1),))

        play_intent = x[:, -3, :]
        discard_intent = x[:, -2, :]

        intent = torch.where(mode.bool().unsqueeze(1), play_intent, discard_intent)
        hand_w_stop = original_x[:, :-3, :]

        if self.action_method == "intent_attention":
            hand = F.normalize(hand_w_stop, dim=-1)
            intent = F.normalize(intent, dim=-1)
            logits = self.tau * (hand @ intent.unsqueeze(-1)).squeeze(-1)
            # logits[:, -1] = logits[:, -1] - 1
        elif self.action_method == "per_card_head":
            if self.reachable_hands_mode == "mlp_inputs":
                reachable_hands = BitwiseHandTypes.possible_hand_types(
                    card_obs,
                    illegal_mask=hand_padding,
                    must_include_mask=already_selected_cards_mask[:, :-1].long(),
                )
            pch_input = torch.cat(
                [
                    hand_w_stop,
                    intent.unsqueeze(1).repeat(1, hand_w_stop.shape[1], 1),
                    h_tokens.repeat(1, hand_w_stop.shape[1], 1),
                    (
                        reachable_hands.unsqueeze(1).repeat(1, hand_w_stop.shape[1], 1)
                        if self.reachable_hands_mode == "mlp_inputs"
                        else torch.empty(0, device=hand_w_stop.device)
                    ),
                ],
                dim=-1,
            )
            logits = self.per_card_head(pch_input).squeeze(-1)
        logits = logits.masked_fill(padding[:, :-3], -1e9)

        return logits
