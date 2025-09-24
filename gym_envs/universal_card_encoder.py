from copy import deepcopy
from gymnasium import spaces as sp
import numpy as np
from random import randint, choice, random
import torch
import torch.nn as nn
from gym_envs.base_card import BaseCard
import math


class UniversalCardEncoder(nn.Module):
    def __init__(self, embedding_size=64):
        super().__init__()
        self.embedding_size = embedding_size
        self.segment_size = 4
        self.edition_size = 4
        self.seal_size = 4
        self.enhancement_size = 4
        self.relational_feature_size = 7
        self.fixed_rank_suit_embeddings = True
        self.scalar_property_size = BaseCard.num_scalar_properties
        self.main_embedding_size = (
            embedding_size
            - self.segment_size
            - self.scalar_property_size
            - self.edition_size
            - self.seal_size
            - self.enhancement_size
        )
        # self.max_norm = math.sqrt(self.main_embedding_size)
        self.max_norm = 999.0

        self.general_index_embedding = nn.Embedding(
            BaseCard.total_cards,
            self.main_embedding_size,
            padding_idx=0,
            max_norm=self.max_norm / 2,
        )

        self.enhancement_embedding = nn.Embedding(
            BaseCard.num_enhancements,
            self.enhancement_size,
            padding_idx=0,
            max_norm=self.max_norm / 2,
        )

        self.edition_embedding = nn.Embedding(
            BaseCard.num_editions,
            self.edition_size,
            padding_idx=0,
            max_norm=self.max_norm / 2,
        )

        self.seal_embedding = nn.Embedding(
            BaseCard.num_seals,
            self.seal_size,
            padding_idx=0,
            max_norm=self.max_norm / 2,
        )

        self.segment_embedding = nn.Embedding(
            BaseCard.num_segments,
            self.segment_size,
            padding_idx=0,
            max_norm=self.max_norm / 2,
        )

        self.suit_embedding = nn.Embedding(
            BaseCard.num_suits,
            (
                BaseCard.num_suits
                if self.fixed_rank_suit_embeddings
                else int(self.main_embedding_size / 4)
            ),
            padding_idx=0,
            max_norm=self.max_norm / 2,
        )

        self.rank_embedding = nn.Embedding(
            BaseCard.num_ranks,
            self.main_embedding_size - self.suit_embedding.embedding_dim,
            padding_idx=0,
            max_norm=self.max_norm / 2,
        )

        if self.fixed_rank_suit_embeddings:
            # If we use fixed rank and suit embeddings, we need to make sure they are not trainable
            self.suit_embedding.weight.requires_grad = False
            self.rank_embedding.weight.requires_grad = False

            # Ensure embedding sizes are large enough for one-hot
            assert (
                self.suit_embedding.embedding_dim >= BaseCard.num_suits
            ), "suit_embedding too small for one-hot"
            assert (
                self.rank_embedding.embedding_dim
                >= BaseCard.num_ranks + self.relational_feature_size
            ), "rank_embedding too small for one-hot"

            with torch.no_grad():
                self.suit_embedding.weight.zero_()
                self.rank_embedding.weight.zero_()
                for i in range(BaseCard.num_suits):
                    self.suit_embedding.weight[i, i] = 1.0
                for i in range(BaseCard.num_ranks):
                    self.rank_embedding.weight[i, i] = 1.0

        self.debuffed_embedding = nn.Embedding(
            2,
            self.main_embedding_size,
            padding_idx=0,
            max_norm=self.max_norm / 2,
        )

        # Value, chips, mult, mult_mult
        self.num_scalar_properties = 4
        self.scalar_property_projectory = nn.Linear(
            self.num_scalar_properties,
            self.main_embedding_size,
            bias=False,  # no bias so that there is no noise added when the values are 0
        )

    def forward(self, cards_obs):
        index_embeddings = self.general_index_embedding(cards_obs["indices"].long())
        enhancement_embeddings = self.enhancement_embedding(
            cards_obs["enhancement"].long()
        )
        edition_embeddings = self.edition_embedding(cards_obs["edition"].long())
        seal_embeddings = self.seal_embedding(cards_obs["seal"].long())
        debuffed_embeddings = self.debuffed_embedding(cards_obs["debuffed"].long())
        segment_embeddings = self.segment_embedding(cards_obs["segment"].long())
        suit_embeddings = self.suit_embedding(cards_obs["suit"].long())
        rank_embeddings = self.rank_embedding(cards_obs["rank"].long())
        if self.fixed_rank_suit_embeddings:
            # If we use fixed rank and suit embeddings, we also need to add the relational features
            same_rank_counts = (
                cards_obs["rank"].unsqueeze(-1) == cards_obs["rank"].unsqueeze(-2)
            ).sum(dim=-1, keepdim=False)
            same_suit_counts = (
                cards_obs["suit"].unsqueeze(-1) == cards_obs["suit"].unsqueeze(-2)
            ).sum(dim=-1, keepdim=False)

            # We need to zero out the relations when rank/suit is 0, which is the case for non-playing cards
            same_rank_counts = torch.where(
                cards_obs["rank"] == 0,
                torch.zeros_like(same_rank_counts),
                same_rank_counts,
            )
            same_suit_counts = torch.where(
                cards_obs["suit"] == 0,
                torch.zeros_like(same_suit_counts),
                same_suit_counts,
            )
            in_flush = same_suit_counts >= 5
            rank_up_counts = (
                cards_obs["rank"].unsqueeze(-1) == cards_obs["rank"].unsqueeze(-2) + 1
            ).sum(dim=-1, keepdim=False)
            rank_down_counts = (
                cards_obs["rank"].unsqueeze(-1) == cards_obs["rank"].unsqueeze(-2) - 1
            ).sum(dim=-1, keepdim=False)
            rank_sin = torch.sin((cards_obs["rank"] + 1) * math.pi / BaseCard.num_ranks)
            rank_cos = torch.cos((cards_obs["rank"] + 1) * math.pi / BaseCard.num_ranks)

            # We made sure there was room for these features in the rank embedding, so we'll just reverse index
            rank_embeddings[:, :, -1] = same_rank_counts.float()
            rank_embeddings[:, :, -2] = same_suit_counts.float()
            rank_embeddings[:, :, -3] = in_flush.float()
            rank_embeddings[:, :, -4] = rank_down_counts.float()
            rank_embeddings[:, :, -5] = rank_up_counts.float()
            rank_embeddings[:, :, -6] = rank_sin.float()
            rank_embeddings[:, :, -7] = rank_cos.float()

        suit_rank_embeddings = torch.cat((suit_embeddings, rank_embeddings), dim=-1)
        # scalar_properties = self.scalar_property_projectory(
        #     cards_obs["scalar_properties"]
        # )
        scalar_properties = cards_obs["scalar_properties"].float()
        # Regular cards have 0 index, but they use the rank+suit embeddings instead
        padding_mask = (cards_obs["indices"] == 0) & (cards_obs["rank"] == 0)

        embeddings = torch.concat(
            [
                index_embeddings + suit_rank_embeddings,
                scalar_properties,
                segment_embeddings,
                enhancement_embeddings,
                edition_embeddings,
                seal_embeddings,
            ],
            dim=-1,
        )

        return embeddings, padding_mask
