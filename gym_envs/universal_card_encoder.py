import torch
import torch.nn as nn
from gym_envs.base_card import BaseCard
import math

class UniversalCardEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_size = 63 
        self.segment_size = 4
        self.edition_size = 4
        self.seal_size = 4
        self.enhancement_size = 4
        
        self.main_embedding_size = 43

        self.general_index_embedding = nn.Embedding(BaseCard.total_cards, self.main_embedding_size, padding_idx=0)
        self.enhancement_embedding = nn.Embedding(BaseCard.num_enhancements, self.enhancement_size, padding_idx=0)
        self.edition_embedding = nn.Embedding(BaseCard.num_editions, self.edition_size, padding_idx=0)
        self.seal_embedding = nn.Embedding(BaseCard.num_seals, self.seal_size, padding_idx=0)
        self.segment_embedding = nn.Embedding(BaseCard.num_segments, self.segment_size, padding_idx=0)

        self.suit_embedding = nn.Embedding(BaseCard.num_suits, BaseCard.num_suits, padding_idx=0)
        self.rank_embedding = nn.Embedding(BaseCard.num_ranks, self.main_embedding_size - BaseCard.num_suits, padding_idx=0)

        self.suit_embedding.weight.requires_grad = False
        self.rank_embedding.weight.requires_grad = False
        with torch.no_grad():
            self.suit_embedding.weight.zero_()
            self.rank_embedding.weight.zero_()
            for i in range(BaseCard.num_suits): self.suit_embedding.weight[i, i] = 1.0
            for i in range(BaseCard.num_ranks): self.rank_embedding.weight[i, i] = 1.0

    def forward(self, cards_obs):
        idx_emb = self.general_index_embedding(cards_obs["indices"].long())
        enh_emb = self.enhancement_embedding(cards_obs["enhancement"].long())
        edi_emb = self.edition_embedding(cards_obs["edition"].long())
        seal_emb = self.seal_embedding(cards_obs["seal"].long())
        seg_emb = self.segment_embedding(cards_obs["segment"].long())
        
        suit_emb = self.suit_embedding(cards_obs["suit"].long())
        rank_emb = self.rank_embedding(cards_obs["rank"].long())
        
        ranks = cards_obs["rank"]
        suits = cards_obs["suit"]
        
        same_rank = (ranks.unsqueeze(-1) == ranks.unsqueeze(-2)).sum(dim=-1)
        same_suit = (suits.unsqueeze(-1) == suits.unsqueeze(-2)).sum(dim=-1)
        
        rank_up = (ranks.unsqueeze(-1) == (ranks.unsqueeze(-2) + 1)).sum(dim=-1)
        rank_down = (ranks.unsqueeze(-1) == (ranks.unsqueeze(-2) - 1)).sum(dim=-1)
        
        rank_sin = torch.sin(ranks.float() * (2 * math.pi / 13))
        rank_cos = torch.cos(ranks.float() * (2 * math.pi / 13))

        mask = (ranks == 0)
        same_rank = torch.where(mask, 0.0, same_rank.float() / 5.0)
        same_suit = torch.where(mask, 0.0, same_suit.float() / 5.0)
        rank_up = torch.where(mask, 0.0, rank_up.float())
        rank_down = torch.where(mask, 0.0, rank_down.float())

        rank_emb[:, :, -1] = same_rank
        rank_emb[:, :, -2] = same_suit
        rank_emb[:, :, -3] = rank_up
        rank_emb[:, :, -4] = rank_down
        rank_emb[:, :, -5] = rank_sin
        rank_emb[:, :, -6] = rank_cos
        
        suit_rank_emb = torch.cat((suit_emb, rank_emb), dim=-1)

        scalars = cards_obs["scalar_properties"].float()
        
        is_debuffed = cards_obs["debuffed"].float().unsqueeze(-1)
        scalars = scalars * (1.0 - is_debuffed)

        scalars[:, :, 0] /= 10.0   # Cost
        scalars[:, :, 1] /= 100.0  # Chips
        scalars[:, :, 2] /= 100.0  # Mult
        scalars[:, :, 3] /= 10.0   # xMult

        embeddings = torch.cat([
            idx_emb + suit_rank_emb, scalars, seg_emb, enh_emb, edi_emb, seal_emb
        ], dim=-1)

        return embeddings, (cards_obs["indices"] == 0) & (mask)