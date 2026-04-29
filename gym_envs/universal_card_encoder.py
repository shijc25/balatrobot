import torch
import torch.nn as nn
from gym_envs.base_card import BaseCard
import math

class UniversalCardEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_size = 63 
        self.segment_size = 2
        self.edition_size = 2
        self.seal_size = 2
        self.enhancement_size = 2
        
        self.main_embedding_size = 49
        self.suit_embedding_size = 23

        self.general_index_embedding = nn.Embedding(BaseCard.total_cards, self.main_embedding_size, padding_idx=0)
        self.enhancement_embedding = nn.Embedding(BaseCard.num_enhancements, self.enhancement_size, padding_idx=0)
        self.edition_embedding = nn.Embedding(BaseCard.num_editions, self.edition_size, padding_idx=0)
        self.seal_embedding = nn.Embedding(BaseCard.num_seals, self.seal_size, padding_idx=0)
        self.segment_embedding = nn.Embedding(BaseCard.num_segments, self.segment_size, padding_idx=0)

        self.suit_embedding = nn.Embedding(BaseCard.num_suits, self.suit_embedding_size - 5, padding_idx=0)
        self.rank_embedding = nn.Embedding(BaseCard.num_ranks, self.main_embedding_size - self.suit_embedding_size - 2, padding_idx=0)

    def forward(self, cards_obs):
        B, L = cards_obs["indices"].shape
        device = cards_obs["indices"].device

        idx_emb = self.general_index_embedding(cards_obs["indices"].long())
        enh_emb = self.enhancement_embedding(cards_obs["enhancement"].long())
        edi_emb = self.edition_embedding(cards_obs["edition"].long())
        seal_emb = self.seal_embedding(cards_obs["seal"].long())
        seg_emb = self.segment_embedding(cards_obs["segment"].long())
        
        suit_emb = self.suit_embedding(cards_obs["suit"].long())
        rank_emb = self.rank_embedding(cards_obs["rank"].long())
        
        ranks = cards_obs["rank"]
        suits = cards_obs["suit"]
        mask = (cards_obs["indices"] == 0)

        same_rank_count = (ranks.unsqueeze(-1) == ranks.unsqueeze(-2)).sum(dim=-1).float()
        same_rank_feat = torch.where(mask, 0.0, same_rank_count / 5.0).unsqueeze(-1)
        
        is_face_card = ((ranks >= 11) & (ranks <= 13)).float().unsqueeze(-1)

        suit_counts = torch.stack([
            (suits == s).sum(dim=1) for s in range(1, 5)
        ], dim=1).float()
        
        suit_ratios = (suit_counts / float(L)).unsqueeze(1).expand(-1, L, -1)
        suit_ratios = torch.where(mask.unsqueeze(-1), 0.0, suit_ratios)

        card_suit_total = torch.gather(suit_counts, 1, (suits.long() - 1).clamp(0, 3))
        flush_potential = (card_suit_total / 5.0).float().unsqueeze(-1)
        flush_potential = torch.where(mask.unsqueeze(-1), 0.0, flush_potential)

        suit_part = torch.cat([suit_emb, suit_ratios, flush_potential], dim=-1)
        rank_part = torch.cat([rank_emb, same_rank_feat, is_face_card], dim=-1)
        
        suit_rank_emb = torch.cat([suit_part, rank_part], dim=-1)

        raw_scalars = cards_obs["scalar_properties"].float()
        is_debuffed = cards_obs["debuffed"].float().unsqueeze(-1)
        raw_scalars = raw_scalars * (1.0 - is_debuffed)

        c_cost  = raw_scalars[:, :, 0]
        c_chips = raw_scalars[:, :, 1]
        c_mult  = raw_scalars[:, :, 2]
        c_xmult = raw_scalars[:, :, 3]

        log_chips = torch.log10(c_chips + 1.0) / 3.0
        log_mult  = torch.log10(c_mult + 1.0) / 2.0
        
        scalars = torch.stack([
            c_cost / 100.0,
            c_chips / 1000.0,
            log_chips,
            c_mult / 100.0,
            log_mult,
            c_xmult / 5.0
        ], dim=-1)

        embeddings = torch.cat([
            idx_emb + suit_rank_emb, # 49
            scalars,                # 6
            seg_emb,                # 2
            enh_emb,                # 2
            edi_emb,                # 2
            seal_emb                # 2
        ], dim=-1)

        return embeddings, mask