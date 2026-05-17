import torch
import torch.nn as nn
from gym_envs.base_card import BaseCard

def init_hidden(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.network = nn.Linear(in_dim, out_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        return self.network(x)
    
def make_projector(in_dim, out_dim):
    return Projector(in_dim, out_dim)

class UniversalCardEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 128
        
        self.general_index_embedding = nn.Embedding(BaseCard.total_cards, 48, padding_idx=0)
        self.rank_embedding = nn.Embedding(BaseCard.num_ranks, 20, padding_idx=0)
        self.suit_embedding = nn.Embedding(BaseCard.num_suits, 20, padding_idx=0)
        
        for emb in [self.general_index_embedding, self.rank_embedding, self.suit_embedding]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        
        self.scalar_projector = make_projector(6, 12)
        self.relational_projector = make_projector(3, 6)
        
        self.enhancement_embedding = nn.Embedding(BaseCard.num_enhancements, 4, padding_idx=0)
        self.edition_embedding = nn.Embedding(BaseCard.num_editions, 4, padding_idx=0)
        self.seal_embedding = nn.Embedding(BaseCard.num_seals, 4, padding_idx=0)
        self.rarity_embedding = nn.Embedding(BaseCard.num_rarities, 4, padding_idx=0)
        self.segment_embedding = nn.Embedding(BaseCard.num_segments, 4, padding_idx=0)

    def forward(self, cards_obs):
        device = cards_obs["indices"].device
        B, L = cards_obs["indices"].shape
        
        seg_raw = cards_obs["segment"].long()
        valid_mask = (cards_obs["segment"] != 0).float().unsqueeze(-1)
        playing_card_mask = ((seg_raw == 1) | (seg_raw == 2)).float().unsqueeze(-1)

        idx_emb = self.general_index_embedding(cards_obs["indices"].long())

        rank_emb = self.rank_embedding(cards_obs["rank"].long())
        suit_emb = self.suit_embedding(cards_obs["suit"].long())
        rank_suit_emb = torch.cat([rank_emb, suit_emb], dim=-1)
        rank_suit_emb = rank_suit_emb * playing_card_mask * valid_mask

        ranks = cards_obs["rank"]
        suits = cards_obs["suit"]
        
        same_rank_count = (ranks.unsqueeze(-1) == ranks.unsqueeze(-2)).sum(dim=-1).float()
        same_rank_feat = (same_rank_count / 10.0).unsqueeze(-1)
        
        same_suit_count = (suits.unsqueeze(-1) == suits.unsqueeze(-2)).sum(dim=-1).float()
        same_suit_feat = (same_suit_count / 10.0).unsqueeze(-1)
        
        is_face_card = ((ranks >= 10) & (ranks <= 12)).float().unsqueeze(-1)
        
        rel_combined = torch.cat([same_rank_feat, same_suit_feat, is_face_card], dim=-1)
        rel_emb = self.relational_projector(rel_combined)
        rel_emb = rel_emb * playing_card_mask * valid_mask

        raw_scalars = cards_obs["scalar_properties"].float()
        is_debuffed = cards_obs["debuffed"].float().unsqueeze(-1)
        raw_scalars = raw_scalars * (1.0 - is_debuffed)
        
        c_cost, c_chips, c_mult, c_xmult = raw_scalars.unbind(dim=-1)
        
        log_chips = torch.log10(torch.abs(c_chips) + 1.0) / 3.0
        log_mult  = torch.log10(torch.abs(c_mult)  + 1.0) / 2.0
        
        scalars_vec = torch.stack([
            c_cost / 100.0,
            c_chips / 1000.0,
            log_chips,
            c_mult / 100.0,
            log_mult,
            c_xmult / 5.0
        ], dim=-1)
        
        scalar_emb = self.scalar_projector(scalars_vec)
        scalar_emb = scalar_emb * valid_mask

        enh_emb = self.enhancement_embedding(cards_obs["enhancement"].long()) 
        edi_emb = self.edition_embedding(cards_obs["edition"].long())         
        seal_emb = self.seal_embedding(cards_obs["seal"].long())              
        rarity_emb = self.rarity_embedding(cards_obs["rarity"].long())        
        seg_emb = self.segment_embedding(seg_raw)                             
        
        cat_emb = torch.cat([enh_emb, edi_emb, seal_emb, rarity_emb, seg_emb], dim=-1)
        cat_emb = cat_emb * valid_mask

        core_embeddings = torch.cat([
            idx_emb, rank_suit_emb, rel_emb, scalar_emb, cat_emb
        ], dim=-1)
        
        logical_flags = torch.zeros(B, L, 2, device=device)
        full_embeddings = torch.cat([core_embeddings, logical_flags], dim=-1)
        
        return full_embeddings