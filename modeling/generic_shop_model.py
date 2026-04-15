import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class BalatroShopModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.dim = 64
        self.h = 128
        self.num_tokens = 30
        from gym_envs.universal_card_encoder import UniversalCardEncoder
        self.card_encoder = UniversalCardEncoder()

        self.action_token = nn.Parameter(torch.randn(1, 1, 63))
        self.money_proj = nn.Linear(1, 63); self.reroll_proj = nn.Linear(1, 63)
        self.goal_proj = nn.Linear(1, 63); self.levels_proj = nn.Linear(4, 63)
        
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_tokens, 63) * 0.02)
        self.blind_emb = nn.Embedding(32, 63)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=8, dim_feedforward=self.h, activation="gelu", batch_first=True,
            dropout=0.0, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.action_head = nn.Sequential(
            nn.Linear(self.dim, self.h), nn.LayerNorm(self.h), nn.GELU(),
            nn.Linear(self.h, self.h), nn.LayerNorm(self.h), nn.GELU(),
            nn.Linear(self.h, 17)
        )
        
        self.planet_head = nn.Sequential(nn.Linear(self.dim, self.h), nn.LayerNorm(self.h), nn.GELU(), nn.Linear(self.h, 1))
        
        self.value_head = nn.Sequential(nn.Linear(self.dim, self.h), nn.LayerNorm(self.h), nn.GELU(), nn.Linear(self.h, 1))

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        B = obs["dollars"].shape[0]
        device = obs["dollars"].device

        m_tok = self.money_proj(obs["dollars"] / 10.0).unsqueeze(1)
        r_tok = self.reroll_proj(obs["reroll_price"] / 10.0).unsqueeze(1)
        
        blind_id = obs["blind_index"].long().view(-1)
        blind_identity = self.blind_emb(blind_id)

        g_tok = (self.goal_proj(obs["goal"] / 10000.0) + blind_identity).unsqueeze(1)
        
        lvls = torch.stack([
            obs["hand_stats"]["level"], obs["hand_stats"]["played_count"], 
            obs["hand_stats"]["chips"] / 100.0, obs["hand_stats"]["mult"] / 10.0
        ], dim=-1)
        l_toks = self.levels_proj(lvls)
        
        s_toks, _ = self.card_encoder(obs["shop_cards"])
        b_toks, _ = self.card_encoder(obs["booster_cards"])
        o_toks, _ = self.card_encoder(obs["owned_jokers"])
        p_toks, _ = self.card_encoder(obs["pack_cards"])

        seq_63 = torch.cat([self.action_token.expand(B, -1, -1), m_tok, r_tok, g_tok, l_toks, s_toks, b_toks, o_toks, p_toks], dim=1)
        seq_64 = torch.cat([seq_63 + self.pos_emb, torch.zeros(B, self.num_tokens, 1, device=device)], dim=2)

        features = self.transformer(seq_64)
        
        decide_feat = features[:, 0, :]
        
        raw_logits = self.action_head(decide_feat)
        self._val_out = self.value_head(decide_feat).squeeze(-1)
        
        planet_pack_logits = torch.full((B, 5), -1e9, device=device)
        pack_indices = obs["pack_card_level_indices"]
        
        for i in range(5):
            l_idx = pack_indices[:, i]
            valid = l_idx >= 0
            if valid.any():
                v_idx = l_idx[valid].long()
                target_feats = features[valid, 4 + v_idx, :]
                planet_pack_logits[valid, i] = self.planet_head(target_feats).squeeze(-1)
        
        is_buffoon = obs["current_pack_is_buffoon"].squeeze(-1).bool()
        is_celestial = obs["current_pack_is_celestial"].squeeze(-1).bool()
        in_pack = obs["in_pack_selection"].squeeze(-1).bool() 

        raw_logits[:, 11:16] = torch.where(
            is_celestial.unsqueeze(-1), 
            planet_pack_logits, 
            raw_logits[:, 11:16]
        )
        
        safe_logits = raw_logits.clone()
        money = obs["dollars"].squeeze(-1)
        j_count = obs["owned_joker_count"].squeeze(-1)

        safe_logits[:, :11] = torch.where(in_pack.unsqueeze(-1), torch.tensor(-1e9, device=device), safe_logits[:, :11])

        safe_logits[:, 11:17] = torch.where(~in_pack.unsqueeze(-1), torch.tensor(-1e9, device=device), safe_logits[:, 11:17])

        safe_logits[:, 1] = torch.where(money < obs["reroll_price"].squeeze(-1), -1e9, safe_logits[:, 1])
        
        for i in range(2):
            price = obs["shop_prices"][:, i]
            is_valid_card = obs["is_joker_on_shelf"][:, i].bool() | obs["is_planet_on_shelf"][:, i].bool()
            joker_full = obs["is_joker_on_shelf"][:, i].bool() & (j_count >= 5)
            mask_cond = (~is_valid_card) | joker_full | (money < price)
            safe_logits[:, 2+i] = torch.where(mask_cond, -1e9, safe_logits[:, 2+i])

        for i in range(2):
            price = obs["booster_prices"][:, i]
            is_valid_pack = obs["is_buffoon_pack"][:, i].bool() | obs["is_celestial_pack"][:, i].bool()
            buffoon_full = obs["is_buffoon_pack"][:, i].bool() & (j_count >= 5)
            mask_cond = (~is_valid_pack) | buffoon_full | (money < price)
            safe_logits[:, 4+i] = torch.where(mask_cond, -1e9, safe_logits[:, 4+i])
            
        for i in range(5):
            is_empty = (obs["owned_jokers"]["indices"][:, i] == 0)
            safe_logits[:, 6+i] = torch.where(is_empty, -1e9, safe_logits[:, 6+i])

        for i in range(5):
            is_joker = obs["pack_card_is_joker"][:, i].bool()
            is_planet = obs["pack_card_is_planet"][:, i].bool()
            joker_legal = is_joker & (j_count < 5)
            is_valid_choice = (is_buffoon & joker_legal) | (is_celestial & is_planet)
            safe_logits[:, 11+i] = torch.where(~is_valid_choice, -1e9, safe_logits[:, 11+i])

        safe_logits[:, 16] = torch.where(is_celestial, -1e9, safe_logits[:, 16])

        return safe_logits, []
    
    def value_function(self):
        return self._val_out