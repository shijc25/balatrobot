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
        
        self.skip_token = nn.Parameter(torch.randn(1, 1, 63))
        self.money_proj = nn.Linear(1, 63); self.reroll_proj = nn.Linear(1, 63)
        self.goal_proj = nn.Linear(1, 63); self.levels_proj = nn.Linear(4, 63)
        
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_tokens, 63) * 0.02)
        self.blind_emb = nn.Embedding(32, 63)
        
        self.shelf_bias = nn.Parameter(torch.randn(1, 1, 63) * 0.02)
        self.owned_bias = nn.Parameter(torch.randn(1, 1, 63) * 0.02)
        self.pack_bias = nn.Parameter(torch.randn(1, 1, 63) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=4, dim_feedforward=self.h, activation="gelu", batch_first=True,
            dropout=0.0, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        def make_head(out_dim):
            return nn.Sequential(
                nn.Linear(self.dim, self.h), nn.LayerNorm(self.h), nn.GELU(),
                nn.Linear(self.h, out_dim)
            )
        
        self.action_head = make_head(1)
        self.planet_head = make_head(1)
        self.value_head = make_head(1)
        
        self._val_out = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        B = obs["dollars"].shape[0]
        device = obs["dollars"].device
        
        sk_tok = self.skip_token.expand(B, -1, -1)
        m_tok = self.money_proj(obs["dollars"] / 10.0).unsqueeze(1)
        r_tok = self.reroll_proj(obs["reroll_price"] / 10.0).unsqueeze(1)
        blind_id = obs["blind_index"].long().view(-1)
        g_tok = (self.goal_proj(obs["goal"] / 10000.0) + self.blind_emb(blind_id)).unsqueeze(1)
        lvls = torch.stack([obs["hand_stats"]["level"], obs["hand_stats"]["played_count"], obs["hand_stats"]["chips"] / 100.0, obs["hand_stats"]["mult"] / 10.0], dim=-1)
        l_toks = self.levels_proj(lvls)
        s_toks, _ = self.card_encoder(obs["shop_cards"])
        b_toks, _ = self.card_encoder(obs["booster_cards"])
        o_toks, _ = self.card_encoder(obs["owned_jokers"])
        p_toks, _ = self.card_encoder(obs["pack_cards"])
        
        s_toks = s_toks + self.shelf_bias
        b_toks = b_toks + self.shelf_bias
        o_toks = o_toks + self.owned_bias
        p_toks = p_toks + self.pack_bias
        
        seq_63 = torch.cat([sk_tok, m_tok, r_tok, g_tok, l_toks, s_toks, b_toks, o_toks, p_toks], dim=1)
        seq_64 = torch.cat([seq_63 + self.pos_emb, torch.zeros(B, self.num_tokens, 1, device=device)], dim=2)
        
        features = self.transformer(seq_64)
        
        goal_feat = features[:, 3, :]
        self._val_out = self.value_head(goal_feat).squeeze(-1)
        
        act_indices = [0, 2, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        all_item_logits = self.action_head(features[:, act_indices, :]).squeeze(-1)

        planet_scores = torch.full((B, 5), -1e9, device=device)
        pack_indices = obs["pack_card_level_indices"]
        batch_range = torch.arange(B, device=device)
        for i in range(5):
            l_idx = pack_indices[:, i].long()
            valid = l_idx >= 0
            if valid.any():
                safe_l_idx = torch.clamp(l_idx, 0, 11)
                target_lv_feat = features[batch_range[valid], 4 + safe_l_idx[valid], :]
                planet_scores[valid, i] = self.planet_head(target_lv_feat).squeeze(-1)

        is_celestial = obs["current_pack_is_celestial"].squeeze(-1).bool()
        
        l0 = all_item_logits[:, 0:1]
        l1_10 = all_item_logits[:, 1:11]
        l11_15 = torch.where(is_celestial.unsqueeze(-1), planet_scores, all_item_logits[:, 11:16])
        l16 = torch.full((B, 1), -1e8, device=device)

        total_logits = torch.cat([l0, l1_10, l11_15, l16], dim=1)

        money = obs["dollars"].view(B)
        j_count = obs["owned_joker_count"].view(B)
        in_pack = obs["in_pack_selection"].view(B).bool()

        shop_m = []
        shop_m.append(torch.zeros(B, device=device, dtype=torch.bool))
        shop_m.append(money < obs["reroll_price"].view(B))
        for i in range(2):
            invalid = ~(obs["is_joker_on_shelf"][:, i].bool() | obs["is_planet_on_shelf"][:, i].bool())
            full = obs["is_joker_on_shelf"][:, i].bool() & (j_count >= 5)
            shop_m.append(invalid | full | (money < obs["shop_prices"][:, i]))
        for i in range(2):
            invalid = ~(obs["is_buffoon_pack"][:, i].bool() | obs["is_celestial_pack"][:, i].bool())
            full = obs["is_buffoon_pack"][:, i].bool() & (j_count >= 5)
            shop_m.append(invalid | full | (money < obs["booster_prices"][:, i]))
        for i in range(5):
            shop_m.append(obs["owned_jokers"]["indices"][:, i] == 0)
        shop_mask_tensor = torch.stack(shop_m, dim=1)

        pack_m = []
        for i in range(5):
            is_j = obs["pack_card_is_joker"][:, i].bool()
            is_p = obs["pack_card_is_planet"][:, i].bool()
            j_legal = is_j & (j_count < 5)
            valid = torch.where(is_celestial, is_p, is_j & j_legal)
            pack_m.append(~valid)
        pack_mask_tensor = torch.stack(pack_m, dim=1)

        masked_shop = torch.where(shop_mask_tensor, torch.tensor(-1e9, device=device), total_logits[:, :11])
        masked_pack = torch.where(pack_mask_tensor, torch.tensor(-1e9, device=device), total_logits[:, 11:16])
        
        final_shop_part = torch.where(in_pack.unsqueeze(-1), torch.tensor(-1e9, device=device), masked_shop)
        
        skip_pack_val = torch.where(is_celestial, torch.tensor(-1e9, device=device), total_logits[:, 16])
        final_pack_part = torch.where(~in_pack.unsqueeze(-1), torch.tensor(-1e9, device=device), torch.cat([masked_pack, skip_pack_val.unsqueeze(1)], dim=1))

        res_logits = torch.cat([final_shop_part, final_pack_part], dim=1)

        return res_logits, []

    def value_function(self):
        return self._val_out