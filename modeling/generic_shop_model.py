import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from gym_envs.universal_card_encoder import UniversalCardEncoder, make_projector, init_hidden
from torch.utils.checkpoint import checkpoint

class GatedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.gate1 = nn.Parameter(torch.zeros(d_model))
        self.gate2 = nn.Parameter(torch.zeros(d_model))

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.linear1.weight, gain=1.0)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.orthogonal_(self.linear2.weight, gain=0.01)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, src, src_key_padding_mask=None):
        nx = self.norm1(src)
        attn_out, _ = self.self_attn(nx, nx, nx, key_padding_mask=src_key_padding_mask)
        src = src + self.gate1 * attn_out
        
        nx = self.norm2(src)
        ffn_out = self.linear2(self.act(self.linear1(nx)))
        src = src + self.gate2 * ffn_out
        
        return src

class GatedTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedTransformerLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_key_padding_mask=src_key_padding_mask)
        return self.norm(src)

def init_head(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def make_head(in_dim, h_dim, out_dim):
    m = nn.Sequential(
        nn.Linear(in_dim, h_dim * 2), nn.LayerNorm(h_dim * 2), nn.GELU(),
        nn.Linear(h_dim * 2, h_dim),  nn.LayerNorm(h_dim),     nn.GELU(),
        nn.Linear(h_dim, out_dim)
    )
    m[0].apply(init_hidden)
    m[3].apply(init_hidden)
    m[-1].apply(init_head)
    return m

class BalatroShopModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.dim = 128
        self.h = 512
        self.num_tokens = 36
        self.card_encoder = UniversalCardEncoder()
        
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_tokens, self.dim) * 0.02)
        self.skip_token = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)
        self.money_proj = make_projector(1, self.dim)
        self.reroll_proj = make_projector(1, self.dim)
        self.goal_proj = make_projector(6, self.dim)
        self.levels_proj = make_projector(6, self.dim)
        
        self.voucher_emb = nn.Embedding(18, self.dim)
        nn.init.normal_(self.voucher_emb.weight, mean=0.0, std=0.02)
        self.voucher_list_proj = make_projector(17, self.dim)
        
        self.blind_emb = nn.Embedding(32, self.dim)
        nn.init.normal_(self.blind_emb.weight, mean=0.0, std=0.02)
        
        self.transformer = GatedTransformerEncoder(
            d_model=self.dim, nhead=8, dim_feedforward=self.h, num_layers=4
        )
        
        self.end_head = make_head(self.dim, self.h, 1)
        self.entity_head = make_head(self.dim, self.h, 1)
        self.planet_head = make_head(self.dim, self.h, 1)
        self.value_head = make_head(self.dim, self.h, 1)
        
        self._val_out = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        B = obs["dollars"].shape[0]
        device = obs["dollars"].device
        
        sk_tok = self.skip_token.expand(B, -1, -1)
        cls_tok = self.cls_token.expand(B, -1, -1)
        m_tok = self.money_proj(obs["dollars"] / 100.0).unsqueeze(1)
        r_tok = self.reroll_proj(obs["reroll_price"] / 100.0).unsqueeze(1)
        
        bl_tok = self.blind_emb(obs["blind_index"].long().view(-1)).unsqueeze(1)
        g_tok = self.goal_proj(torch.cat([obs["goal"] / 100000.0, torch.log10(obs["goal"] + 1) / 10.0, obs["round"] / 100.0, (25 - obs["round"]) / 100.0, obs["owned_joker_count"] / 10.0, (5 - obs["owned_joker_count"]) / 10.0], dim=1)).unsqueeze(1)
        lvls = torch.stack([obs["hand_stats"]["level"] / 10.0, obs["hand_stats"]["played_count"] / 100.0, obs["hand_stats"]["chips"] / 1000.0, torch.log10(obs["hand_stats"]["chips"] + 1.0) / 3.0, obs["hand_stats"]["mult"] / 100.0, torch.log10(obs["hand_stats"]["mult"] + 1.0) / 2.0], dim=-1)
        l_toks = self.levels_proj(lvls)
        
        s_toks = self.card_encoder(obs["shop_cards"])
        b_toks = self.card_encoder(obs["booster_cards"])
        o_toks = self.card_encoder(obs["owned_jokers"])
        p_toks = self.card_encoder(obs["pack_cards"])
        
        v_shelf_tok = self.voucher_emb(obs["shelf_voucher"].long().view(-1)).unsqueeze(1)
        v_owned_tok = self.voucher_list_proj(obs["owned_vouchers"]).unsqueeze(1)
        
        seq = torch.cat([sk_tok, m_tok, r_tok, g_tok, bl_tok, l_toks, s_toks, b_toks, o_toks, p_toks, v_shelf_tok, v_owned_tok, cls_tok], dim=1) + self.pos_emb
        
        pad_mask = torch.zeros(B, self.num_tokens, dtype=torch.bool, device=device)
        pad_mask[:, 17:21] = (obs["shop_cards"]["indices"] == 0)
        pad_mask[:, 21:23] = (obs["booster_cards"]["indices"] == 0)
        pad_mask[:, 23:28] = (obs["owned_jokers"]["indices"] == 0)
        pad_mask[:, 28:33] = (obs["pack_cards"]["indices"] == 0)
        pad_mask[:, 33] = (obs["shelf_voucher"].view(-1) == 0)
        
        if self.training:
            features = checkpoint(lambda t: self.transformer(t, src_key_padding_mask=pad_mask), seq, use_reentrant=False)
        else:
            features = self.transformer(seq, src_key_padding_mask=pad_mask)
        
        raw_val = self.value_head(features[:, 35, :]).squeeze(-1)
        self._val_out = torch.sigmoid(raw_val) * 4.0
        
        l0 = self.end_head(features[:, 0, :])
        l1 = self.entity_head(features[:, 2, :])
        l2_5 = self.entity_head(features[:, 17:21, :]).squeeze(-1)
        l6_7 = self.entity_head(features[:, 21:23, :]).squeeze(-1)
        l8_12 = self.entity_head(features[:, 23:28, :]).squeeze(-1)
        
        is_celestial = obs["current_pack_is_celestial"].squeeze(-1).bool()
        planet_scores = torch.full((B, 5), -1e9, device=device)
        pack_indices = obs["pack_card_level_indices"]
        batch_range = torch.arange(B, device=device)
        for i in range(5):
            l_idx = pack_indices[:, i].long()
            valid = l_idx >= 0
            if valid.any():
                safe_l_idx = torch.clamp(l_idx, 0, 11)
                target_lv_feat = features[batch_range[valid], 5 + safe_l_idx[valid], :]
                planet_scores[valid, i] = self.planet_head(target_lv_feat).squeeze(-1)
        
        pack_scores = self.entity_head(features[:, 28:33, :]).squeeze(-1)
        l13_17 = torch.where(is_celestial.unsqueeze(-1), planet_scores, pack_scores)
        l18 = torch.full((B, 1), -1e8, device=device)
        l19 = self.entity_head(features[:, 33, :])
        
        total_logits = torch.cat([l0, l1, l2_5, l6_7, l8_12, l13_17, l18, l19], dim=1)
        
        money = obs["dollars"].view(B)
        j_count = obs["owned_joker_count"].view(B)
        in_pack = obs["in_pack_selection"].view(B).bool()
        
        shop_m = []
        shop_m.append(torch.zeros(B, device=device, dtype=torch.bool))
        shop_m.append(money < obs["reroll_price"].view(B))
        for i in range(4):
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
            valid = torch.where(is_celestial, is_p, j_legal)
            pack_m.append(~valid)
        pack_mask_tensor = torch.stack(pack_m, dim=1)
        
        voucher_mask = ~((obs["shelf_voucher"].view(-1) != 0) & (money >= obs["voucher_price"].view(-1)))

        masked_shop = torch.where(shop_mask_tensor, torch.tensor(-1e9, device=device), total_logits[:, :13])
        masked_pack = torch.where(pack_mask_tensor, torch.tensor(-1e9, device=device), total_logits[:, 13:18])
        masked_voucher = torch.where(voucher_mask.unsqueeze(-1), torch.tensor(-1e9, device=device), total_logits[:, 19:20])
        
        final_shop_part = torch.where(in_pack.unsqueeze(-1), torch.tensor(-1e9, device=device), torch.cat([masked_shop, masked_voucher], dim=1))
        
        skip_pack_val = torch.where(is_celestial, torch.tensor(-1e9, device=device), total_logits[:, 18])
        final_pack_part = torch.where(~in_pack.unsqueeze(-1), torch.tensor(-1e9, device=device), torch.cat([masked_pack, skip_pack_val.unsqueeze(1)], dim=1))

        res_logits = torch.cat([final_shop_part[:, :13], final_pack_part, final_shop_part[:, 13:]], dim=1)

        return res_logits, []

    def value_function(self):
        return self._val_out