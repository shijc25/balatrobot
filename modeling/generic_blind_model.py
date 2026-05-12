import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from gym_envs.universal_card_encoder import UniversalCardEncoder, make_projector, init_hidden
from torch.utils.checkpoint import checkpoint

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

def init_transformer(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=1.414)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class BalatroBlindModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.dim = 128
        self.h = 512
        self.num_tokens = 35
        self.card_encoder = UniversalCardEncoder()
        
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_tokens, self.dim) * 0.02)
        self.blind_emb = nn.Embedding(32, self.dim)
        nn.init.normal_(self.blind_emb.weight, std=0.02)
        
        self.goal_proj = make_projector(4, self.dim)
        self.state_proj = make_projector(2, self.dim - 2)
        self.levels_proj = make_projector(7, self.dim)
        self.deck_proj = make_projector(17, self.dim)
        self.summary_proj = make_projector(21, self.dim)
        
        self.stop_token = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=8, dim_feedforward=self.h,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.transformer.apply(init_transformer)

        self.mode_head = make_head(self.dim * 2, self.h, 2)
        self.selection_head = make_head(self.dim, self.h, 1)
        self.value_head = make_head(self.dim, self.h, 1)
        
        self._val_out = None

    def _get_summary_vector(self, ranks, suits, mask):
        B, L = ranks.shape
        m = mask.unsqueeze(-1)
        suit_one_hot = nn.functional.one_hot(suits.long().clamp(0, 5), num_classes=6)[:, :, 1:5]
        suit_counts = (suit_one_hot.float() * m).sum(dim=1)
        red_count = suit_counts[:, 0:2].sum(dim=1, keepdim=True)
        black_count = suit_counts[:, 2:4].sum(dim=1, keepdim=True)
        rank_one_hot = nn.functional.one_hot(ranks.long().clamp(0, 14), num_classes=15)[:, :, 1:14]
        rank_counts = (rank_one_hot.float() * m).sum(dim=1)
        face_mask = ((ranks >= 10) & (ranks <= 12)).float()
        face_count = (face_mask * mask).sum(dim=1, keepdim=True)
        card_count = mask.sum(dim=1, keepdim=True)
        return torch.cat([suit_counts, red_count, black_count, rank_counts, face_count, card_count], dim=1) / 10.0

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        B = obs["hands_left"].shape[0]
        device = obs["hands_left"].device
        
        hand_invalid_mask = (obs["hand"]["indices"] == 0) & (obs["hand"]["rank"] == 0)
        joker_invalid_mask = (obs["jokers"]["indices"] == 0)
        h_ranks, h_suits = obs["hand"]["rank"].float(), obs["hand"]["suit"].float()
        
        blind_identity = self.blind_emb(obs["blind_index"].long().view(-1))
        g_tok = self.goal_proj(torch.cat([obs["chips"] / 100000.0, torch.log10(obs["chips"] + 1) / 10.0, obs["chips_average"] / 100000.0, torch.log10(obs["chips_average"] + 1) / 10.0], dim=1)).unsqueeze(1)
        bl_tok = blind_identity.unsqueeze(1)
        
        s_tok_base = self.state_proj(torch.cat([obs["hands_left"] / 10.0, obs["discards_left"] / 10.0], dim=1)).unsqueeze(1)
        s_tok = torch.cat([s_tok_base, torch.zeros(B, 1, 2, device=device)], dim=-1)
        
        levels = torch.stack([
            obs["hand_stats"]["level"] / 10.0, obs["hand_stats"]["played_count"] / 100.0,
            obs["hand_stats"]["chips"] / 1000.0, torch.log10(obs["hand_stats"]["chips"] + 1.0) / 3.0,
            obs["hand_stats"]["mult"] / 100.0, torch.log10(obs["hand_stats"]["mult"] + 1.0) / 2.0,
            obs["hand_stats"]["played_this_blind"],
        ], dim=-1)
        l_toks = self.levels_proj(levels)
        d_tok = self.deck_proj(torch.cat([obs["deck_ranks"], obs["deck_suits"]], dim=1)).unsqueeze(1)
        j_tok = self.card_encoder(obs["jokers"])[:, :5, :]
        h_tok = self.card_encoder(obs["hand"])[:, :10, :]
        stop_tok = self.stop_token.expand(B, -1, -1)
        cls_tok = self.cls_token.expand(B, -1, -1)
        
        seq_base = torch.cat([g_tok, s_tok, bl_tok, l_toks, d_tok, j_tok, h_tok, stop_tok, cls_tok], dim=1)
        
        initial_hand_mask = torch.zeros(B, 10, device=device)
        initial_rem_mask = (~hand_invalid_mask.bool()).float()
        sel_sum_tok = self.summary_proj(self._get_summary_vector(h_ranks, h_suits, initial_hand_mask)).unsqueeze(1)
        rem_sum_tok = self.summary_proj(self._get_summary_vector(h_ranks, h_suits, initial_rem_mask)).unsqueeze(1)
        
        seq = torch.cat([seq_base, sel_sum_tok, rem_sum_tok], dim=1) + self.pos_emb

        x_base = seq[:, :, :126]
        logical_flags = torch.zeros(B, 35, 2, device=device)
        x_final = torch.cat([x_base, logical_flags], dim=-1)
        
        pad_mask = torch.zeros(B, self.num_tokens, dtype=torch.bool, device=device)
        pad_mask[:, 16:21] = joker_invalid_mask.bool()
        pad_mask[:, 21:31] = hand_invalid_mask.bool()
        
        if self.training:
            features = checkpoint(lambda t: self.transformer(t, src_key_padding_mask=pad_mask), x_final, use_reentrant=False)
        else:
            features = self.transformer(x_final, src_key_padding_mask=pad_mask)
        
        self._val_out = self.value_head(features[:, 32, :]).squeeze(-1)
        
        masks = torch.cat([hand_invalid_mask.float(), h_ranks, h_suits, joker_invalid_mask.float()], dim=1)
        return torch.cat([seq_base.reshape(B, -1), masks], dim=1), []
    
    def ar_step(self, cached_features, hand_mask, masks, step_idx, selected_mode=None):
        B = cached_features.shape[0]
        device = cached_features.device
        
        invalid_mask = masks[:, :10]
        h_ranks = masks[:, 10:20]
        h_suits = masks[:, 20:30]
        joker_invalid_mask = masks[:, 30:35]
        
        seq_base = cached_features.view(B, 33, 128)
        
        sel_sum_tok = self.summary_proj(self._get_summary_vector(h_ranks, h_suits, hand_mask)).unsqueeze(1)
        rem_mask_logic = (~invalid_mask.bool()) & (~hand_mask.bool())
        rem_sum_tok = self.summary_proj(self._get_summary_vector(h_ranks, h_suits, rem_mask_logic.float())).unsqueeze(1)
        
        x = torch.cat([seq_base, sel_sum_tok, rem_sum_tok], dim=1) + self.pos_emb
        
        x_base = x[:, :, :126]
        
        h_select_flags = hand_mask.unsqueeze(-1)
        
        if selected_mode is not None:
            m_play_flag = (selected_mode == 0).float().view(B, 1, 1)
            m_discard_flag = (selected_mode == 1).float().view(B, 1, 1)
            m_flags = torch.cat([m_discard_flag, m_play_flag], dim=-1)
        else:
            m_flags = torch.zeros(B, 1, 2, device=device)
        
        pre_h_flags = torch.cat([
            torch.zeros(B, 1, 2, device=device),
            m_flags,
            torch.zeros(B, 19, 2, device=device)
        ], dim=1)
        h_flags = torch.cat([torch.zeros(B, 10, 1, device=device), h_select_flags], dim=-1)
        post_h_flags = torch.zeros(B, 4, 2, device=device)
        
        logical_flags = torch.cat([pre_h_flags, h_flags, post_h_flags], dim=1)
        x_final = torch.cat([x_base, logical_flags], dim=-1)
        
        pad_mask = torch.zeros(B, self.num_tokens, dtype=torch.bool, device=device)
        pad_mask[:, 16:21] = joker_invalid_mask.bool()
        pad_mask[:, 21:31] = invalid_mask.bool()
        
        if self.training:
            features = checkpoint(lambda t: self.transformer(t, src_key_padding_mask=pad_mask), x_final, use_reentrant=False)
        else:
            features = self.transformer(x_final, src_key_padding_mask=pad_mask)
        
        mode_logits = self.mode_head(torch.cat([features[:, 0, :], features[:, 1, :]], dim=-1))
        card_features = features[:, 21:32, :]
        card_logits = self.selection_head(card_features).squeeze(-1)
        
        safe_card_logits = card_logits.clone()
        safe_card_logits[:, :10] = safe_card_logits[:, :10].masked_fill(hand_mask.bool(), -1e9)
        safe_card_logits[:, :10] = safe_card_logits[:, :10].masked_fill(invalid_mask.bool(), -1e9)
        
        if step_idx == 1:
            safe_card_logits[:, 10] = -1e9
        
        return mode_logits, safe_card_logits
    
    def value_function(self):
        return self._val_out