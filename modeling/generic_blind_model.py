import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from gym_envs.universal_card_encoder import UniversalCardEncoder

class BalatroBlindModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.dim = 64
        self.num_tokens = 31
        self.h = 128
        self.card_encoder = UniversalCardEncoder()
        
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_tokens, 63) * 0.02)
        self.blind_emb = nn.Embedding(32, 63)

        self.goal_proj = nn.Linear(1, 63)
        self.state_proj = nn.Linear(2, 63)
        self.levels_proj = nn.Linear(5, 63)
        self.deck_proj = nn.Linear(17, 63)
        self.stop_token = nn.Parameter(torch.randn(1, 1, 63))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=4, dim_feedforward=self.h, 
            activation="gelu", batch_first=True, dropout=0.0,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        def make_head(out_dim):
            return nn.Sequential(
                nn.Linear(self.dim, self.h),
                nn.LayerNorm(self.h),
                nn.GELU(),
                nn.Linear(self.h, out_dim)
            )

        self.mode_head = make_head(2)
        self.selection_head = make_head(1)
        self.value_head = make_head(1)
        
        self._val_out = None
        self._cached_seq_63 = None
        self._hand_invalid_mask = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        B = obs["hands_left"].shape[0]
        device = obs["hands_left"].device

        self._hand_invalid_mask = (obs["hand"]["indices"] == 0) & (obs["hand"]["rank"] == 0)

        blind_id = obs["blind_index"].long().view(-1)
        blind_identity = self.blind_emb(blind_id)

        g_tok = (self.goal_proj(obs["goal"] / 10000.0) + blind_identity).unsqueeze(1)
        s_tok = self.state_proj(torch.cat([obs["hands_left"], obs["discards_left"]], dim=1)).unsqueeze(1)
        
        levels = torch.stack([
            obs["hand_stats"]["level"], 
            obs["hand_stats"]["played_count"],
            obs["hand_stats"]["chips"] / 100.0, 
            obs["hand_stats"]["mult"] / 10.0,
            obs["hand_stats"]["played_this_blind"],
        ], dim=-1)
        l_toks = self.levels_proj(levels)
        
        d_tok = self.deck_proj(torch.cat([obs["deck_ranks"], obs["deck_suits"]], dim=1)).unsqueeze(1)
        
        j_tok, _ = self.card_encoder(obs["jokers"])
        j_tok = j_tok[:, :5, :] 
        
        h_tok, _ = self.card_encoder(obs["hand"])
        h_tok = h_tok[:, :10, :] 
        
        stop_tok = self.stop_token.expand(B, -1, -1)
        
        self._cached_seq_63 = torch.cat([g_tok, s_tok, l_toks, d_tok, j_tok, h_tok, stop_tok], dim=1) + self.pos_emb
        
        dummy_mask = torch.zeros(B, 31, 1, device=device)
        initial_feat = self.transformer(torch.cat([self._cached_seq_63, dummy_mask], dim=2))
        self._val_out = self.value_head(initial_feat[:, 0, :]).squeeze(-1)
        
        return torch.zeros(B, 1, device=device), []

    def value_function(self):
        return self._val_out

    def ar_step(self, hand_mask, step_idx, selected_mode=None):
        B = self._cached_seq_63.shape[0]
        device = self._cached_seq_63.device
        
        full_mask = torch.zeros(B, 31, 1, device=device)
        
        full_mask[:, 20:30, 0] = hand_mask 
        
        if selected_mode is not None:
            mode_flag = (selected_mode * 2 - 1).float() 
            full_mask[:, 0, 0] = mode_flag
        
        x = torch.cat([self._cached_seq_63, full_mask], dim=2)
        features = self.transformer(x)
        
        mode_logits = self.mode_head(features[:, 0, :])
        
        card_features = features[:, 20:31, :]
        card_logits = self.selection_head(card_features).squeeze(-1)
        
        safe_card_logits = card_logits.clone()
        safe_card_logits[:, :10] = safe_card_logits[:, :10].masked_fill(hand_mask.bool(), -1e9)
        safe_card_logits[:, :10] = safe_card_logits[:, :10].masked_fill(self._hand_invalid_mask, -1e9)
        if step_idx == 1:
            safe_card_logits[:, 10] = -1e9
            
        return mode_logits, safe_card_logits