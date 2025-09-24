from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym_envs.base_card import BaseCard
from gym_envs.universal_card_encoder import UniversalCardEncoder
from modeling.submodules.card_self_attention import CardSelfAttention
from modeling.bitwise_hand_types import BitwiseHandTypes


class DeepSetEncoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, pool: str = "mean"):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(d_hidden, d_out), nn.GELU(), nn.LayerNorm(d_out)
        )
        assert pool in {"mean", "sum", "max"}
        self.pool = pool

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.phi(x)
        if mask is not None:
            if self.pool == "max":
                h = h.masked_fill(mask.unsqueeze(-1), float("-inf"))
            else:
                h = h.masked_fill(mask.unsqueeze(-1), 0.0)
        if self.pool == "mean":
            denom = (
                (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
                if mask is not None
                else x.new_tensor(x.size(1)).unsqueeze(0)
            )
            pooled = h.sum(dim=1) / denom
        elif self.pool == "sum":
            pooled = h.sum(dim=1)
        else:
            pooled, _ = h.max(dim=1)
        return self.rho(pooled)


class NextCardFactoredHead(nn.Module):
    def __init__(
        self,
        card_embedding_size: int,
        num_attention_layers: int = 1,
        card_encoder: Optional[UniversalCardEncoder] = None,
        use_remaining_self_attn: bool = True,
        reachable_hands_mode: str = "mlp_inputs",
        reachable_dim: int = 8,
    ):
        super().__init__()
        self.card_embedding_size = card_embedding_size
        self.use_remaining_self_attn = use_remaining_self_attn
        self.reachable_hands_mode = reachable_hands_mode
        self.reachable_dim = reachable_dim

        self.empty_rem_embed = nn.Parameter(torch.zeros(card_embedding_size))
        if use_remaining_self_attn:
            self.remaining_self_attn = nn.ModuleList(
                [
                    CardSelfAttention(
                        d_model=card_embedding_size, n_heads=4, dropout=0.0, noisy=False
                    )
                    for _ in range(num_attention_layers)
                ]
            )
        else:
            self.remaining_self_attn = None

        self.sel_encoder = DeepSetEncoder(
            card_embedding_size, card_embedding_size, card_embedding_size, pool="mean"
        )
        self.rem_encoder = DeepSetEncoder(
            card_embedding_size, card_embedding_size, card_embedding_size, pool="mean"
        )

        pch_inputs = card_embedding_size * 4 + self.reachable_dim
        self.per_card_head = nn.Sequential(
            nn.Linear(pch_inputs, pch_inputs // 2),
            nn.GELU(),
            nn.Linear(pch_inputs // 2, 1),
        )

        self.stop_head = nn.Sequential(
            nn.Linear(card_embedding_size * 3, card_embedding_size),
            nn.GELU(),
            nn.Linear(card_embedding_size, 1),
        )

        self.tau = nn.Parameter(torch.tensor(10.0))

        if card_encoder is None:
            card_encoder = UniversalCardEncoder(embedding_size=card_embedding_size)
        self.card_encoder = card_encoder

    @staticmethod
    def _pool_tokens(x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.mean(dim=1), (x.size(-1),))

    def forward(
        self,
        mode: torch.Tensor,
        hand_embeddings: torch.Tensor,
        hand_padding: torch.Tensor,
        already_selected_cards_mask: torch.Tensor,
        h_tokens: torch.Tensor,
        card_obs: dict,
    ) -> torch.Tensor:
        B, N, D = hand_embeddings.shape

        # Check if any row in the batch has all 1s in the hand_padding
        all_padded = hand_padding.all(dim=1)  # shape: (B,)
        if all_padded.any():
            print("All padded rows found")

        selected_mask = already_selected_cards_mask[:, :-1].bool()
        pad_mask = hand_padding.bool()
        unselected_mask = (~selected_mask) & (~pad_mask)

        e_sel = self.sel_encoder(hand_embeddings, mask=(~selected_mask) | pad_mask)

        rem_tokens = hand_embeddings
        rem_mask = ~unselected_mask
        if self.remaining_self_attn is not None:
            # process only rows that actually have remaining (unselected) cards
            has_remaining = unselected_mask.any(
                dim=1
            )  # True for rows that DO have remaining cards
            no_remaining = ~has_remaining

            x = rem_tokens.clone()
            if has_remaining.any():
                idx = torch.nonzero(has_remaining, as_tuple=True)[0]
                x_sub = rem_tokens[idx]
                padding_sub = rem_mask[idx]
                for layer in self.remaining_self_attn:
                    x_sub = layer(x_sub, padding=padding_sub)
                x_sub = F.layer_norm(x_sub, (x_sub.size(-1),))
                x[idx] = x_sub

            # now encode and replace rows with no remaining by empty embedding
            e_rem = self.rem_encoder(x, mask=rem_mask)
            if no_remaining.any():
                e_rem[no_remaining] = self.empty_rem_embed
        else:
            e_rem = self.rem_encoder(rem_tokens, mask=rem_mask)

        h_ctx = self._pool_tokens(h_tokens)

        reachable_hands = BitwiseHandTypes.possible_hand_types(
            card_obs, illegal_mask=hand_padding, must_include_mask=selected_mask.long()
        )
        assert (
            reachable_hands.size(-1) == self.reachable_dim
        ), f"Expected {self.reachable_dim}, got {reachable_hands.size(-1)}"

        e_sel_t = e_sel.unsqueeze(1).expand(B, N, D)
        e_rem_t = e_rem.unsqueeze(1).expand(B, N, D)
        h_ctx_t = h_ctx.unsqueeze(1).expand(B, N, D)
        # Only use reachability bits in PLAY mode; zero them in DISCARD mode so discards are never biased by hand-type reachability.
        is_play = mode.bool().view(B, 1, 1)  # (B,1,1)
        rh_play = reachable_hands.unsqueeze(1).expand(B, N, self.reachable_dim)
        rh_zero = torch.zeros_like(rh_play)
        rh_t = torch.where(is_play, rh_play, rh_zero)

        feats = [hand_embeddings, e_sel_t, e_rem_t, h_ctx_t, rh_t]
        x = torch.cat(feats, dim=-1)
        logits_cards = self.per_card_head(x).squeeze(-1)

        # Mask out already-selected and padded cards. Use -inf so softmax
        # produces zero mass for those positions and it's unambiguous.
        logits_cards = logits_cards.masked_fill(~unselected_mask, float("-inf"))
        # Debug: detect rows where all card logits are -inf
        # if torch.isneginf(logits_cards).all(dim=1).any():
        #     print("All card logits are masked for at least one batch row")

        stop_in = torch.cat([e_sel, e_rem, h_ctx], dim=-1)
        logit_stop = self.stop_head(stop_in).squeeze(-1)

        logits = torch.cat([logits_cards, logit_stop.unsqueeze(-1)], dim=-1)
        return logits
