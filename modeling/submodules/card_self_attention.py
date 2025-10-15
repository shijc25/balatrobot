import torch.nn as nn

from modeling.optional_torchrl import noisy_linear_cls


class CardSelfAttention(nn.Module):
    """1 layer transformer encoder block over a sequence of cards
    Use padding mask to handle variable number of cards
    """

    def __init__(
        self, d_model: int, n_heads: int = 4, dropout: float = 0.0, noisy: bool = False
    ):
        super().__init__()
        self.lin_cls = noisy_linear_cls(noisy)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout
        )
        self.pre_ln = nn.LayerNorm(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            self.lin_cls(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            self.lin_cls(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, padding=None):
        if padding is None:
            attn_out, _ = self.attn(x, x, x)
        else:
            attn_out, _ = self.attn(x, x, x, key_padding_mask=padding)
        x = self.ln1(x + attn_out)

        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x
