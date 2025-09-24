from gym_envs.universal_card_encoder import UniversalCardEncoder
from modeling.submodules.card_self_attention import CardSelfAttention


class SharedParameters:
    embedding_size = 64
    encoder = UniversalCardEncoder(embedding_size=embedding_size)
    # deck_attention = CardSelfAttention(
    #     d_model=embedding_size, n_heads=4, dropout=0.0, noisy=False
    # )
