from modeling.submodules.next_card_factored_head import NextCardFactoredHead


# Unit tests for the NextCardFactoredHead class
import torch


def _make_card_obs(B, N, device):
    # card_obs expected keys: 'rank' (B,N), 'suit' (B,N)
    # ranks: 0-13 (14 classes), suits: 0-5 (6 classes)
    ranks = torch.randint(0, 14, (B, N), device=device)
    suits = torch.randint(0, 6, (B, N), device=device)
    return {"rank": ranks, "suit": suits}


# Simple smoke tests
def test_forward_shapes():
    device = torch.device("cpu")
    B = 2
    N = 5
    D = 64

    model = NextCardFactoredHead(card_embedding_size=D, num_attention_layers=1)
    model.to(device)

    # mode: 1 for PLAY, 0 for DISCARD
    mode = torch.tensor([1, 0], device=device)

    # hand_embeddings: (B,N,D)
    hand_embeddings = torch.randn(B, N, D, device=device)

    # hand_padding: 1=pad, 0=real
    hand_padding = torch.zeros(B, N, dtype=torch.uint8, device=device)
    # make one padded position in second batch
    hand_padding[1, -1] = 1

    # already_selected_cards_mask: extra trailing token for stop
    already_selected = torch.zeros(B, N + 1, dtype=torch.uint8, device=device)
    # mark first card of batch 0 as selected
    already_selected[0, 0] = 1

    # h_tokens: arbitrary tokens, shape (B, T, D)
    h_tokens = torch.randn(B, 3, D, device=device)

    card_obs = _make_card_obs(B, N, device)

    logits = model.forward(
        mode, hand_embeddings, hand_padding, already_selected, h_tokens, card_obs
    )

    # logits should be (B, N+1)
    assert logits.shape == (B, N + 1), f"unexpected logits shape: {logits.shape}"


def test_all_padded_row():
    device = torch.device("cpu")
    B = 2
    N = 4
    D = 64

    model = NextCardFactoredHead(card_embedding_size=D, num_attention_layers=0)
    model.to(device)

    mode = torch.tensor([1, 1], device=device)
    hand_embeddings = torch.randn(B, N, D, device=device)

    # mark all padded in first batch
    hand_padding = torch.zeros(B, N, dtype=torch.uint8, device=device)
    hand_padding[0] = 1

    already_selected = torch.zeros(B, N + 1, dtype=torch.uint8, device=device)

    h_tokens = torch.randn(B, 2, D, device=device)
    card_obs = _make_card_obs(B, N, device)

    # This should run without raising and should print the debug message
    logits = model.forward(
        mode, hand_embeddings, hand_padding, already_selected, h_tokens, card_obs
    )
    assert logits.shape == (B, N + 1)


def test_all_selected_row():
    device = torch.device("cpu")
    B = 2
    N = 4
    D = 64

    model = NextCardFactoredHead(card_embedding_size=D, num_attention_layers=0)
    model.to(device)

    mode = torch.tensor([1, 1], device=device)
    hand_embeddings = torch.randn(B, N, D, device=device)

    # mark all selected in first batch
    hand_padding = torch.zeros(B, N, dtype=torch.uint8, device=device)
    hand_padding[0] = 1

    already_selected = torch.zeros(B, N + 1, dtype=torch.uint8, device=device)

    h_tokens = torch.randn(B, 2, D, device=device)
    card_obs = _make_card_obs(B, N, device)

    # This should run without raising and should print the debug message
    logits = model.forward(
        mode, hand_embeddings, hand_padding, already_selected, h_tokens, card_obs
    )
    assert logits.shape == (B, N + 1)
    assert not torch.isnan(logits).any(), "logits contain NaN"
    assert (logits > -1e8).any(), "All logits are <= -1e8"


# Run tests
test_forward_shapes()
test_all_padded_row()
test_all_selected_row()
print("next_card_factored_head.py: unit tests passed")
