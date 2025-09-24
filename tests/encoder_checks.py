from gym_envs.joker import Joker
from gym_envs.pseudo.card import Card
from gym_envs.pseudo.tarot import TarotCard
from gym_envs.pseudo.spectral import SpectralCard
from gym_envs.pseudo.planet import PlanetCard


if __name__ == "__main__":
    B = 1000

    card_samples = {}
    for card_class in [Card, TarotCard, SpectralCard, PlanetCard, Joker]:
        card_samples[card_class.__name__] = [card_class.random() for _ in range(B)]

    card_uids = {
        card_class: set(
            [card.get_universal_index() for card in card_samples[card_class]]
        )
        for card_class in card_samples
    }

    # Make sure that all card types have unique universal indices
    for card_class in card_samples:
        for other_class in card_samples:
            if card_class == other_class:
                continue
            if len(card_uids[card_class].intersection(card_uids[other_class])) != 0:
                print(
                    f"Duplicate universal indices found between {card_class} and {other_class}"
                )
                print(
                    f"Indices: {card_uids[card_class].intersection(card_uids[other_class])}"
                )

    playing_cards = [Card.random(vanilla_only=True) for _ in range(B)]
    suit_ids = set(card.get_u_suit_index() for card in playing_cards)
    rank_ids = set(card.get_u_rank_index() for card in playing_cards)
    print(suit_ids)
    print(rank_ids)
