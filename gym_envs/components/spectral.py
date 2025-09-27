from gym_envs.base_card import BaseCard
from gym_envs.components.card import Card
from gym_envs.joker import Joker
from random import choice, random
from copy import deepcopy

from gym_envs.components.consumable import Consumable


class SpectralCard(Consumable):
    def __init__(self, name, i, num_targets=0, segment=BaseCard.Segments.PLANET):
        super().__init__(name, num_targets=num_targets, segment=segment)
        self.universal_index = BaseCard.FIRST_SPECTRAL_INDEX + i

    @staticmethod
    def from_gamestate_card(gamestate_card):
        label = gamestate_card.get("label", None)
        if label is None:
            return None
        for i, card in enumerate(SpectralCard.all()):
            if card.name == label:
                return card

        return None

    def trigger(
        self,
        targets,
        gamestate,
    ):
        enh_pool = [
            x
            for x in BaseCard.Enhancements.ALL.values()
            if x != BaseCard.Enhancements.NORMAL
        ]
        match self.name:
            case "Familiar":
                target = choice(gamestate.hand.cards)
                gamestate.destroy_card(target)
                for _ in range(3):
                    new_card = Card.random(vanilla_only=True)
                    new_card.value = choice([11, 12, 13])  # Face cards only
                    new_card.enhancement = choice(enh_pool)
                    gamestate.create_card(new_card)
                return {}
            case "Grim":
                if len(gamestate.hand.cards) == 0:
                    return {}
                target = choice(gamestate.hand.cards)
                gamestate.destroy_card(target)
                for _ in range(2):
                    new_card = Card.random(vanilla_only=True)
                    new_card.value = 14
                    new_card.enhancement = choice(enh_pool)
                    gamestate.create_card(new_card)
                return {}
            case "Incantation":
                target = choice(gamestate.hand.cards)
                gamestate.destroy_card(target)
                for _ in range(4):
                    new_card = Card.random(vanilla_only=True)
                    new_card.value = choice([2, 3, 4, 5, 6, 7, 8, 9, 10])
                    new_card.enhancement = choice(enh_pool)
                    gamestate.create_card(new_card)
                return {}
            case "Talisman":
                for target in targets:
                    target.seal = BaseCard.Seals.GOLD
                return {}
            case "Aura":
                for target in targets:
                    target.edition = choice(
                        [
                            BaseCard.Editions.FOIL,
                            BaseCard.Editions.HOLOGRAPHIC,
                            BaseCard.Editions.POLYCHROME,
                        ]
                    )
                return {}
            case "Wraith":
                joker = Joker.random(force_rarity=2)
                gamestate.add_joker(joker)
                gamestate.dollars = 0
                return {}
            case "Sigil":
                suit = choice(Card.SUITS)
                for card in gamestate.hand:
                    card.suit = suit
                return {}
            case "Ouija":
                rank = choice(Card.RANKS)
                for card in gamestate.hand:
                    card.rank = rank
                gamestate.hand_size_mod -= 1
                return {}
            case "Ectoplasm":
                no_edition_jokers = [
                    j
                    for j in gamestate.owned_jokers
                    if j.edition is BaseCard.Editions.NO_EDITION
                ]
                if len(no_edition_jokers) > 0:
                    joker = choice(no_edition_jokers)
                    joker.edition = BaseCard.Editions.NEGATIVE
                gamestate.hand_size_mod -= 1
                return {}
            case "Immolate":
                for _ in range(5):
                    if len(gamestate.hand.cards) == 0:
                        break
                    target = choice(gamestate.hand.cards)
                    gamestate.destroy_card(target)
                gamestate.dollars += 20
            case "Ankh":
                if len(gamestate.owned_jokers) == 0:
                    return {}

                chosen_joker = choice(gamestate.owned_jokers)
                new_joker = deepcopy(chosen_joker)
                if new_joker.edition == BaseCard.Editions.NEGATIVE:
                    new_joker.edition = BaseCard.Editions.NO_EDITION
                for x in gamestate.owned_jokers:
                    if x != chosen_joker:
                        gamestate.destroy_card(x)
                gamestate.add_joker(new_joker, fix_price=False)
                return {}
            case "Deja Vu":
                for target in targets:
                    target.seal = BaseCard.Seals.RED
                return {}
            case "Hex":
                no_edition_jokers = [
                    j
                    for j in gamestate.owned_jokers
                    if j.edition is BaseCard.Editions.NO_EDITION
                ]
                if len(no_edition_jokers) == 0:
                    return {}
                chosen_joker = choice(no_edition_jokers)
                chosen_joker.edition = BaseCard.Editions.POLYCHROME
                for x in gamestate.owned_jokers:
                    if x != chosen_joker:
                        gamestate.destroy_card(x)
                return {}
            case "Trance":
                for target in targets:
                    target.seal = BaseCard.Seals.BLUE
                return {}
            case "Medium":
                for target in targets:
                    target.seal = BaseCard.Seals.PURPLE
                return {}
            case "Cryptid":
                for target in targets:
                    gamestate.create_card(deepcopy(target))
                    gamestate.create_card(deepcopy(target))
                return {}
            case "The Soul":
                new_joker = Joker.random(force_rarity=3)
                gamestate.add_joker(new_joker)
                return {}
            case "Black Hole":
                gamestate.black_hole()
                return {}
            case _:
                print(f"Spectral card {self.name} does not have a trigger defined.")
                return {}

    @staticmethod
    def all():
        return [
            SpectralCard("Familiar", 0, num_targets=0),
            SpectralCard("Grim", 1, num_targets=0),
            SpectralCard("Incantation", 2, num_targets=0),
            SpectralCard("Talisman", 3, num_targets=1),
            SpectralCard("Aura", 4, num_targets=1),
            SpectralCard("Wraith", 5, num_targets=0),
            SpectralCard("Sigil", 6, num_targets=0),
            SpectralCard("Ouija", 7, num_targets=0),
            SpectralCard("Ectoplasm", 8, num_targets=0),
            SpectralCard("Immolate", 9, num_targets=0),
            SpectralCard("Ankh", 10, num_targets=0),
            SpectralCard("Deja Vu", 11, num_targets=1),
            SpectralCard("Hex", 12, num_targets=0),
            SpectralCard("Trance", 13, num_targets=1),
            SpectralCard("Medium", 14, num_targets=1),
            SpectralCard("Cryptid", 15, num_targets=1),
            SpectralCard("The Soul", 16, num_targets=0),
            SpectralCard("Black Hole", 17, num_targets=0),
        ]

    @staticmethod
    def from_name(name):
        for card in SpectralCard.all():
            if card.name == name:
                return card
        return None

    @staticmethod
    def random():
        sample = choice(SpectralCard.all())
        # These can only be found in boosters, and need special handling for different odds
        while sample.name in ["The Soul", "Black Hole"]:
            sample = choice(SpectralCard.all())
        return sample
