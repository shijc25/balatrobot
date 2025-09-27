from gym_envs.base_card import BaseCard
from random import choice, random
from gym_envs.joker import Joker
from gym_envs.components.consumable import Consumable


class TarotCard(Consumable):
    def __init__(self, name, i, num_targets=0, segment=BaseCard.Segments.TAROT):
        super().__init__(name, num_targets=num_targets, segment=segment)
        self.universal_index = BaseCard.FIRST_TAROT_INDEX + i

    @staticmethod
    def from_gamestate_card(gamestate_card):
        label = gamestate_card.get("label", None)
        if label is None:
            return None
        for i, card in enumerate(TarotCard.all()):
            if card.name == label:
                return card

        return None

    def trigger(self, targets, gamestate):
        gamestate.tarot_used += 1
        match self.name:
            case "The Sun":
                for target in targets:
                    target.suit = "Hearts"
                return {}
            case "The Moon":
                for target in targets:
                    target.suit = "Clubs"
                return {}
            case "The Star":
                for target in targets:
                    target.suit = "Diamonds"
                return {}
            case "The World":
                for target in targets:
                    target.suit = "Spades"
                return {}
            case "Strength":
                for target in targets:
                    target.value += 1
                    if target.value > 14:
                        target.value = 2
                return {}
            case "Death":
                targets[1].copy_from(targets[0])
                return {}
            case "The Hermit":
                income = max(min(20, gamestate.dollars), 0)
                gamestate.dollars += income
                return {}
            case "Temperance":
                income = sum(j.value for j in gamestate.owned_jokers)
                gamestate.dollars += income
                return {}
            case "Judgment":
                new_joker = Joker.random()
                gamestate.add_joker(new_joker)
                return {}
            case "The Hanged Man":
                for target in targets:
                    gamestate.destroy_card(target)
                return {}
            case "The Magician":
                for x in targets:
                    x.enhancement = BaseCard.Enhancements.LUCKY
                gamestate.update_unlocked_jokers()
                return {}
            case "The Hierophant":
                for x in targets:
                    x.enhancement = BaseCard.Enhancements.BONUS
                return {}
            case "The Empress":
                for x in targets:
                    x.enhancement = BaseCard.Enhancements.MULT
                return {}
            case "The Wheel of Fortune":
                no_edition_jokers = [
                    j
                    for j in gamestate.owned_jokers
                    if j.edition is BaseCard.Editions.NO_EDITION
                ]
                if random() < 0.25 and len(no_edition_jokers) > 0:
                    joker = choice(no_edition_jokers)
                    joker.edition = choice(
                        [
                            BaseCard.Editions.FOIL,
                            BaseCard.Editions.HOLOGRAPHIC,
                            BaseCard.Editions.POLYCHROME,
                        ]
                    )
                return {}
            case "The Lovers":
                for x in targets:
                    x.enhancement = BaseCard.Enhancements.WILD
                return {}
            case "The Chariot":
                for x in targets:
                    x.enhancement = BaseCard.Enhancements.STEEL
                gamestate.update_unlocked_jokers()
                return {}
            case "Justice":
                for x in targets:
                    x.enhancement = BaseCard.Enhancements.GLASS
                gamestate.update_unlocked_jokers()
                return {}
            case "The Devil":
                for x in targets:
                    x.enhancement = BaseCard.Enhancements.GOLD
                gamestate.update_unlocked_jokers()
                return {}
            case "The Tower":
                for x in targets:
                    x.enhancement = BaseCard.Enhancements.STONE
                gamestate.update_unlocked_jokers()
                return {}
            case _:
                print(f"Tarot card {self.name} does not have a trigger defined.")
                return {}

    @staticmethod
    def all():
        return [
            # TarotCard("The Fool", 0),
            TarotCard("The Magician", 1, num_targets=2),
            # TarotCard("The High Priestess", 2),
            TarotCard("The Empress", 3, num_targets=2),
            # TarotCard("The Emperor", 4),
            TarotCard("The Hierophant", 5, num_targets=2),
            TarotCard("The Lovers", 6, num_targets=1),
            TarotCard("The Chariot", 7, num_targets=1),
            TarotCard("Justice", 8, num_targets=1),
            TarotCard("The Hermit", 9),
            TarotCard("The Wheel of Fortune", 10),
            TarotCard("Strength", 11, num_targets=2),
            TarotCard("The Hanged Man", 12, num_targets=2),
            TarotCard("Death", 13, num_targets=2),
            TarotCard("Temperance", 14),
            TarotCard("The Devil", 15, num_targets=1),
            TarotCard("The Tower", 16, num_targets=1),
            TarotCard("The Star", 17, num_targets=3),
            TarotCard("The Moon", 18, num_targets=3),
            TarotCard("The Sun", 19, num_targets=3),
            TarotCard("Judgment", 20),
            TarotCard("The World", 21, num_targets=3),
        ]

    @staticmethod
    def random():
        return choice(TarotCard.all())
