from gym_envs.base_card import BaseCard
from random import choice

from gym_envs.components.consumable import Consumable


class PlanetCard(Consumable):
    def __init__(self, name, i, hand_type, segment=BaseCard.Segments.PLANET):
        super().__init__(name, num_targets=0, segment=segment)
        self.hand_type = hand_type
        self.universal_index = BaseCard.FIRST_PLANET_INDEX + i

    @staticmethod
    def from_gamestate_card(gamestate_card):
        label = gamestate_card.get("label", None)
        if label is None:
            return None
        for i, planet in enumerate(PlanetCard.all()):
            if planet.name == label:
                return planet

        return None

    def trigger(self, targets, gamestate):
        gamestate.use_planet(self)

    @staticmethod
    def base_set():
        return [
            PlanetCard("Pluto", 0, "High Card"),
            PlanetCard("Mercury", 1, "Pair"),
            PlanetCard("Uranus", 2, "Two Pair"),
            PlanetCard("Venus", 3, "Three of a Kind"),
            PlanetCard("Saturn", 4, "Straight"),
            PlanetCard("Jupiter", 5, "Flush"),
            PlanetCard("Earth", 6, "Full House"),
            PlanetCard("Mars", 7, "Four of a Kind"),
            PlanetCard("Neptune", 8, "Straight Flush"),
        ]

    @staticmethod
    def unlock_set():
        return [
            PlanetCard("Planet X", 9, "Five of a Kind"),
            PlanetCard("Ceres", 10, "Flush House"),
            PlanetCard("Eris", 11, "Flush Five"),
        ]

    @staticmethod
    def all():
        return PlanetCard.base_set() + PlanetCard.unlock_set()

    @staticmethod
    def random(unlocked=[]):
        planets = PlanetCard.base_set()

        # Not currently supported, need to be unlocked by playing an impossible hand
        extra_planets = PlanetCard.unlock_set()

        for planet in extra_planets:
            if planet.name in unlocked:
                planets.append(planet)

        return choice(planets)
