from copy import deepcopy
from gymnasium import spaces as sp
import numpy as np
from random import randint, choice, random, choices
import torch
from gym_envs.base_card import BaseCard
from gym_envs.components.card import Card
from gym_envs.joker import Joker
from gym_envs.components.planet import PlanetCard
from gym_envs.components.tarot import TarotCard
from gym_envs.components.spectral import SpectralCard


class Booster(BaseCard):
    SIZES = ["Normal", "Jumbo", "Mega"]
    NAMES = ["Buffoon", "Celestial", "Arcana", "Spectral", "Standard"]

    def __init__(self, name, size, weight):
        super().__init__(segment=BaseCard.Segments.SHOP_BUYABLE)
        self.name = name
        self.size = size
        self.weight = weight
        self.num_cards = 2
        self.is_open = False
        if size in ["Jumbo", "Mega"]:
            self.num_cards = 4
        if name in ["Celestial", "Arcana", "Standard"]:
            self.num_cards += 1

        self.value = 4
        if size == "Jumbo":
            self.value = 6
        if size == "Mega":
            self.value = 8

        self.num_choices = 1
        if size == "Mega":
            self.num_choices = 2

        self.has_target_hand = False
        if name in ["Arcana", "Spectral"]:
            self.has_target_hand = True

        self.cards = []

    @staticmethod
    def from_gamestate_card(gamestate_card):
        label = gamestate_card.get("label", None)
        if label is None:
            return None
        size = "Normal"
        for s in Booster.SIZES:
            if s in label:
                size = s
                break
        name = None
        for n in Booster.NAMES:
            if n in label:
                name = n
                break
        if name is None or size is None:
            return None
        weight = 1  # default weight, doesn't matter since we're loading from a gamestate, not rolling

        booster = Booster(name, size, weight)

        cost = gamestate_card.get("cost", None)
        if cost is not None:
            booster.value = cost
        return booster

    def full_name(self):
        return f"{self.name} {self.size} Booster"

    def get_universal_index(self):
        i = BaseCard.FIRST_BOOSTER_INDEX
        i += ["Buffoon", "Celestial", "Arcana", "Spectral", "Standard"].index(
            self.name
        ) * 3
        # i += ["Normal", "Jumbo", "Mega"].index(self.size)
        return i

    def open(self, owned, unlocked, ignore_rarity=False, stake=0):
        if len(self.cards) > 0:
            return self.cards

        while len(self.cards) < self.num_cards:
            if self.name == "Buffoon":
                no_dupes = True
                card_gen = lambda: Joker.random(
                    unlocked_jokers=unlocked, ignore_rarity=ignore_rarity, stake=stake
                )
            elif self.name == "Celestial":
                no_dupes = True
                card_gen = lambda: (
                    PlanetCard.random(unlocked=unlocked)
                    if random() > 0.003
                    else SpectralCard.from_name("Black Hole")
                )
            elif self.name == "Standard":
                no_dupes = False
                card_gen = lambda: Card.random()
            elif self.name == "Arcana":
                no_dupes = True
                card_gen = lambda: (
                    TarotCard.random()
                    if random() > 0.003
                    else SpectralCard.from_name("The Soul")
                )
            elif self.name == "Spectral":
                no_dupes = True
                card_gen = lambda: (
                    SpectralCard.random()
                    if random() > 0.006
                    else SpectralCard.from_name(choice(["The Soul", "Black Hole"]))
                )
            else:
                raise NotImplementedError(
                    f"Booster type {self.name} not implemented for opening."
                )
            card = card_gen()
            if no_dupes and any(card.is_dupe(other) for other in self.cards + owned):
                continue
            card.segment = BaseCard.Segments.BOOSTER_CHOICE
            self.cards.append(card)
        return self.cards

    @staticmethod
    def random():
        booster = choices(
            boosters,
            weights=[b.weight for b in boosters],
            k=1,
        )[0]
        return deepcopy(booster)

    @staticmethod
    def all():
        return deepcopy(boosters)


boosters = [
    Booster("Buffoon", "Normal", 1.2),
    Booster("Buffoon", "Jumbo", 0.6),
    Booster("Buffoon", "Mega", 0.15),
    Booster("Celestial", "Normal", 4),
    Booster("Celestial", "Jumbo", 2),
    Booster("Celestial", "Mega", 0.5),
    Booster("Standard", "Normal", 4),
    Booster("Standard", "Jumbo", 2),
    Booster("Standard", "Mega", 0.5),
    Booster("Arcana", "Normal", 4),
    Booster("Arcana", "Jumbo", 2),
    Booster("Arcana", "Mega", 0.5),
    Booster("Spectral", "Normal", 0.6),
    Booster("Spectral", "Jumbo", 0.3),
    Booster("Spectral", "Mega", 0.07),
]
