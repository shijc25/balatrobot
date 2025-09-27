from gym_envs.base_card import BaseCard
from gym_envs.components.card import Card
from gym_envs.joker import Joker
from random import choice, random
from copy import deepcopy


class Consumable(BaseCard):
    def __init__(self, name, num_targets=0, segment=None):
        super().__init__(segment=segment)
        self.name = name
        self.num_targets = num_targets

    def trigger(self, targets, gamestate):
        print(f"Base consumable trigger called for {self.name}")
        return {}
