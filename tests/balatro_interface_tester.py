from balatro_connection import BalatroConnection
from gym_envs.real_balatro.balatro_stepper import BalatroStepper
import time
from gym_envs.pseudo.blind_env import PseudoBlindEnv
from gym_envs.pseudo.shop_env import PseudoShopEnv
from bot import Bot, Actions

blind_steps = ["select_cards_from_hand"]
shop_steps = ["select_shop_action", "select_booster_action"]

balatro_connection = BalatroConnection(bot_port=12345)
stepper = BalatroStepper(balatro_connection, agency_states=blind_steps + shop_steps)
blind_env = PseudoBlindEnv()
blind_env.reset()
shop_env = PseudoShopEnv()
shop_env.reset()

while True:
    game_state = stepper.get_gamestate()
    if game_state.get("waitingForAction", False):
        trimmed_game_state = {k: v for k, v in game_state.items() if k not in ["deck"]}
        print(trimmed_game_state)
        if game_state.get("waitingFor", None) in blind_steps:
            blind_env.load_gamestate(game_state)
            print(blind_env.get_obs())
        # if game_state.get("waitingFor", None) == "select_booster_action":
        #     balatro_connection.send_action([Actions.SELECT_BOOSTER_CARD, [1]])
        else:
            shop_env.load_gamestate(game_state)
            print(shop_env.get_obs())
            print([x.name for x in shop_env.jokers])
        time.sleep(1)
