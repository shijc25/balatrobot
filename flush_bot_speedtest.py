from bot import Bot, Actions
from gamestates import cache_state
import time

class FlushBot(Bot):
    def skip_or_select_blind(self, G):
        global tot
        tot += 1
        return [Actions.SELECT_BLIND]

    def select_cards_from_hand(self, G):
        suit_count = {
            "Hearts": 0,
            "Diamonds": 0,
            "Clubs": 0,
            "Spades": 0,
        }
        for card in G["hand"]:
            suit_count[card["suit"]] += 1

        most_common_suit = max(suit_count, key=suit_count.get)
        most_common_suit_count = suit_count[most_common_suit]
        if most_common_suit_count >= 5:
            flush_cards = []
            for card in G["hand"]:
                if card["suit"] == most_common_suit:
                    flush_cards.append(card)
            flush_cards.sort(key=lambda x: x["value"], reverse=True)
            return [
                Actions.PLAY_HAND,
                [G["hand"].index(card) + 1 for card in flush_cards[:5]],
            ]

        discards = []
        for card in G["hand"]:
            if card["suit"] != most_common_suit:
                discards.append(card)
        discards.sort(key=lambda x: x["value"], reverse=True)
        discards = discards[:5]
        if len(discards) > 0:
            if G["current_round"]["discards_left"] > 0:
                action = Actions.DISCARD_HAND
            else:
                action = Actions.PLAY_HAND
            return [action, [G["hand"].index(card) + 1 for card in discards]]
        
        return [Actions.PLAY_HAND, [1]]

    def select_shop_action(self, G):
        return [Actions.END_SHOP]

    def select_booster_action(self, G):
        return [Actions.SKIP_BOOSTER_PACK]

    def sell_jokers(self, G):
        if len(G["jokers"]) > 1:
            return [Actions.SELL_JOKER, [2]]

        return [Actions.SELL_JOKER, []]

    def rearrange_jokers(self, G):
        return [Actions.REARRANGE_JOKERS, []]

    def use_or_sell_consumables(self, G):
        return [Actions.USE_CONSUMABLE, []]

    def rearrange_consumables(self, G):
        return [Actions.REARRANGE_CONSUMABLES, []]

    def rearrange_hand(self, G):
        return [Actions.REARRANGE_HAND, []]

def run():
    bot_counts = [1]
    for bot_count in bot_counts:
        first_time = None
        target = bot_count * 100
        bots = []
        for i in range(bot_count):
            mybot = FlushBot(
                deck="Blue Deck",
                stake=1,
                seed=None,
                challenge=None,
                bot_port=12348 + i,
            )
            bots.append(mybot)

        try:
            for bot in bots:
                bot.start_balatro_instance()
            start_time = time.time()
            global tot
            tot = 0
            while tot < target:
                for bot in bots:
                    bot.run_step()
            end_time = time.time()

            blind_per_sec = target / (end_time - start_time)
            print(f"Bot count: {bot_count}, blind/sec: {blind_per_sec}")
        finally:
            # Stop the bots
            for bot in bots:
                bot.stop_balatro_instance()


if __name__ == "__main__":
    global first_time
    first_time = None
    run()