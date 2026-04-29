from ray.rllib.env import MultiAgentEnv
from gym_envs.envs.blind_env import BlindEnv
from gym_envs.envs.shop_env import ShopEnv
from copy import deepcopy
from collections import defaultdict
import math
from balatro_connection import BalatroConnection
from gym_envs.integrations.balatro_stepper import BalatroStepper
from gym_envs.blind import Blind
import json
import random
import numpy as np
import os
import uuid

class BlindShopEnv(MultiAgentEnv):
    def __init__(
        self,
        env_config={},
    ):
        self.blind_env = BlindEnv(env_config.get("blind_env_config", {}))
        
        shop_cfg = env_config.get("shop_env_config", {})
        shop_cfg["hand_to_id"] = self.blind_env.hand_to_id
        self.shop_env = ShopEnv(shop_cfg)
        
        self.stake = env_config.get("stake", 0)
        self.blind_env.stake = self.stake
        self.shop_env.stake = self.stake

        self.cash_gained_reward = env_config.get("cash_gained_reward", 0.0)
        self.round_won_reward = env_config.get("round_won_reward", 0.0)
        self.start_phase = env_config.get("start_phase", "shop")
        self.reward_norm = env_config.get("reward_norm", False)
        self.catchup_probability = env_config.get("catchup_probability", 0.0)
        self.catchup_round_range = env_config.get("catchup_round_range", (5, 20))
        self.run_win_reward = env_config.get("run_win_reward", 0.0)
        self.shop_enabled = env_config.get(
            "shop_starts_enabled", True
        )  # shop starts disabled until the blind agent warms up
        self.cycle_blind_agents = env_config.get("cycle_blind_agents", True)
        self.truncate_blind_agents = env_config.get("truncate_blind_agents", False)
        self.connect_to_balatro = env_config.get("connect_to_balatro", False)
        self.balatro_connection = None
        self.blind_steps = ["select_cards_from_hand"]
        self.shop_steps = ["select_shop_action", "select_booster_action"]

        self.reward_stats = defaultdict(lambda: [0.0, 1.0, 0.0])
        self.alpha = 0.003
        self.eps = 1e-8

        assert self.start_phase in (
            "blind",
            "shop",
        ), f"start_phase must be 'blind' or 'shop' {self.start_phase}"
        self._phase = None
        self.held_blind_result = None
        self.observation_space = {
            "blind_agent": self.blind_env.observation_space,
            "shop_agent": self.shop_env.observation_space,
        }
        self.action_space = {
            "blind_agent": self.blind_env.action_space,
            "shop_agent": self.shop_env.action_space,
        }

        self.blind_agent_i = 0
        self.steps_in_phase = 0
        self.total_steps = 0
        
        # self.data_dir = self.data_dir = "/root/autodl-tmp/evaluator_data_3"
        # if not os.path.exists(self.data_dir):
        #    try:
        #        os.makedirs(self.data_dir, exist_ok=True)
        #    except:
        #        pass
            
        # self.current_ep_buffer = []
        # self.worker_global_buffer = []
        # self.flush_threshold = 100

    def blind_agent_id(self):
        return f"blind_agent-{self.blind_agent_i}"

    def reset(self, seed=None, options=None):
        self.current_ep_buffer = []
        self._phase = self.start_phase
        self.steps_in_phase = 0
        self.total_steps = 0
        self.accumulated_blind_reward = 0.0
        catchup_round = 1
        if random.random() < self.catchup_probability:
            catchup_round = random.randint(*self.catchup_round_range)
        options = {} if catchup_round == 1 else {"catchup_round": catchup_round}
        blind_obs = self.blind_env.reset(options=options)[0]
        self.shop_env.reset()
        self.data_from_blind_to_shop()
        shop_obs = self.shop_env.get_obs()
        self.blind_env.G.dollars = self.shop_env.G.dollars
        if self.connect_to_balatro:
            if self.balatro_connection is None:
                self.balatro_connection = BalatroConnection(bot_port=12345)
                agency = self.blind_steps.copy()
                if self.shop_enabled:
                    agency += self.shop_steps
                self.balatro_stepper = BalatroStepper(self.balatro_connection, agency)
                self.balatro_connection.send_cmd("MENU")
                G = self.balatro_stepper.get_gamestate(pending_action="MENU")
                self.blind_env.load_gamestate(G)
                blind_obs = self.blind_env.get_obs()

        if self._phase == "blind":
            self.held_shop_info = {}
            return {self.blind_agent_id(): blind_obs}, {self.blind_agent_id(): {}}
        else:
            self.held_blind_reward = 0
            self.held_blind_info = {}
            return {"shop_agent": shop_obs}, {"shop_agent": {}}

    def results_to_dicts(self, results, end_all=False):
        obs = {}
        rewards = {}
        dones = {}
        truncs = {}
        infos = {}

        for aid, obs_, reward, done, trunc, info in results:
            obs[aid] = obs_
            rewards[aid] = reward
            dones[aid] = done
            truncs[aid] = trunc
            infos[aid] = info

        dones["__all__"] = end_all
        truncs["__all__"] = False

        return obs, rewards, dones, truncs, infos

    def remote_step(self, action_dict):
        if self.blind_agent_id() in action_dict:
            action = self.blind_env.action_vector_to_action(
                action_dict[self.blind_agent_id()]
            )
            print("Sending blind action:", action)
            self.balatro_connection.send_action(action)
            G = self.balatro_stepper.get_gamestate(pending_action=action[0].value)
        elif "shop_agent" in action_dict:
            action = self.shop_env.action_vector_to_action(action_dict["shop_agent"])
            print("Sending shop action:", action)
            response = self.balatro_connection.send_action(action)
            pending_action = action[0].value
            if response and "Error" in response.get("response", {}):
                print(
                    "Error while trying to perform action, ignoring pending action and going again"
                )
                pending_action = None
            G = self.balatro_stepper.get_gamestate(pending_action=pending_action)

        if G["waitingFor"] in self.blind_steps:
            self.blind_env.load_gamestate(G)
            # No reward for real env, only for demoing rn
            blind_result = (
                self.blind_agent_id(),
                self.blind_env.get_obs(),
                0,
                False,
                False,
                {},
            )
            return self.results_to_dicts([blind_result], end_all=False)
        elif G["waitingFor"] in self.shop_steps:
            self.shop_env.load_gamestate(G)
            # No reward for real env, only for demoing rn
            shop_result = (
                "shop_agent",
                self.shop_env.get_obs(),
                0,
                False,
                False,
                {},
            )
            return self.results_to_dicts([shop_result], end_all=False)
        else:
            raise ValueError(f"Unexpected waitingFor state: {G['waitingFor']}")

    # obs, reward, done, truncated, info
    def step(self, action_dict):
        if self.connect_to_balatro:
            return self.remote_step(action_dict)

        if len(action_dict) == 0:
            raise ValueError("Action dict is empty. Please provide actions for agents.")
        self.steps_in_phase += 1

        if self.steps_in_phase > 1000:
            raise ValueError(
                f"Too many steps in phase: {self._phase}. This likely means the environment is not stepping correctly."
            )
        self.total_steps += 1
        if self.total_steps > 2000:
            raise ValueError(
                "Too many total steps. This likely means the environment is not resetting correctly."
            )
        
        # if self._phase == "shop" and "shop_agent" in action_dict:
        #    raw_shop_obs = self.shop_env.get_obs()
        #    self.current_ep_buffer.append(self._capture_snapshot(raw_shop_obs))
            
        # if shop phase
        
        if self._phase == "shop":
            # step shop environment
            shop_obs, shop_reward, shop_done, shop_trunc, shop_info = (
                self.shop_env.step(action_dict["shop_agent"])
            )
            # If not done, return shop observation, etc
            if not shop_done:
                return (
                    {"shop_agent": shop_obs},
                    {"shop_agent": shop_reward},
                    {"shop_agent": False, "__all__": False},
                    {"shop_agent": False, "__all__": False},
                    {"shop_agent": shop_info},
                )
            # If done, return held blind observation
            else:
                self.held_shop_info = shop_info
                blind_obs, blind_reward, blind_info = self.ended_shop()
                return (
                    {self.blind_agent_id(): blind_obs},
                    {self.blind_agent_id(): blind_reward},
                    {self.blind_agent_id(): False, "__all__": False},
                    {self.blind_agent_id(): False, "__all__": False},
                    {self.blind_agent_id(): blind_info},
                )
        # if blind phase
        elif self._phase == "blind":
            # step blind environment
            blind_obs, blind_reward, blind_done, blind_trunc, blind_info = (
                self.blind_env.step(action_dict[self.blind_agent_id()])
            )
            self.accumulated_blind_reward += blind_reward
            # If they won the round, set phase to shop and return shop observation with win reward for the shop
            if blind_info.get("won_round", False):
                self.held_blind_reward = blind_reward
                self.held_blind_info = blind_info
                shop_obs, shop_reward, shop_info = self.won_blind()
                
                shop_reward += self.accumulated_blind_reward
                
                self.accumulated_blind_reward = 0.0

                won_run = self.shop_env.round == 25
                
                result = [("shop_agent", shop_obs, shop_reward, won_run, False, shop_info)]
                
                old_blind_result = (
                    self.blind_agent_id(),
                    blind_obs,
                    blind_reward,
                    blind_done or won_run,
                    blind_trunc,
                    blind_info,
                )

                if self.cycle_blind_agents:
                    if self.truncate_blind_agents and not won_run:
                        # We have to partially reset the round so that the value function can
                        # Bootstrap a reasonable value for the next agent
                        self.blind_env.fresh_blind(fake_reset=True)
                        old_blind_result = (
                            old_blind_result[0],
                            self.blind_env.get_obs(),
                            old_blind_result[2],
                            True,
                            True,
                            old_blind_result[5],
                        )

                    result.append(old_blind_result)
                    
                # if won_run:
                #     self._on_episode_end(round=24)
                    
                return self.results_to_dicts(result, end_all=won_run)
            # If not done, return blind observation, etc
            elif not blind_done:
                return (
                    {self.blind_agent_id(): blind_obs},
                    {self.blind_agent_id(): blind_reward},
                    {self.blind_agent_id(): False, "__all__": False},
                    {self.blind_agent_id(): False, "__all__": False},
                    {self.blind_agent_id(): blind_info},
                )

            # If done and not a won round, then they lost and it's game over. End episode and return done for __all__
            else:
                # self._on_episode_end(round=self.blind_env.round-1)
                shop_reward = self.accumulated_blind_reward
                self.accumulated_blind_reward = 0.0
                self.data_from_blind_to_shop()
                return self.results_to_dicts([
                    (self.blind_agent_id(), blind_obs, blind_reward, True, False, blind_info),
                    ("shop_agent", self.shop_env.get_obs(), shop_reward, True, False, {})
                ], end_all=True)
    def _capture_snapshot(self, obs):
        # lvls: [12, 4] -> Level, Count, Chips, Mult
        lvls = np.stack([
            obs["hand_stats"]["level"],
            obs["hand_stats"]["played_count"],
            obs["hand_stats"]["chips"],
            obs["hand_stats"]["mult"]
        ], axis=-1)

        return {
            "money": obs["dollars"].copy(),
            "reroll": obs["reroll_price"].copy(),
            "goal": obs["goal"].copy(),
            "blind_idx": obs["blind_index"].copy(),
            "lvls": lvls,
            "j_idx": obs["owned_jokers"]["indices"].copy(),
            "j_val": obs["owned_jokers"]["scalar_properties"].copy(),
            "j_edition": obs["owned_jokers"]["edition"].copy()
        }
    
    def _on_episode_end(self, round):
        if not self.current_ep_buffer:
            return

        label = round
        self.worker_global_buffer.append({
            "samples": [random.choice(self.current_ep_buffer)],
            "label": label
        })
        self.current_ep_buffer = []

        if len(self.worker_global_buffer) >= self.flush_threshold:
            self._flush_data_to_disk()
            
    def _flush_data_to_disk(self):
        file_id = uuid.uuid4().hex
        filename = os.path.join(self.data_dir, f"batch_{file_id}.npz")
        
        all_samples = []
        all_labels = []
        for ep in self.worker_global_buffer:
            for s in ep["samples"]:
                all_samples.append(s)
                all_labels.append(ep["label"])
        
        save_dict = {
            "money": np.array([s["money"] for s in all_samples], dtype=np.float32),
            "reroll": np.array([s["reroll"] for s in all_samples], dtype=np.float32),
            "goal": np.array([s["goal"] for s in all_samples], dtype=np.float32),
            "blind_idx": np.array([s["blind_idx"] for s in all_samples], dtype=np.int32),
            "lvls": np.array([s["lvls"] for s in all_samples], dtype=np.float32),
            "j_idx": np.array([s["j_idx"] for s in all_samples], dtype=np.int32),
            "j_val": np.array([s["j_val"] for s in all_samples], dtype=np.float32),
            "j_edition": np.array([s["j_edition"] for s in all_samples], dtype=np.int32),
            "label": np.array(all_labels, dtype=np.float32)
        }
        
        try:
            np.savez_compressed(filename, **save_dict)
            self.worker_global_buffer = []
        except Exception as e:
            print(f"FAILED TO WRITE DATA: {e}")

    def data_from_shop_to_blind(self):
        self.blind_env.G = self.shop_env.G

    def data_from_blind_to_shop(self):
        self.shop_env.G = self.blind_env.G

    def won_blind(self):
        self._phase = "shop"
        self.steps_in_phase = 0
        mid_blind_income = self.blind_env.G.dollars - self.shop_env.G.dollars

        self.data_from_blind_to_shop()

        # Interest earned from remaining cash after blind
        interest = int(self.blind_env.G.dollars / 5)
        interest = max(min(interest, 5), 0)  # Cap interest at 5
        if any(joker.name == "To the Moon" for joker in self.blind_env.G.owned_jokers):
            interest *= 2

        cash_gained = interest + 3 + self.blind_env.hands_left
        self.shop_env.G.dollars += cash_gained

        self.shop_env.round += 1
        self.shop_env.next_blind = Blind.random(round=self.shop_env.round)
        self.shop_env.new_shop()
        
        reward = 0.0
        
        won_run = self.shop_env.round == 25
        
        if(won_run):
            reward += 2.0
        
        return self.shop_env.get_obs(), reward, self.held_shop_info

    def ended_shop(self):
        self._phase = "blind"
        self.steps_in_phase = 0
        self.data_from_shop_to_blind()
        self.blind_env.fresh_blind(external_blind=self.shop_env.next_blind)
        if self.cycle_blind_agents:
            # if not self.truncate_blind_agents:
            self.blind_agent_i += 1
            return self.blind_env.get_obs(), 0, {}
        return self.blind_env.get_obs(), self.held_blind_reward, self.held_blind_info
