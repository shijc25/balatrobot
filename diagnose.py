import os
import json
import torch
import numpy as np
from pathlib import Path
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from modeling.generic_blind_model import BalatroBlindModel
from modeling.generic_shop_model import BalatroShopModel
from modeling.distributions import AutoregressiveCardDist
from modeling.distributions import ShopActionDist

from gym_envs.envs.blind_shop_env import BlindShopEnv
from gym_envs.blind import Blind
from gym_envs.joker import Joker
from gym_envs.components.planet import PlanetCard

def register_custom_assets():
    ModelCatalog.register_custom_model("generic_blind_model", BalatroBlindModel)
    ModelCatalog.register_custom_model("generic_shop_model", BalatroShopModel)
    ModelCatalog.register_custom_action_dist("ar_custom_dist", AutoregressiveCardDist)
    ModelCatalog.register_custom_action_dist("shop_custom_dist", ShopActionDist)

# ==========================================
# 1. 配置参数 (请填入你的真实路径)
# ==========================================
CHECKPOINT_PATH = r"/root/autodl-tmp/run_data/blind_shop/blind_shop_9f560_00000_0_2026-04-27_13-33-29/checkpoint_000086/"
NUM_EPISODES = 10

def format_cards(cards):
    """将卡牌对象转化为人类可读的字符串列表"""
    res = []
    for c in cards:
        if isinstance(c, Joker):
            # 打印小丑名字和当前成长值
            chips = c.state.get("chips", 0) 
            mult = c.state.get("mult", 0)
            xmult = c.state.get("mult_mult", 1.0)
            stat_str = []
            if chips != 0: stat_str.append(f"+{chips}Chips")
            if mult != 0: stat_str.append(f"+{mult}Mult")
            if xmult != 1.0: stat_str.append(f"x{xmult:.1f}Mult")
            stats = f"({', '.join(stat_str)})" if stat_str else ""
            res.append(f"[{c.name}{stats}]")
        elif isinstance(c, PlanetCard):
            # 显示星球名字及其对应的牌型，例如: [木星 (Flush)]
            res.append(f"[星球: {c.name} ({c.hand_type})]")
        elif hasattr(c, 'suit') and hasattr(c, 'value') and c.value is not None:
            suit_icon = {"Hearts":"红桃", "Diamonds":"方片", "Clubs":"草花", "Spades":"黑桃"}.get(c.suit, c.suit)
            rank_name = {11:"J", 12:"Q", 13:"K", 14:"A"}.get(c.value, str(c.value))
            res.append(f"{suit_icon}{rank_name}")
        else:
            res.append("[空]")
    return res

def format_hand_levels(hand_stats):
    """
    格式化显示已经升级过的牌型等级
    """
    upgraded = []
    # 核心牌型列表
    core_hands = [
        "High Card", "Pair", "Two Pair", "Three of a Kind", 
        "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"
    ]
    
    for name in core_hands:
        h_obj = hand_stats.get(name)
        if h_obj and h_obj.level > 1:
            upgraded.append(f"{name}:Lv.{h_obj.level}")
    
    if not upgraded:
        return "[全1级]"
    return " | ".join(upgraded)

def evaluate_agent():
    print(f"正在加载 Checkpoint: {CHECKPOINT_PATH}")
    # 这里直接加载算法，RLlib 会自动用你当时的 config
    algo = PPO.from_checkpoint(CHECKPOINT_PATH)
    
    # 提取我们需要的策略环境配置 (假设你用的是默认的)
    env_config = algo.config.env_config
    
    print("\n" + "="*50)
    print(f"开始执行 {NUM_EPISODES} 局诊断测试 (Explore=False)")
    print("="*50 + "\n")

    for ep_idx in range(NUM_EPISODES):
        # 我们直接使用底层环境进行单步模拟，不走 Ray 的并发架构，方便打印
        env = BlindShopEnv(env_config)
        
        # RLlib 多智能体环境初始化
        obs_dict, info_dict = env.reset()
        done = {"__all__": False}
        
        print(f"\n>>>>>> [Episode {ep_idx + 1} 开始] >>>>>>")
        
        step_count = 0
        while not done["__all__"]:
            step_count += 1
            action_dict = {}
            
            active_agents = list(obs_dict.keys())
            if not active_agents: break
            
            for agent_id in active_agents:
                obs = obs_dict[agent_id]
                # --- 处理 商店 Agent ---
                if agent_id == "shop_agent":
                    # 获取环境里的真实物理对象用于打印
                    shop_env = env.shop_env
                    print(f"\n[Step {step_count}] ---> 🛒 商店阶段 (Round {shop_env.round})")
                    print(f"  💰 资金: ${shop_env.G.dollars} | 🎲 Reroll: ${shop_env.reroll_cost}")
                    print(f"  📈 牌型等级: {format_hand_levels(shop_env.G.hand_stats)}")
                    print(f"  🎯 下一关: {shop_env.next_blind.name} (目标: {shop_env.next_blind.chip_goal})")
                    print(f"  🎒 拥有小丑: {format_cards(shop_env.G.owned_jokers)}")
                    print(f"  🏪 货架物品: {format_cards(shop_env.shop_jokers)}")
                    print(f"  📦 货架卡包: {[b.name for b in shop_env.boosters]}")
                    if shop_env.in_pack_selection:
                        print(f"  🎁 拆开的包: {format_cards(shop_env.booster_contents)}")

                    # 让模型预测动作 (关闭探索，输出纯概率分布里的 Argmax)
                    action = algo.compute_single_action(
                        obs, 
                        policy_id="shop_agent", 
                        explore=False
                    )
                    action_dict[agent_id] = action
                    
                    # 翻译动作给人看
                    act_meaning = shop_env.action_vector_to_action(action)
                    print(f"  🤖 AI 决定 -> {act_meaning[0].name} (参数: {act_meaning[1:]})")
                # --- 处理 打牌 Agent ---
                elif agent_id.startswith("blind_agent"):
                    blind_env = env.blind_env
                    print(f"\n[Step {step_count}] ---> ⚔️ 战斗阶段 (Round {blind_env.round})")
                    print(f"  🎯 Boss: {blind_env.G.current_blind.name} | 分数: {blind_env.chips} / {blind_env.chip_goal}")
                    print(f"  💰 资金: ${blind_env.G.dollars}")
                    print(f"  📈 牌型等级: {format_hand_levels(blind_env.G.hand_stats)}")
                    print(f"  ✋ 剩余次数: {blind_env.hands_left} 手牌 | {blind_env.discards_left} 弃牌")
                    print(f"  🎒 拥有小丑: {format_cards(blind_env.G.owned_jokers)}")
                    print(f"  🃏 当前手牌: {format_cards(blind_env.G.hand.cards)}")
                    
                    # 你的 AR 模型在这里调用
                    action = algo.compute_single_action(
                        obs, 
                        policy_id="blind_agent", # 注意策略名字
                        explore=False
                    )
                    action_dict[agent_id] = action
                    
                    act_meaning = blind_env.action_vector_to_action(action)
                    # act_meaning 通常是 [Actions.PLAY_HAND, [1, 2, 3]]
                    act_type = act_meaning[0].name
                    card_indices = act_meaning[1]
                    # 找出它具体选了哪几张牌
                    chosen_cards = [blind_env.G.hand.cards[i-1] for i in card_indices if i-1 < len(blind_env.G.hand.cards)]
                    print(f"  🤖 AI 决定 -> {act_type}: {format_cards(chosen_cards)}")

            # --- 执行环境 Step ---
            obs_dict, rewards_dict, done, truncs_dict, infos_dict = env.step(action_dict)
            
            if done["__all__"]:
                # 判定输赢
                if env.shop_env.round >= 25:
                    print(f"\n🎉 结果：通关 Ante 8 (Round 24)！")
                else:
                    dead_round = env.shop_env.round if env._phase == "shop" else env.blind_env.round
                    print(f"\n💀 结果：死在了 Round {dead_round}。")

if __name__ == "__main__":
    torch.set_num_threads(1)
    ray.init(local_mode=True, ignore_reinit_error=True)
    register_custom_assets()
    evaluate_agent()
    
# python diagnose.py > diagnose_report.log 2>&1