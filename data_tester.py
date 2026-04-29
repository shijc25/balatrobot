import torch
import numpy as np
import os
from tqdm import tqdm
# 导入你之前的模型定义
from data_train import BalatroEvaluator

def run_blind_test(model_path, test_data_dir):
    device = torch.device("cuda")
    
    # 1. 加载模型
    model = BalatroEvaluator().to(device)
    # 使用 weights_only=True 是安全实践
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    files = [f for f in os.listdir(test_data_dir) if f.endswith('.npz')]
    print(f">>> 启动盲测：正在读取 {len(files)} 个全新数据包...")

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for f in tqdm(files):
            d = np.load(os.path.join(test_data_dir, f))
            # 这里的转换逻辑必须和训练时完全一致
            batch = {
                "money": torch.tensor(d["money"]).to(device),
                "goal": torch.tensor(d["goal"]).to(device),
                "lvls": torch.tensor(d["lvls"]).to(device),
                "j_idx": torch.tensor(d["j_idx"]).to(device),
                "j_val": torch.tensor(d["j_val"]).to(device),
                "j_edition": torch.tensor(d["j_edition"]).to(device)
            }
            # 这里的 label 是 [1] 维，需要广播到和快照数量一致
            # 如果你的 npz 里的 label 是 (N,) 那就直接取
            labels = d["label"] 
            if labels.shape[0] == 1:
                labels = np.repeat(labels, d["money"].shape[0])

            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(labels)

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    # 3. 核心指标分析
    acc = ((probs > 0.5).astype(float) == labels).mean()
    
    # 计算预测高分（Top 5%）的真实通关率
    # 这是一个 Evaluator 是否合格的终极指标
    top_5_idx = np.argsort(probs)[-int(len(probs)*0.05):]
    top_5_real_wr = labels[top_5_idx].mean()
    
    avg_real_wr = labels.mean()

    print("\n" + "="*40)
    print("      EVALUATOR 真实战力报告")
    print("-"*40)
    print(f"测试样本总数: {len(labels)}")
    print(f"全局准确率:   {acc:.2%}")
    print(f"数据总平均胜率: {avg_real_wr:.2%}")
    print(f"模型看好(Top 5%)的真实胜率: {top_5_real_wr:.2%}")
    
    # 计算“伯乐系数” (Lift): 模型看好的比平均水平高多少
    lift = top_5_real_wr / avg_real_wr if avg_real_wr > 0 else 0
    print(f"伯乐系数 (Lift): {lift:.2f}x")
    print("="*40)

if __name__ == "__main__":
    run_blind_test("evaluator_acc_0.90.pth", "/root/autodl-tmp/evaluator_data_1")