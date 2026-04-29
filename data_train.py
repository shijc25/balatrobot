import torch
import torch.nn as nn
import numpy as np
import os
import wandb
from tqdm import tqdm
from gym_envs.universal_card_encoder import UniversalCardEncoder

CONFIG = {
    "data_dir_train": "/root/autodl-tmp/evaluator_data_3",
    "token_dim": 64,
    "batch_size": 16384,
    "lr": 1e-4,
    "epochs": 500,
    "device": "cuda"
}

class BalatroEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.card_encoder = UniversalCardEncoder()
        
        self.money_proj = nn.Linear(1, 63)
        self.goal_proj = nn.Linear(1, 63)
        # self.levels_proj = nn.Linear(4, 63)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, 63))
        self.pos_emb = nn.Parameter(torch.randn(1, 7, 63) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=8,
            dim_feedforward=512,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.head = nn.Sequential(
            nn.Linear(64, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 1)
        )

    def forward(self, batch):
        B = batch["money"].shape[0]
        
        m_tok = self.money_proj(batch["money"] / 10.0).unsqueeze(1)
        g_tok = self.goal_proj(batch["goal"] / 10000.0).unsqueeze(1)
        # l_toks = self.levels_proj(batch["lvls"])
        
        j_toks, _ = self.card_encoder({
            "indices": batch["j_idx"],
            "rank": torch.zeros_like(batch["j_idx"]),
            "suit": torch.zeros_like(batch["j_idx"]),
            "enhancement": torch.zeros_like(batch["j_idx"]),
            "edition": batch["j_edition"],
            "seal": torch.zeros_like(batch["j_idx"]),
            "segment": torch.full_like(batch["j_idx"], 3),
            "scalar_properties": batch["j_val"],
            "debuffed": torch.zeros_like(batch["j_idx"])
        })
        
        seq_63 = torch.cat([g_tok, m_tok, j_toks], dim=1)
        x = torch.cat([seq_63 + self.pos_emb, torch.zeros(B, 7, 1, device=batch["money"].device)], dim=2)
        
        features = self.transformer(x)
        
        return self.head(features[:, 0, :]).squeeze(-1)

class CUDAFullDataset:
    def __init__(self, data_dir):
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        raw_data = {k: [] for k in ["money", "goal", "lvls", "j_idx", "j_val", "j_edition", "label"]}
        
        for f in tqdm(files, desc="Loading NPZ files"):
            d = np.load(f)
            num_samples_in_file = d['label'].shape[0]
            pick_count = num_samples_in_file
            indices = np.random.choice(num_samples_in_file, pick_count, replace=False)
            
            for k in raw_data.keys():
                raw_data[k].append(d[k][indices])
        
        base_chips = [300, 800, 2000, 5000, 11000, 20000, 35000, 50000]
        goal_to_round = {}
        r_idx = 1
        for base in base_chips:
            goal_to_round[int(base * 1)] = r_idx       # Small Blind
            goal_to_round[int(base * 1.5)] = r_idx + 1 # Big Blind
            goal_to_round[int(base * 2)] = r_idx + 2   # Boss Blind
            r_idx += 3

        merged_data = {}
        for k in raw_data.keys():
            merged_data[k] = np.concatenate(raw_data[k], axis=0)

        goals = merged_data["goal"]
        final_rounds = merged_data["label"]
        
        valid_indices = []
        new_labels = []
        
        for i, g in enumerate(goals):
            g_int = int(g)
            if g_int in goal_to_round:
                valid_indices.append(i)
                curr_r = goal_to_round[g_int]
                print(curr_r)
                is_win = 1.0 if (final_rounds[i] == 24) else 0.0
                new_labels.append(is_win)
        
        valid_indices = np.array(valid_indices)
        new_labels = np.array(new_labels, dtype=np.float32)

        for k in merged_data.keys():
            merged_data[k] = merged_data[k][valid_indices]
        merged_data["label"] = new_labels
        
        self.tensors = {}
        for k in ["money", "goal", "lvls", "j_idx", "j_val", "j_edition"]:
            self.tensors[k] = torch.from_numpy(merged_data[k][valid_indices]).to(CONFIG["device"])
        
        self.tensors["label"] = torch.from_numpy(new_labels).to(CONFIG["device"])
        self.num_samples = len(new_labels)

        num_pos = np.sum(new_labels)
        num_neg = len(new_labels) - num_pos
        self.pos_weight = torch.tensor([num_neg / num_pos]).to(CONFIG["device"])
        
        print(f"有效样本: {len(new_labels)} | 正负比: 1 : {num_neg/num_pos:.2f}")

def get_batches_from_dict(tensors, batch_size, shuffle=True):
    num_samples = tensors["label"].shape[0]
    if shuffle:
        indices = torch.randperm(num_samples, device=tensors["label"].device)
    else:
        indices = torch.arange(num_samples, device=tensors["label"].device)
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield {k: v[batch_indices] for k, v in tensors.items()}

def train():
    wandb.init(project="balatro-evaluator-bce", config=CONFIG)
    
    full_data = CUDAFullDataset(CONFIG["data_dir_train"])
    num_total = full_data.num_samples
    
    indices = torch.arange(num_total, device=CONFIG["device"])
    
    val_size = int(0.05 * num_total)
    train_size = num_total - val_size
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    dataset_train = {k: v[train_idx] for k, v in full_data.tensors.items()}
    dataset_val = {k: v[val_idx] for k, v in full_data.tensors.items()}
    
    model = BalatroEvaluator().to(CONFIG["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.05)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=full_data.pos_weight)

    best_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0
        pbar = tqdm(get_batches_from_dict(dataset_train, CONFIG["batch_size"]), desc=f"Epoch {epoch}")
        for batch in pbar:
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch["label"]) 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_logits = model(dataset_val)
            val_loss = criterion(val_logits, dataset_val["label"]).item()
            
            probs = torch.sigmoid(val_logits)
            preds_class = (probs > 0.5).float()
            correct = (preds_class == dataset_val["label"]).float()
            accuracy = correct.mean().item()
            
            pos_mask = dataset_val["label"] == 1.0
            neg_mask = dataset_val["label"] == 0.0
            pos_acc = correct[pos_mask].mean().item() if pos_mask.any() else 0
            neg_acc = correct[neg_mask].mean().item() if neg_mask.any() else 0
        
        avg_train_loss = train_loss / (train_size / CONFIG["batch_size"])
        print(f"Epoch {epoch} | Train BCE: {avg_train_loss:.4f} | Val BCE: {val_loss:.4f} | Val Acc: {accuracy:.2%} (Pos: {pos_acc:.0%}, Neg: {neg_acc:.0%})")
        
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_acc": accuracy,
            "pos_acc": pos_acc,
            "neg_acc": neg_acc
        })
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), f"evaluator_acc_{accuracy:.4f}.pth")
    
    wandb.finish()

if __name__ == "__main__":
    train()