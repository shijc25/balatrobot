import torch
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from torch.distributions import Categorical

class AutoregressiveCardDist(TorchDistributionWrapper):
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        B = inputs.shape[0]
        self.cached_features = inputs[:, :33*128].view(B, 33, 128)
        self.masks = inputs[:, 33*128:]
        self.model = model
        self._cached_entropy = None
        
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 1 
    
    def _get_torch_distribution(self):
        return self

    def sample(self):
        with torch.no_grad(): 
            B = self.inputs.shape[0]
            device = self.inputs.device
            
            hand_mask = torch.zeros(B, 10, device=device)
            actions = []
            selected_mode = None
            stopped = torch.zeros(B, dtype=torch.bool, device=device)
            
            for i in range(6):
                if i > 0 and stopped.all():
                    for _ in range(i, 6):
                        actions.append(torch.full((B,), 10, device=device, dtype=torch.long))
                    break

                mode_logits, card_logits = self.model.ar_step(
                    self.cached_features, hand_mask, self.masks, step_idx=i, selected_mode=selected_mode
                )
                
                if i == 0:
                    mode_act = Categorical(logits=mode_logits).sample()
                    actions.append(mode_act)
                    selected_mode = mode_act
                    continue
                
                card_act = Categorical(logits=card_logits).sample()
                
                final_act = torch.where(stopped, torch.tensor(10, device=device), card_act)
                actions.append(final_act)
                
                stopped = stopped | (final_act == 10)
                
                is_card = (final_act < 10)
                if is_card.any():
                    batch_idx = torch.where(is_card & (~stopped))[0]
                    if len(batch_idx) > 0:
                        selected_card = final_act[batch_idx]
                        hand_mask[batch_idx, selected_card] = 1.0

            self.last_sample = torch.stack(actions, dim=1)
            return self.last_sample
    
    def deterministic_sample(self):
        with torch.no_grad(): 
            B = self.inputs.shape[0]
            device = self.inputs.device
            
            hand_mask = torch.zeros(B, 10, device=device)
            actions = []
            selected_mode = None
            stopped = torch.zeros(B, dtype=torch.bool, device=device)
            
            for i in range(6):
                if i > 0 and stopped.all():
                    for _ in range(i, 6):
                        actions.append(torch.full((B,), 10, device=device, dtype=torch.long))
                    break

                mode_logits, card_logits = self.model.ar_step(
                    self.cached_features, hand_mask, self.masks, step_idx=i, selected_mode=selected_mode
                )
                
                if i == 0:
                    mode_act = torch.argmax(mode_logits, dim=-1)
                    actions.append(mode_act)
                    selected_mode = mode_act
                    continue
                
                card_act = torch.argmax(card_logits, dim=-1)
                final_act = torch.where(stopped, torch.tensor(10, device=device), card_act)
                actions.append(final_act)
                
                stopped = stopped | (final_act == 10)
                is_card = (final_act < 10)
                if is_card.any():
                    batch_idx = torch.where(is_card & (~stopped))[0]
                    if len(batch_idx) > 0:
                        selected_card = final_act[batch_idx]
                        hand_mask[batch_idx, selected_card] = 1.0

            self.last_sample = torch.stack(actions, dim=1)
            return self.last_sample

    def logp(self, actions):
        B = actions.shape[0]
        device = actions.device
        hand_mask = torch.zeros(B, 10, device=device)
        
        total_logp = torch.zeros(B, device=device)
        total_entropy = torch.zeros(B, device=device)
        stopped = torch.zeros(B, dtype=torch.bool, device=device)
        
        true_mode = actions[:, 0]

        for i in range(6):
            if i > 0 and stopped.all(): break
                
            current_mode = None if i == 0 else true_mode
            mode_logits, card_logits = self.model.ar_step(
                self.cached_features, hand_mask, self.masks, step_idx=i, selected_mode=current_mode
            )
            
            if i == 0:
                mode_dist = Categorical(logits=mode_logits)
                total_logp = total_logp + mode_dist.log_prob(actions[:, 0])
                total_entropy = total_entropy + mode_dist.entropy()
                continue
                
            card_dist = Categorical(logits=card_logits)
            card_act = actions[:, i]
            
            logp_val = card_dist.log_prob(card_act)
            total_logp = total_logp + torch.where(stopped, torch.zeros_like(logp_val), logp_val)
            
            ent = card_dist.entropy()
            total_entropy = total_entropy + torch.where(stopped, torch.zeros_like(ent), ent)
            
            is_card = (card_act < 10) & (~stopped)
            if is_card.any():
                hand_mask = hand_mask.scatter(1, card_act.unsqueeze(-1).long().clamp(0, 9), is_card.float().unsqueeze(-1))
            
            stopped = stopped | (card_act == 10)
        
        self._cached_entropy = total_entropy
        return total_logp
    
    def kl(self, other):
        return torch.zeros(self.inputs.shape[0], device=self.inputs.device)

    def entropy(self):
        if self._cached_entropy is not None:
            return self._cached_entropy
        return torch.zeros(self.inputs.shape[0], device=self.inputs.device)
    
class ShopActionDist(TorchCategorical):
    def __init__(self, inputs, model):
        super().__init__(inputs, model)