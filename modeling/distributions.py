import torch
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from torch.distributions import Categorical

class AutoregressiveCardDist(TorchDistributionWrapper):
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        self.model = model
        self._cached_entropy = None
        self._last_actions = None
        
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
            
            for i in range(6):
                mode_logits, card_logits = self.model.ar_step(hand_mask, step_idx=i, selected_mode=selected_mode)
                
                if i == 0:
                    mode_dist = Categorical(logits=mode_logits)
                    mode_act = mode_dist.sample()
                    actions.append(mode_act)
                    selected_mode = mode_act
                    continue
                    
                card_dist = Categorical(logits=card_logits)
                card_act = card_dist.sample()
                actions.append(card_act)
                
                is_card = card_act < 10
                if is_card.any():
                    batch_idx = torch.where(is_card)[0]
                    selected_card = card_act[is_card]
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
            
            for i in range(6):
                mode_logits, card_logits = self.model.ar_step(hand_mask, step_idx=i, selected_mode=selected_mode)
                
                if i == 0:
                    mode_act = torch.argmax(mode_logits, dim=-1)
                    actions.append(mode_act)
                    selected_mode = mode_act
                    continue
                
                card_act = torch.argmax(card_logits, dim=-1)
                actions.append(card_act)
                
                is_card = card_act < 10
                if is_card.any():
                    batch_idx = torch.where(is_card)[0]
                    selected_card = card_act[is_card]
                    hand_mask[batch_idx, selected_card] = 1.0

            self.last_sample = torch.stack(actions, dim=1)
            return self.last_sample

    def logp(self, actions):
        self._last_actions = actions
        
        B = actions.shape[0]
        device = actions.device
        hand_mask = torch.zeros(B, 10, device=device)
        
        total_logp = torch.zeros(B, device=device)
        total_entropy = torch.zeros(B, device=device)
        stopped = torch.zeros(B, dtype=torch.bool, device=device)
        
        true_mode = actions[:, 0]

        for i in range(6):
            if stopped.all() and i > 0: break
                
            current_mode = None if i == 0 else true_mode
            mode_logits, card_logits = self.model.ar_step(hand_mask, step_idx=i, selected_mode=current_mode)
            
            if i == 0:
                mode_dist = Categorical(logits=mode_logits)
                total_logp += mode_dist.log_prob(actions[:, 0])
                total_entropy += mode_dist.entropy()
                continue
                
            card_dist = Categorical(logits=card_logits)
            card_act = actions[:, i]
            
            logp = card_dist.log_prob(card_act)
            total_logp += torch.where(stopped, torch.zeros_like(logp), logp)
            
            ent = card_dist.entropy()
            total_entropy += torch.where(stopped, torch.zeros_like(ent), ent)
            
            is_stop = card_act == 10
            stopped = stopped | is_stop
            
            is_card = (card_act < 10) & (~stopped)
            if is_card.any():
                batch_idx = torch.where(is_card)[0]
                selected_card = card_act[is_card]
                hand_mask[batch_idx, selected_card] = 1.0
        
        self._cached_entropy = total_entropy
        return total_logp
    
    def kl(self, other):
        actions = other._last_actions
        if actions is None:
            return torch.tensor(0.0, device=self.inputs.device)

        B = actions.shape[0]
        device = actions.device
        hand_mask = torch.zeros(B, 10, device=device)
        total_kl = torch.zeros(B, device=device)
        stopped = torch.zeros(B, dtype=torch.bool, device=device)
        
        true_mode = actions[:, 0]

        for i in range(6):
            if stopped.all() and i > 0: break
            
            current_mode = None if i == 0 else true_mode
            mode_logits_old, card_logits_old = self.model.ar_step(hand_mask, step_idx=i, selected_mode=current_mode)
            mode_logits_new, card_logits_new = other.model.ar_step(hand_mask, step_idx=i, selected_mode=current_mode)
            
            if i == 0:
                d_old = Categorical(logits=mode_logits_old)
                d_new = Categorical(logits=mode_logits_new)
                total_kl += torch.distributions.kl.kl_divergence(d_old, d_new)
                continue
                
            d_old = Categorical(logits=card_logits_old)
            d_new = Categorical(logits=card_logits_new)
            
            step_kl = torch.distributions.kl.kl_divergence(d_old, d_new)
            total_kl += torch.where(stopped, torch.zeros_like(step_kl), step_kl)
            
            card_act = actions[:, i]
            is_stop = card_act == 10
            stopped = stopped | is_stop
            is_card = (card_act < 10) & (~stopped)
            if is_card.any():
                batch_idx = torch.where(is_card)[0]
                selected_card = card_act[is_card]
                hand_mask[batch_idx, selected_card] = 1.0
        
        return total_kl

    def entropy(self):
        if self._cached_entropy is not None:
            return self._cached_entropy
        return torch.zeros(self.inputs.shape[0], device=self.inputs.device)
    
class ShopActionDist(TorchCategorical):
    def __init__(self, inputs, model):
        super().__init__(inputs, model)