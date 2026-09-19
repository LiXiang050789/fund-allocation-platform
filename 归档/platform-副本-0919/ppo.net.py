import torch
import torch.nn as nn
import torch.nn.functional as F

class PortfolioActor(nn.Module):
    def __init__(self, obs_dim, act_dim, min_weight=0.02, max_weight=0.4):
        super().__init__()
        self.min_w = min_weight
        self.max_w = max_weight
        # 网络层
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.logit_out = nn.Linear(128, act_dim)

    def forward(self, obs):
        # 1. 特征前向
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.logit_out(x)

        # 2. softmax 转为原始权重分布
        raw_weights = F.softmax(logits, dim=-1)

        # ========== 权重约束代码放在这里 ==========
        # 单资产上下限截断
        clip_w = torch.clamp(raw_weights, self.min_w, self.max_w)
        # 重新归一化保证总和=1
        final_weights = clip_w / torch.sum(clip_w, dim=-1, keepdim=True)
        # ==========================================

        return final_weights

class PortfolioCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.value_out = nn.Linear(128, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        value = self.value_out(x)
        return value
