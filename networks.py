import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np



class PolicyNet(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, config["hidden_units"])
        self.fc_mu = nn.Linear(config["hidden_units"],action_dim)
        self.fc_std  = nn.Linear(config["hidden_units"],action_dim)

        self.log_std_min = config["log_std_min"]
        self.log_std_max = config["log_std_max"]

        self.optimizer = optim.Adam(self.parameters(), lr=config["learning_rate"]["policy"])

        init_alpha = config["init_alpha"]
        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=config["learning_rate"]["alpha"])
        
        self.target_entropy = -action_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        # std = F.softplus(self.fc_std(x))
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob
        entropy = entropy.sum(dim=-1, keepdim = True)

        q1_val, q2_val = q1(s,a), q2(s,a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return loss.mean().item(), alpha_loss.item()


class QNet(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(obs_dim ,config["hidden_units"]//2)
        self.fc_a = nn.Linear(action_dim,config["hidden_units"]//2)
        self.fc_cat = nn.Linear(config["hidden_units"],config["hidden_units"]//4)
        self.fc_out = nn.Linear(config["hidden_units"]//4,1)
        self.optimizer = optim.Adam(self.parameters(), lr=config["learning_rate"]["critic"])

        self.tau = config["target_smoothing_coefficient"]
        
    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=-1)
        
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a) , target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        return loss.mean().item()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)



