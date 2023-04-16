import torch
import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0 
        self.state_memory = np.zeros((self.mem_size, input_shape))
        
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
        
    def put(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 0.0 if done else 1.0
        
        self.mem_cntr += 1
        
    
    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        
        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, dones

    @staticmethod
    def batch_to_device(mini_batch, device):
        s, a, r, s_prime, done = mini_batch
        s = torch.tensor(s).to(device).float()
        a = torch.tensor(a).to(device).float()
        r = torch.tensor(r).to(device).float().unsqueeze(-1)
        s_prime = torch.tensor(s_prime).to(device).float()
        done = torch.tensor(done).to(device).float().unsqueeze(-1)
        return s, a, r, s_prime, done
    
    def size(self):
        return min(self.mem_cntr, self.mem_size)
