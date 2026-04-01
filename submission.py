import torch
import numpy as np

class OpenEnvNavigation:
    def __init__(self, size=10):
        self.size = size
        self.agent_pos = None
        self.target_pos = torch.tensor([9, 9])
        
    def reset(self):
        """Reset environment and return initial agent and target positions as tensors"""
        # Random starting position (not on target)
        self.agent_pos = torch.randint(0, self.size, (2,))
        while torch.equal(self.agent_pos, self.target_pos):
            self.agent_pos = torch.randint(0, self.size, (2,))
            
        return self.agent_pos.clone(), self.target_pos.clone()
    
    def step(self, action):
        """
        Take action (0:Up, 1:Down, 2:Left, 3:Right)
        Returns: next_obs (agent_pos), reward, done
        """
        if self.agent_pos is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        # Movement deltas
        deltas = {
            0: torch.tensor([-1, 0]),  # Up
            1: torch.tensor([1, 0]),   # Down
            2: torch.tensor([0, -1]),  # Left
            3: torch.tensor([0, 1])    # Right
        }
        
        delta = deltas[action]
        next_pos = self.agent_pos + delta
        
        # Boundary check
        next_pos = torch.clamp(next_pos, 0, self.size - 1)
        self.agent_pos = next_pos
        
        # Check if target reached
        done = torch.equal(self.agent_pos, self.target_pos)
        reward = 1.0 if done else -0.1
        
        next_obs = self.agent_pos.clone()
        
        return next_obs, reward, done.item()

def get_env():
    """Returns an instance of OpenEnvNavigation"""
    return OpenEnvNavigation()
