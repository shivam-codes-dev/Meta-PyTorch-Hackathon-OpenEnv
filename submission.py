import torch
import numpy as np

class OpenEnvNavigation:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.agent_pos = np.array([0, 0])
        self.target_pos = np.array([9, 9])

    def reset(self):
        self.agent_pos = np.array([0, 0])
        # Return observation as a torch tensor
        obs = np.concatenate([self.agent_pos, self.target_pos]).astype(np.float32)
        return torch.from_numpy(obs)

    def step(self, action):
        # 0: Up, 1: Down, 2: Left, 3: Right
        if action == 0: self.agent_pos[1] = min(9, self.agent_pos[1] + 1)
        elif action == 1: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 2: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3: self.agent_pos[0] = min(9, self.agent_pos[0] + 1)
        
        done = np.array_equal(self.agent_pos, self.target_pos)
        reward = 1.0 if done else -0.1
        obs = np.concatenate([self.agent_pos, self.target_pos]).astype(np.float32)
        return torch.from_numpy(obs), reward, done

# Global instance for Scaler validation
def get_env():
    return OpenEnvNavigation()
