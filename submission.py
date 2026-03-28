
import torch
import numpy as np

class OpenEnvNavigation:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.target_pos = np.array([self.grid_size - 1, self.grid_size - 1])
        return self._get_obs()

    def _get_obs(self):
        obs = np.concatenate([self.agent_pos / self.grid_size, self.target_pos / self.grid_size])
        return torch.tensor(obs, dtype=torch.float32)

    def step(self, action):
        # 0: Up, 1: Down, 2: Left, 3: Right
        if action == 0: self.agent_pos[1] = min(self.grid_size-1, self.agent_pos[1] + 1)
        elif action == 1: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 2: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3: self.agent_pos[0] = min(self.grid_size-1, self.agent_pos[0] + 1)
        
        done = np.array_equal(self.agent_pos, self.target_pos)
        reward = 10.0 if done else -0.1
        return self._get_obs(), reward, done
