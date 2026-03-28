import torch
import numpy as np

# --- AAPKA REAL WORLD CODE ---
class OpenEnvNavigation:
    def __init__(self, grid_size=10, num_obstacles=15, max_steps=100):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.target_pos = np.array([self.grid_size - 1, self.grid_size - 1])
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        # Normalizing for AI (0 to 1)
        obs = np.concatenate([
            self.agent_pos / self.grid_size,
            self.target_pos / self.grid_size
        ])
        return torch.tensor(obs, dtype=torch.float32)

    def step(self, action):
        self.steps += 1
        # 0: Up, 1: Down, 2: Left, 3: Right
        if action == 0: self.agent_pos[1] = min(self.grid_size-1, self.agent_pos[1] + 1)
        elif action == 1: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 2: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3: self.agent_pos[0] = min(self.grid_size-1, self.agent_pos[0] + 1)

        done = np.array_equal(self.agent_pos, self.target_pos) or self.steps >= self.max_steps
        reward = 100.0 if np.array_equal(self.agent_pos, self.target_pos) else -0.1
        
        return self._get_obs(), reward, done

    def state(self):
        return {"pos": self.agent_pos, "steps": self.steps}

# --- VALIDATION PART ---
if __name__ == "__main__":
    env = OpenEnvNavigation()
    print("Environment logic is working!")
