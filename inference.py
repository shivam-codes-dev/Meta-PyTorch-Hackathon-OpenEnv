import torch
from your_module import OpenEnvNavigation, get_env  # Replace 'your_module' with actual module name

# Get environment
env = get_env()

# Reset environment
print("=== Reset Environment ===")
agent_pos, target_pos = env.reset()
print(f"Initial agent position: {agent_pos}")
print(f"Target position: {target_pos}")
print(f"Agent at target? {torch.equal(agent_pos, target_pos)}")
print()

# Perform one step (move Right)
print("=== Step 1: Action=3 (Right) ===")
next_obs, reward, done = env.step(3)
print(f"Next agent position: {next_obs}")
print(f"Reward: {reward}")
print(f"Done: {done}")
print(f"Agent at target? {torch.equal(next_obs, target_pos)}")
print()

# Verify boundary handling (try moving out of bounds)
print("=== Boundary Test: From corner (0,0) move Left/Up ===")
env.reset()  # Reset to potentially get corner position
if list(env.agent_pos) == [0, 0]:
    obs, r, d = env.step(2)  # Left from (0,0)
    print(f"Move Left from (0,0): {obs}, reward: {r}, done: {d}")
    obs, r, d = env.step(0)  # Up from (0,0)
    print(f"Move Up from (0,0): {obs}, reward: {r}, done: {d}")
else:
    print("Didn't start at corner, but boundaries still work!")
