
class OpenEnvNavigation(WarehouseEnv): # Ise WarehouseEnv ki jagah likhein
    pass

```python
# Simple usage example
from warehouse_env import WarehouseEnv
import torch

# Create environment
env = WarehouseEnv(grid_size=10, num_obstacles=15, max_steps=100, seed=42)

# Reset environment
observation = env.reset()
print(f"Observation shape: {observation.shape}")  # torch.Size([3, 10, 10])
print(f"Observation type: {type(observation)}")   # <class 'torch.Tensor'>

# Render initial state
env.render(mode='human')

# Run one episode
done = False
total_reward = 0

while not done:
    # Take random action (replace with your RL agent)
    action = torch.randint(0, 4, (1,)).item()
    
    # Execute action
    observation, reward, done, info = env.step(action)
    total_reward += reward
    
    # Optional: render every 10 steps
    if env.current_step % 10 == 0:
        env.render(mode='human')

print(f"Episode finished! Total reward: {total_reward:.2f}")
if 'success' in info:
    print("✓ SUCCESS! Agent reached the target!")
```

## 3. Test Script (`test_warehouse_env.py`)
Action: file_editor view /app/test_warehouse_env.py
Observation: [Showing lines 1-96 of 96 total] /app/test_warehouse_env.py:
1|"""
2|Simple Test Script for Warehouse Environment
3|============================================
4|Quick verification that the environment works correctly.
5|"""
6|
7|import torch
8|from warehouse_env import WarehouseEnv
9|
10|
11|def test_basic_functionality():
12|    """Test basic environment functionality."""
13|    print("Testing Warehouse Environment...")
14|    print("=" * 60)
15|    
16|    # Create environment
17|    env = WarehouseEnv(grid_size=5, num_obstacles=3, seed=42)
18|    
19|    # Test reset
20|    print("\n✓ Testing reset()...")
21|    obs = env.reset()
22|    assert obs.shape == (3, 5, 5), f"Wrong observation shape: {obs.shape}"
23|    assert obs.min() >= 0.0 and obs.max() <= 1.0, "Observation not normalized!"
24|    print(f"  Observation shape: {obs.shape}")
25|    print(f"  Value range: [{obs.min():.1f}, {obs.max():.1f}]")
26|    
27|    # Test step
28|    print("\n✓ Testing step()...")
29|    obs, reward, done, info = env.step(0)
30|    assert isinstance(reward, float), "Reward must be float"
31|    assert isinstance(done, bool), "Done must be boolean"
32|    assert isinstance(info, dict), "Info must be dict"
33|    print("  Action executed successfully")
34|    print(f"  Reward: {reward}, Done: {done}")
35|    
36|    # Test state
37|    print("\n✓ Testing state()...")
38|    state = env.state()
39|    assert state.shape == (3, 5, 5), "Wrong state shape"
40|    print(f"  State shape: {state.shape}")
41|    
42|    # Test action space
43|    print("\n✓ Testing action space...")
44|    action_size = env.get_action_space_size()
45|    assert action_size == 4, "Action space should be 4"
46|    print(f"  Action space size: {action_size}")
47|    
48|    # Test observation space
49|    print("\n✓ Testing observation space...")
50|    obs_shape = env.get_observation_space_shape()
51|    assert obs_shape == (3, 5, 5), "Wrong observation space shape"
52|    print(f"  Observation space shape: {obs_shape}")
53|    
54|    # Test rewards
55|    print("\n✓ Testing reward system...")
56|    
57|    # Test step penalty
58|    env.reset()
59|    old_pos = env.agent_pos
60|    obs, reward, done, info = env.step(WarehouseEnv.RIGHT)
61|    if env.agent_pos != old_pos:
62|        assert reward == -0.1, f"Step penalty wrong: {reward}"
63|        print(f"  ✓ Step penalty: {reward}")
64|    
65|    # Test boundary collision
66|    env.agent_pos = (0, 0)
67|    obs, reward, done, info = env.step(WarehouseEnv.UP)
68|    assert reward == -10.0, f"Boundary penalty wrong: {reward}"
69|    print(f"  ✓ Boundary penalty: {reward}")
70|    
71|    # Test target reward
72|    env.agent_pos = (2, 2)
73|    env.target_pos = (2, 3)
74|    obs, reward, done, info = env.step(WarehouseEnv.RIGHT)
75|    assert reward == 100.0, f"Target reward wrong: {reward}"
76|    assert done, "Episode should end at target"
77|    print(f"  ✓ Target reward: {reward}")
78|    print(f"  ✓ Episode ends when target reached: {done}")
79|    
80|    # Test render
81|    print("\n✓ Testing render()...")
82|    env.reset()
83|    env.render(mode='human')
84|    rgb = env.render(mode='rgb_array')
85|    assert rgb.shape == (5, 5, 3), "Wrong RGB array shape"
86|    print(f"  RGB array shape: {rgb.shape}")
87|    
88|    print("\n" + "=" * 60)
89|    print("✅ ALL TESTS PASSED!")
90|    print("=" * 60)
91|    print("\nEnvironment is ready for RL training!")
92|
93|
94|if __name__ == "__main__":
95|    test_basic_functionality()
96|
[End of file]
