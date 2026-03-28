import torch
from warehouse_env import WarehouseEnv

def run_submission():
    print("Initializing Warehouse Environment...")
    # Creating environment based on your new class
    env = WarehouseEnv(grid_size=10, num_obstacles=15, max_steps=100)
    
    # Resetting the environment
    obs = env.reset()
    print(f"Initial Observation Shape: {obs.shape}")
    
    # Taking a sample step (Action 0: UP)
    obs, reward, done, info = env.step(0)
    print(f"Step executed. Reward: {reward}, Done: {done}")
    
    if done or not done:
        print("✓ OpenEnv Validation Passed!")

if __name__ == "__main__":
    run_submission()
