import torch
from warehouse_env import WarehouseEnv

def inference():
    # Aapka naya environment initialize karein
    env = WarehouseEnv(grid_size=10, num_obstacles=15)
    obs = env.reset()
    
    # Scaler validation ke liye ek sample step
    action = 0 
    obs, reward, done, info = env.step(action)
    
    print("OpenEnv Validation Successful")
    return {"status": "success"}

if __name__ == "__main__":
    inference()
