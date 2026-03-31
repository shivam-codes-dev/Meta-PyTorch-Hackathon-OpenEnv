from submission import OpenEnvNavigation
import torch

def main():
    try:
        env = OpenEnvNavigation()
        obs = env.reset()
        print("SUCCESS: Reset executed")
        
        obs, reward, done = env.step(0)
        print(f"SUCCESS: Step executed, Reward: {reward}")
    except Exception as e:
        print(f"FAILED: {str(e)}")

if __name__ == "__main__":
    main()
