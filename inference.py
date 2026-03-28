import torch
from submission import OpenEnvNavigation

def main():
    env = OpenEnvNavigation()
    obs = env.reset()
    print("Reset OK")
    obs, reward, done = env.step(0)
    print("Step OK")

if __name__ == "__main__":
    main()
