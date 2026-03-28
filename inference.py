import torch
from Submission import OpenEnvNavigation

def run_inference():
    env = OpenEnvNavigation()
    state = env.reset()
    print("Inference started successfully!")
    return state

if __name__ == "__main__":
    run_inference()
