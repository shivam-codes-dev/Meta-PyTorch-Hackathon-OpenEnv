import os
from openai import OpenAI
import torch
from submission import get_env

# 1. Environment Variables (Checklist Rule: No hardcoded tokens)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.scaler.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
HF_TOKEN = os.getenv("HF_TOKEN")

# 2. Configure OpenAI Client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN  # Checklist says use HF_TOKEN for auth
)

def run_inference():
    print("START") # Required by Checklist
    
    env = get_env()
    obs = env.reset()
    
    # 3. LLM Call using configured client
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": f"Current observation: {obs.tolist()}. What is the next best move (0,1,2,3)?"}]
    )
    
    # Simple logic to extract action from LLM response
    action = 0 # Default
    try:
        content = response.choices[0].message.content
        if "1" in content: action = 1
        elif "2" in content: action = 2
        elif "3" in content: action = 3
    except:
        pass

    obs, reward, done = env.step(action)
    print(f"STEP: Action={action}, Reward={reward}") # Required format
    
    print("END") # Required by Checklist

if __name__ == "__main__":
    run_inference()
