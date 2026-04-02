import torch
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

class OpenEnvNavigation:
    def __init__(self):
        self.agent_pos = np.array([0, 0])
        self.target_pos = np.array([9, 9])

    def reset(self):
        self.agent_pos = np.array([0, 0])
        return self.agent_pos.tolist()

    def step(self, action):
        # 0: Up, 1: Down, 2: Left, 3: Right
        if action == 0: self.agent_pos[1] = min(9, self.agent_pos[1] + 1)
        elif action == 1: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 2: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3: self.agent_pos[0] = min(9, self.agent_pos[0] + 1)
        
        done = np.array_equal(self.agent_pos, self.target_pos)
        reward = 1.0 if done else -0.1
        return self.agent_pos.tolist(), reward, done

env = OpenEnvNavigation()

# Scaler isi endpoint ko hit karta hai (POST OK check ke liye)
@app.route('/reset', methods=['POST'])
def reset():
    obs = env.reset()
    return jsonify({"observation": obs})

@app.route('/step', methods=['POST'])
def step():
    data = request.json
    action = data.get('action', 0)
    obs, reward, done = env.step(action)
    return jsonify({"observation": obs, "reward": reward, "done": done})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
