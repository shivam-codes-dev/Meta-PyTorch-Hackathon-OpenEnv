# AI Warehouse Navigation System (Meta x Scaler Hackathon)

This project implements a professional Reinforcement Learning environment where an AI agent learns to navigate a warehouse grid, avoid obstacles, and reach a target destination efficiently.

## Features
- **Library:** Built with PyTorch and NumPy.
- **Observation Space:** 3x10x10 normalized tensor (Agent, Target, and Obstacle layers).
- **Rewards:** - +100 for reaching the Target.
  - -10 for hitting boundaries/obstacles.
  - -0.1 step penalty for efficiency.
- **API:** Fully compliant with standard step(), reset(), and state() methods.

## Files
- `warehouse_env.py`: Core Environment logic.
- `inference.py`: Script to run the environment.
- `Dockerfile`: Containerization for automated evaluation.
