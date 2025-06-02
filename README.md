# 🏓 Deep Reinforcement Learning Pong Game (Gymnasium + PyGame)

This project is a Deep Q-Network (DQN) implementation of the classic Pong game, built using **Gymnasium** (https://gymnasium.farama.org/) and **PyGame**. It enables both AI-vs-AI training and manual-vs-AI play modes.

## 🎮 Game Mechanics

- Reward System:
  - +15 for paddle collision with the ball
  - +15 when opponent misses the ball
  - -10 when the agent misses the ball

- Players are trained **simultaneously** on both sides using DQN.

- Post-training:
  - You can **manually control one paddle** against the AI.
  - Or let the game run in **AI vs AI** mode.

---

## 🧠 Project Structure

| File | Description |
|------|-------------|
| `game_environment.py` | Contains the Gymnasium environment logic using PyGame for rendering and simulation. |
| `agent.py` | Main script to train/test the agent. Integrates all modules. |
| `replay_memory.py` | Implements experience replay for stable training and future Q-value estimation. |
| `neural_network.py` | Contains the DQN model architecture used by the agent. |
| `hyperparameters.py` | Central file to configure and tweak all hyperparameters for training/testing. |

---

## 💾 Features

- Model autosaves when it achieves a new **highest reward**, along with:
  - Epsilon value
  - Training step count
  - Current best result

- Training can **resume seamlessly** from the last saved checkpoint.

- Training progress is visualized with **graphs** (e.g., reward per episode, epsilon decay, etc.).

---
## Visualization:
Training progress is plotted and saved as graphs, including:
  -Reward per episode
  -Epsilon decay
Graphs are automatically saved to the project directory during training.

## 🚀 Getting Started

### 🔧 Install Dependencies

Make sure you have Python 3.8+ installed.
Dependencies include: gymnasium, pygame, numpy, torch, matplotlib

### Train agent
python agent.py pong --train

### Play the game
python agent.py pong




