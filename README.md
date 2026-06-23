# Dynamic Pricing using Reinforcement Learning in a Bertrand Competition Environment

## Overview
This project implements a Reinforcement Learning framework for dynamic pricing in a competitive market using the Bertrand Competition model. A tabular Q-Learning agent is trained to learn pricing strategies through repeated interaction with a simulated market environment.
The project combines concepts from:
* Industrial Organization Economics
* Reinforcement Learning
* Dynamic Pricing
* Multi-Period Strategic Competition

The primary objective is to investigate whether a learning agent can discover profitable pricing strategies without being explicitly programmed with economic equilibrium rules.

---

# Project Structure

```text
.
├── bertrand_pricing_env.py      # Market simulation environment
├── agents.py                    # Rule-based benchmark agents
├── q_agent.py                   # Q-Learning implementation
├── q_trainer.py                 # Training and evaluation pipeline               
├── tournament_1.py              # Tournament execution script
├── main.py                 # Main training and evaluation script
│
├── test_bertrand_env.py         # Environment tests
├── test_agents.py               # Agent tests
├── test_q_agent.py              # Q-learning tests
├── test_tournaments.py          # Tournament tests
│
└── README.md
```

---
# Running the Project
## Install Dependencies

```bash
pip install numpy gymnasium matplotlib
```

---

## Train the Q-Learning Agent

```bash
python main.py
```

This will:

1. Create the environment
2. Train the Q-Learning agent
3. Evaluate the learned policy
4. Generate performance metrics
5. Save results for analysis

---

## Run Rule-Based Tournament

```bash
python tournament_1.py
```

This executes a round-robin competition among benchmark agents.

---

## Run Tests

Environment Tests:

```bash
python test_bertrand_env.py
```

Agent Tests:

```bash
python test_agents.py
```

Q-Learning Tests:

```bash
python test_q_agent.py
```

---

# Example Results

A representative training run produced:

```json
{
  "mean_profit": 1624.67,
  "mean_price": 55.02,
  "collusion_index": 0.585,
  "q_table_size": 2148
}
```

Observations:
* The agent consistently learns profitable pricing behavior.
* Average prices lie between Nash and Monopoly benchmarks.
* The policy exhibits moderate collusive tendencies.
* Learned profits remain stable across evaluation episodes.

---

