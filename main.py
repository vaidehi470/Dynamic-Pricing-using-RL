"""
Week 4 — Main runner script
============================
Run this file to:
  1. Train the Q-learning agent
  2. Evaluate it against all Week 3 baselines
  3. Run the mid-project review gate
  4. Save results to JSON for the widget / report

Usage:  python run_week4.py
"""

import sys, json, time
import numpy as np

# sys.path.insert(0, "/mnt/user-data/outputs")
# sys.path.insert(0, "/home/claude")

from bertrand_pricing_env import BertrandPricingEnv
from agents import AlwaysNashAgent, AlwaysColludeAgent, TitForTatAgent, RandomAgent
from q_agent import QLearningAgent, QHyperparams
from q_trainer import QTrainer, MidProjectReview

# ── Environment ─────────────────────────────────────────────────────
env = BertrandPricingEnv(
    a=100, b=1.0, d=0.5, marginal_cost=20,
    n_price_levels=30, max_steps=200, noise_std=2.0
)

# ── Q-agent ─────────────────────────────────────────────────────────
hp = QHyperparams(
    alpha=0.10, gamma=0.95,
    eps_start=1.0, eps_end=0.05,
    eps_decay_frac=0.80, n_bins=10
)
agent = QLearningAgent(n_actions=env.action_space.n, hp=hp, seed=42)

# ── Train ────────────────────────────────────────────────────────────
trainer = QTrainer(
    env=env, agent=agent,
    n_episodes=3000, n_steps=200,
    log_interval=500, seed=0
)
log = trainer.train()

import matplotlib.pyplot as plt

episodes = [r.episode for r in log.records]
rewards = [r.mean_reward for r in log.records]

plt.figure(figsize=(8,5))
plt.plot(episodes, rewards)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Learning Curve")
plt.grid(True)
plt.show()


prices = [r.mean_price for r in log.records]
plt.figure(figsize=(8,5))
plt.plot(episodes, prices)

plt.axhline(env.nash_price,
            linestyle="--",
            label="Nash Price")

plt.axhline(env.monopoly_price,
            linestyle="--",
            label="Monopoly Price")

plt.xlabel("Episode")
plt.ylabel("Price")
plt.title("Average Price Evolution")
plt.legend()
plt.show()

# ── Evaluate vs baselines ────────────────────────────────────────────
baselines = [
    ("Always-Nash",    AlwaysNashAgent(env.price_grid, env.nash_price)),
    ("Always-Collude", AlwaysColludeAgent(env.price_grid, env.monopoly_price)),
    ("Tit-for-Tat",   TitForTatAgent(env.price_grid, env.monopoly_price)),
    ("Random",         RandomAgent(env.price_grid, seed=42)),
]

# print("\n  BASELINE COMPARISON (Q-agent as firm 1, 50 eval episodes)")
# baseline_results = {}
# for name, baseline in baselines:
#     profits = []
#     agent._epsilon = 0.0
#     for ep in range(50):
#         obs, info = env.reset(seed=80000+ep)
#         agent.reset()
#         ep_pi = []
#         for _ in range(200):
#             action = agent.act(obs, info, training=False)
#             obs, _, _, trunc, info = env.step(action)
#             ep_pi.append(info["profit_firm1"])
#             if trunc: break
#         profits.append(np.mean(ep_pi))       
#     m = float(np.mean(profits))
#     baseline_results[name] = round(m, 1)
#     print(f"  Q vs {name:<16}: Q profit = {m:.1f}")
print("\nBASELINE COMPARISON")

baseline_results = {}

for name, baseline in baselines:

    profits = []

    for ep in range(50):

        obs, info = env.reset(seed=80000 + ep)
        baseline.reset()

        ep_pi = []

        for _ in range(200):

            action = baseline.act(obs, info)

            obs, reward, terminated, truncated, info = env.step(action)

            ep_pi.append(info["profit_firm1"])

            if terminated or truncated:
                break

        profits.append(np.mean(ep_pi))

    baseline_results[name] = round(float(np.mean(profits)), 2)

    print(
        f"{name:<16} "
        f"Profit = {baseline_results[name]:.2f}"
    )

# Evaluate trained Q agent separately

q_eval = trainer.evaluate(
    n_episodes=50,
    n_steps=200,
    verbose=False
)

print(
    f"\nQ-Learning Agent Profit = "
    f"{q_eval['mean_profit']:.2f}"
)

plt.figure(figsize=(8,5))
plt.bar(
    baseline_results.keys(),
    baseline_results.values()
)
plt.ylabel("Average Profit")
plt.title("Q-Agent vs Baselines")

plt.show()
# ── Mid-project review ────────────────────────────────────────────────
review = MidProjectReview(env, trainer, eval_episodes=100, eval_steps=200)
passed = review.run()

# ── Save training curve data ──────────────────────────────────────────
# Downsample to every 50 episodes for the widget
step = max(1, len(log.records) // 60)
curve_eps     = [r.episode    for r in log.records[::step]]
curve_price   = [round(r.mean_price, 2)   for r in log.records[::step]]
curve_reward  = [round(r.mean_reward, 4)  for r in log.records[::step]]
curve_epsilon = [round(r.epsilon, 4)      for r in log.records[::step]]
curve_td      = [round(r.mean_td_error,4) for r in log.records[::step]]
ci_list       = log.collusion_indices()
curve_ci      = [round(ci_list[i], 3) for i in range(0, len(ci_list), step)]

plt.figure(figsize=(8,5))
plt.plot(ci_list)

plt.xlabel("Episode")
plt.ylabel("Collusion Index")
plt.title("Collusion Index Evolution")
plt.grid(True)
plt.show()

td = [r.mean_td_error for r in log.records]
plt.figure(figsize=(8,5))
plt.plot(td)
plt.xlabel("Episode")
plt.ylabel("TD Error")
plt.title("Bellman Error Convergence")
plt.grid(True)
plt.show()

# Smooth with rolling window
def smooth(arr, w=5):
    out = []
    for i in range(len(arr)):
        sl = arr[max(0,i-w):i+1]
        out.append(round(float(np.mean(sl)), 4))
    return out

final_eval = trainer.evaluate(n_episodes=100, n_steps=200, verbose=True)

output = {
    "nash_price"     : env.nash_price,
    "monopoly_price" : env.monopoly_price,
    "n_episodes"     : 3000,
    "hp"             : {"alpha":hp.alpha,"gamma":hp.gamma,
                        "eps_start":hp.eps_start,"eps_end":hp.eps_end,
                        "decay_frac":hp.eps_decay_frac,"n_bins":hp.n_bins},
    "curve": {
        "episodes" : curve_eps,
        "price"    : smooth(curve_price),
        "reward"   : smooth(curve_reward),
        "epsilon"  : curve_epsilon,
        "td_error" : smooth(curve_td),
        "ci"       : smooth(curve_ci),
    },
    "final_eval"       : final_eval,
    "baseline_results" : baseline_results,
    "gate_passed"      : passed,
    "q_table_size"     : agent.q_table_size,
}

with open("week4_results.json","w") as f:
    json.dump(output, f, indent=2)
print("\nResults saved → week4_results.json")
