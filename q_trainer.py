"""
Week 4 — Q-Learning Trainer + Evaluator
=========================================
Handles the full training loop, evaluation, and mid-project review gate.

Classes
-------
TrainingLog   — per-episode statistics collected during training
QTrainer      — runs the training loop, manages ε-schedule, logs results
MidProjectReview — automated gate check (resource sheet §Week4)
"""

import sys
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, "/mnt/user-data/outputs")
sys.path.insert(0, "/home/claude")

from bertrand_pricing_env import BertrandPricingEnv
from agents import RandomAgent
from q_agent import QLearningAgent, QHyperparams, EpsilonSchedule


# ═══════════════════════════════════════════════════════════════════════
#  Per-episode log
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeRecord:
    episode        : int
    total_reward   : float
    mean_reward    : float
    mean_price     : float
    mean_profit    : float
    mean_td_error  : float
    epsilon        : float
    q_table_size   : int
    n_steps        : int

    @property
    def collusion_index(self) -> float:
        # Filled in post-hoc by trainer using env benchmarks
        return getattr(self, '_ci', 0.0)


@dataclass
class TrainingLog:
    records        : List[EpisodeRecord] = field(default_factory=list)
    nash_price     : float = 48.0
    monopoly_price : float = 60.0

    def append(self, rec: EpisodeRecord) -> None:
        self.records.append(rec)

    def window_mean(self, field: str, window: int = 100) -> float:
        vals = [getattr(r, field) for r in self.records[-window:]]
        return float(np.mean(vals)) if vals else 0.0

    def collusion_indices(self) -> List[float]:
        rng = self.monopoly_price - self.nash_price
        if rng <= 0:
            return [0.0] * len(self.records)
        return [
            float(np.clip((r.mean_price - self.nash_price) / rng, 0, 1))
            for r in self.records
        ]


# ═══════════════════════════════════════════════════════════════════════
#  Trainer
# ═══════════════════════════════════════════════════════════════════════

class QTrainer:
    """
    Training loop for QLearningAgent in BertrandPricingEnv.

    Each episode:
      1. Reset environment + agent
      2. Step through n_steps, calling agent.update() after each step
      3. Decay ε after each episode
      4. Log statistics

    Parameters
    ----------
    env            : BertrandPricingEnv
    agent          : QLearningAgent
    n_episodes     : int   — total training episodes
    n_steps        : int   — steps per episode (overrides env.max_steps)
    log_interval   : int   — print progress every N episodes
    seed           : int   — base random seed
    """

    def __init__(
        self,
        env         : BertrandPricingEnv,
        agent       : QLearningAgent,
        n_episodes  : int = 5000,
        n_steps     : int = 200,
        log_interval: int = 500,
        seed        : int = 0,
    ):
        self.env          = env
        self.agent        = agent
        self.n_episodes   = n_episodes
        self.n_steps      = n_steps
        self.log_interval = log_interval
        self.seed         = seed

        self.eps_schedule = EpsilonSchedule(
            eps_start      = agent.hp.eps_start,
            eps_end        = agent.hp.eps_end,
            total_episodes = n_episodes,
            decay_frac     = agent.hp.eps_decay_frac,
        )
        self.log = TrainingLog(
            nash_price     = env.nash_price,
            monopoly_price = env.monopoly_price,
        )

    # ── Main training loop ───────────────────────────────────────────

    def train(self) -> TrainingLog:
        print(f"\n{'═'*60}")
        print(f"  Q-LEARNING TRAINING")
        print(f"  {self.n_episodes} episodes × {self.n_steps} steps")
        print(f"  α={self.agent.hp.alpha}  γ={self.agent.hp.gamma}"
              f"  ε: {self.agent.hp.eps_start}→{self.agent.hp.eps_end}"
              f"  bins={self.agent.hp.n_bins}")
        print(f"  Nash={self.env.nash_price:.1f}  Monopoly={self.env.monopoly_price:.1f}")
        print(f"{'═'*60}")

        t0 = time.time()

        for ep in range(self.n_episodes):
            ep_seed = self.seed + ep
            obs, info = self.env.reset(seed=ep_seed)
            self.agent.reset()

            # Set current ε on agent before the episode
            self.agent._epsilon = self.eps_schedule.step()

            rewards, prices, profits, td_errors = [], [], [], []

            for step in range(self.n_steps):
                action = self.agent.act(obs, info, training=True)
                next_obs, reward, terminated, truncated, next_info = self.env.step(action)

                td_err = self.agent.update(
                    obs, action, reward, next_obs, terminated, truncated
                )

                rewards.append(reward)
                prices.append(next_info["p1"])
                profits.append(next_info["profit_firm1"])
                td_errors.append(abs(td_err))

                obs  = next_obs
                info = next_info

                if terminated or truncated:
                    break

            rec = EpisodeRecord(
                episode       = ep,
                total_reward  = float(np.sum(rewards)),
                mean_reward   = float(np.mean(rewards)),
                mean_price    = float(np.mean(prices)),
                mean_profit   = float(np.mean(profits)),
                mean_td_error = float(np.mean(td_errors)),
                epsilon       = self.agent._epsilon,
                q_table_size  = self.agent.q_table_size,
                n_steps       = len(rewards),
            )
            self.log.append(rec)

            if (ep + 1) % self.log_interval == 0 or ep == 0:
                ci = np.clip(
                    (rec.mean_price - self.env.nash_price) /
                    (self.env.monopoly_price - self.env.nash_price),
                    0, 1
                )
                elapsed = time.time() - t0
                print(
                    f"  ep {ep+1:>5d}/{self.n_episodes}"
                    f"  ε={self.agent._epsilon:.3f}"
                    f"  r̄={self.log.window_mean('mean_reward'):6.4f}"
                    f"  P̄={self.log.window_mean('mean_price'):6.2f}"
                    f"  CI={ci:.3f}"
                    f"  |TD|={self.log.window_mean('mean_td_error'):.4f}"
                    f"  Q-states={self.agent.q_table_size:>6d}"
                    f"  {elapsed:.0f}s"
                )

        total_time = time.time() - t0
        print(f"\n  Training complete in {total_time:.1f}s")
        print(f"  Q-table: {self.agent.q_table_size} unique states visited")
        return self.log

    # ── Evaluation (greedy, no exploration) ─────────────────────────

    def evaluate(
        self,
        n_episodes : int = 50,
        n_steps    : int = 200,
        seed_offset: int = 99000,
        verbose    : bool = True,
    ) -> dict:
        """
        Run the trained agent greedily (ε=0) against the env's
        best-response firm 2. Returns mean profit, price, and CI.
        """
        saved_eps = self.agent._epsilon
        self.agent._epsilon = 0.0  # fully greedy

        ep_profits, ep_prices, ep_rewards = [], [], []

        for ep in range(n_episodes):
            obs, info = self.env.reset(seed=seed_offset + ep)
            self.agent.reset()
            profits, prices, rewards = [], [], []

            for _ in range(n_steps):
                action = self.agent.act(obs, info, training=False)
                obs, reward, terminated, truncated, info = self.env.step(action)
                profits.append(info["profit_firm1"])
                prices.append(info["p1"])
                rewards.append(reward)
                if terminated or truncated:
                    break

            ep_profits.append(np.mean(profits))
            ep_prices.append(np.mean(prices))
            ep_rewards.append(np.mean(rewards))

        self.agent._epsilon = saved_eps

        mean_profit = float(np.mean(ep_profits))
        mean_price  = float(np.mean(ep_prices))
        ci = float(np.clip(
            (mean_price - self.env.nash_price) /
            (self.env.monopoly_price - self.env.nash_price),
            0, 1
        ))

        result = {
            "mean_profit" : mean_profit,
            "std_profit"  : float(np.std(ep_profits)),
            "mean_price"  : mean_price,
            "collusion_index": ci,
            "mean_reward" : float(np.mean(ep_rewards)),
            "n_episodes"  : n_episodes,
        }

        if verbose:
            print(f"\n  EVALUATION ({n_episodes} greedy episodes)")
            print(f"  Mean profit : {mean_profit:.1f} ± {result['std_profit']:.1f}")
            print(f"  Mean price  : {mean_price:.2f}  (Nash={self.env.nash_price:.1f}  Mono={self.env.monopoly_price:.1f})")
            print(f"  Collusion index : {ci:.3f}")

        return result


# ═══════════════════════════════════════════════════════════════════════
#  Mid-project review gate
# ═══════════════════════════════════════════════════════════════════════

class MidProjectReview:
    """
    Automated gate — resource sheet requirement:
    'The Q-learning agent must consistently beat the Random baseline.'

    Runs both the trained Q-agent and a Random agent for eval_episodes
    episodes each (same seeds), then compares mean profits.

    Also checks:
    - Q-agent mean price is ABOVE Nash (it learned something)
    - Q-agent mean price is BELOW monopoly (not trivially colluding)
    - Q-table has a reasonable number of visited states
    """

    def __init__(
        self,
        env           : BertrandPricingEnv,
        trainer       : QTrainer,
        eval_episodes : int = 100,
        eval_steps    : int = 200,
        seed_offset   : int = 50000,
    ):
        self.env           = env
        self.trainer       = trainer
        self.eval_episodes = eval_episodes
        self.eval_steps    = eval_steps
        self.seed_offset   = seed_offset

    def run(self) -> bool:
        print(f"\n{'═'*60}")
        print("  MID-PROJECT REVIEW GATE")
        print(f"{'═'*60}")

        # ── Q-agent evaluation ──────────────────────────────────────
        print("\n  [1/3] Evaluating trained Q-agent (greedy)...")
        q_result = self.trainer.evaluate(
            n_episodes  = self.eval_episodes,
            n_steps     = self.eval_steps,
            seed_offset = self.seed_offset,
            verbose     = False,
        )

        # ── Random baseline ─────────────────────────────────────────
        print("  [2/3] Evaluating Random baseline...")
        rand_agent = RandomAgent(self.env.price_grid, seed=99)
        rand_profits = []

        for ep in range(self.eval_episodes):
            rand_agent.reset()
            obs, info = self.env.reset(seed=self.seed_offset + ep)
            ep_profits = []
            for _ in range(self.eval_steps):
                action = rand_agent.act(obs, info)
                obs, _, _, trunc, info = self.env.step(action)
                ep_profits.append(info["profit_firm1"])
                if trunc: break
            rand_profits.append(np.mean(ep_profits))

        rand_mean = float(np.mean(rand_profits))
        rand_std  = float(np.std(rand_profits))

        # ── Gate checks ─────────────────────────────────────────────
        print("\n  [3/3] Running gate checks...")
        checks = {}

        checks["Q-agent beats Random"] = (
            q_result["mean_profit"] > rand_mean,
            f"Q={q_result['mean_profit']:.1f}  Random={rand_mean:.1f}"
        )
        checks["Q-agent price above Nash"] = (
            q_result["mean_price"] > self.env.nash_price,
            f"Q price={q_result['mean_price']:.2f}  Nash={self.env.nash_price:.1f}"
        )
        checks["Q-agent price below Monopoly"] = (
            q_result["mean_price"] < self.env.monopoly_price,
            f"Q price={q_result['mean_price']:.2f}  Mono={self.env.monopoly_price:.1f}"
        )
        checks["Collusion index in (0,1)"] = (
            0.0 < q_result["collusion_index"] < 1.0,
            f"CI={q_result['collusion_index']:.3f}"
        )
        checks["Q-table has visited states"] = (
            self.trainer.agent.q_table_size > 100,
            f"{self.trainer.agent.q_table_size} states"
        )

        print(f"\n  {'Check':<36} {'Result':<8} Detail")
        print(f"  {'─'*58}")
        all_pass = True
        for name, (passed, detail) in checks.items():
            icon = "✅" if passed else "❌"
            print(f"  {icon}  {name:<34} {detail}")
            if not passed:
                all_pass = False

        print(f"\n  Random baseline  : {rand_mean:.1f} ± {rand_std:.1f}")
        print(f"  Q-agent result   : {q_result['mean_profit']:.1f} ± {q_result['std_profit']:.1f}")
        margin = q_result['mean_profit'] - rand_mean
        print(f"  Improvement      : +{margin:.1f} ({margin/rand_mean*100:.1f}% over Random)")

        print(f"\n{'═'*60}")
        if all_pass:
            print("  ✅  GATE PASSED — cleared to proceed to Week 5 (DQN/PPO)")
        else:
            print("  ❌  GATE FAILED — debug environment or hyperparams before Week 5")
            print("      Hint: check reward scaling, epsilon decay schedule, n_bins")
        print(f"{'═'*60}\n")

        return all_pass
