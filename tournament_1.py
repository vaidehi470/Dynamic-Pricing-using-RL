"""
Week 3 — Round-Robin Tournament Runner
=======================================
Runs every pair of rule-based agents against each other in the
BertrandPricingEnv and collects structured results.

Usage
-----
    from tournament import Tournament
    t = Tournament(env, agents, n_episodes=20, n_steps=1000)
    results = t.run()          # returns list[MatchResult]
    t.print_summary(results)
"""

import sys
import time
import itertools
import numpy as np
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, "/mnt/user-data/outputs")
from bertrand_pricing_env import BertrandPricingEnv
from agents import BaseAgent


# ─────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeLog:
    """Raw data from one episode."""
    episode      : int
    prices_firm1 : list = field(default_factory=list)
    prices_firm2 : list = field(default_factory=list)
    profits_firm1: list = field(default_factory=list)
    profits_firm2: list = field(default_factory=list)
    rewards      : list = field(default_factory=list)

    @property
    def mean_profit_firm1(self) -> float:
        return float(np.mean(self.profits_firm1)) if self.profits_firm1 else 0.0

    @property
    def mean_profit_firm2(self) -> float:
        return float(np.mean(self.profits_firm2)) if self.profits_firm2 else 0.0

    @property
    def mean_price_firm1(self) -> float:
        return float(np.mean(self.prices_firm1)) if self.prices_firm1 else 0.0

    @property
    def mean_price_firm2(self) -> float:
        return float(np.mean(self.prices_firm2)) if self.prices_firm2 else 0.0


@dataclass
class MatchResult:
    """Aggregated result for one agent-vs-agent matchup."""
    agent1_name        : str
    agent2_name        : str
    n_episodes         : int
    n_steps            : int
    nash_price         : float
    monopoly_price     : float

    # Per-episode logs (kept for report export)
    episode_logs       : List[EpisodeLog] = field(default_factory=list)

    # Aggregated across all episodes
    mean_profit_agent1 : float = 0.0
    mean_profit_agent2 : float = 0.0
    std_profit_agent1  : float = 0.0
    std_profit_agent2  : float = 0.0
    mean_price_agent1  : float = 0.0
    mean_price_agent2  : float = 0.0

    # Collusion index: 0 = Nash, 1 = full monopoly
    collusion_index    : float = 0.0

    def compile(self) -> None:
        """Compute aggregated stats from episode_logs."""
        ep_pi1 = [e.mean_profit_firm1 for e in self.episode_logs]
        ep_pi2 = [e.mean_profit_firm2 for e in self.episode_logs]
        ep_p1  = [e.mean_price_firm1  for e in self.episode_logs]
        ep_p2  = [e.mean_price_firm2  for e in self.episode_logs]

        self.mean_profit_agent1 = float(np.mean(ep_pi1))
        self.mean_profit_agent2 = float(np.mean(ep_pi2))
        self.std_profit_agent1  = float(np.std(ep_pi1))
        self.std_profit_agent2  = float(np.std(ep_pi2))
        self.mean_price_agent1  = float(np.mean(ep_p1))
        self.mean_price_agent2  = float(np.mean(ep_p2))

        # Collusion index based on agent1's average price
        price_range = self.monopoly_price - self.nash_price
        if price_range > 0:
            self.collusion_index = np.clip(
                (self.mean_price_agent1 - self.nash_price) / price_range, 0, 1
            )


# ─────────────────────────────────────────────────────────────────────
#  Tournament
# ─────────────────────────────────────────────────────────────────────

class Tournament:
    """
    Round-robin tournament: every ordered pair of agents plays
    n_episodes episodes of n_steps steps each.

    Parameters
    ----------
    env        : BertrandPricingEnv  — shared environment (reset per episode)
    agents     : list[BaseAgent]     — all agents to pit against each other
    n_episodes : int                 — episodes per matchup (≥20 recommended)
    n_steps    : int                 — steps per episode  (≥1000 per resource sheet)
    seed       : int                 — base random seed (logged for reproducibility)
    verbose    : bool                — print live progress
    """

    def __init__(
        self,
        env       : BertrandPricingEnv,
        agents    : list,
        n_episodes: int  = 20,
        n_steps   : int  = 1000,
        seed      : int  = 0,
        verbose   : bool = True,
    ):
        self.env        = env
        self.agents     = agents
        self.n_episodes = n_episodes
        self.n_steps    = n_steps
        self.seed       = seed
        self.verbose    = verbose

    # ── main entry point ─────────────────────────────────────────────

    def run(self) -> List[MatchResult]:
        """Run the full round-robin. Returns list of MatchResult."""
        results = []
        pairs   = list(itertools.permutations(self.agents, 2))
        total   = len(pairs)

        for i, (a1, a2) in enumerate(pairs, 1):
            if self.verbose:
                print(f"\n[{i}/{total}]  {a1.name}  vs  {a2.name}")
            result = self._run_matchup(a1, a2)
            result.compile()
            results.append(result)

        return results

    # ── single matchup ────────────────────────────────────────────────

    def _run_matchup(self, agent1: BaseAgent, agent2: BaseAgent) -> MatchResult:
        mr = MatchResult(
            agent1_name    = agent1.name,
            agent2_name    = agent2.name,
            n_episodes     = self.n_episodes,
            n_steps        = self.n_steps,
            nash_price     = self.env.nash_price,
            monopoly_price = self.env.monopoly_price,
        )

        for ep in range(self.n_episodes):
            ep_seed = self.seed + ep * 100
            obs, info = self.env.reset(seed=ep_seed)
            agent1.reset()
            agent2.reset()

            log = EpisodeLog(episode=ep)

            for _ in range(self.n_steps):
                # Agent 1 acts as firm 1
                action = agent1.act(obs, info)
                obs, reward, terminated, truncated, info = self.env.step(action)

                log.prices_firm1.append(info["p1"])
                log.prices_firm2.append(info["p2"])
                log.profits_firm1.append(info["profit_firm1"])
                log.profits_firm2.append(info["profit_firm2"])
                log.rewards.append(reward)

                if terminated or truncated:
                    break

            mr.episode_logs.append(log)

            if self.verbose:
                print(
                    f"  ep {ep+1:>3d}/{self.n_episodes}"
                    f"  π1={log.mean_profit_firm1:8.1f}"
                    f"  π2={log.mean_profit_firm2:8.1f}"
                    f"  P̄1={log.mean_price_firm1:6.2f}"
                    f"  P̄2={log.mean_price_firm2:6.2f}"
                )

        return mr

    # ── reporting ─────────────────────────────────────────────────────

    def print_summary(self, results: List[MatchResult]) -> None:
        """Print a formatted results table — needed for final report."""
        print("\n" + "═"*82)
        print(f"  TOURNAMENT SUMMARY   "
              f"({self.n_episodes} eps × {self.n_steps} steps each)")
        print("═"*82)
        header = f"{'Matchup':<36} {'π̄₁':>9} {'π̄₂':>9} {'P̄₁':>7} {'P̄₂':>7} {'CI':>6}"
        print(header)
        print("─"*82)
        for r in results:
            label  = f"{r.agent1_name} vs {r.agent2_name}"
            ci_bar = "█" * int(r.collusion_index * 8)
            print(
                f"  {label:<34}"
                f"  {r.mean_profit_agent1:>8.1f}"
                f"  {r.mean_profit_agent2:>8.1f}"
                f"  {r.mean_price_agent1:>6.2f}"
                f"  {r.mean_price_agent2:>6.2f}"
                f"  {r.collusion_index:>4.2f} {ci_bar}"
            )
        print("═"*82)

    def export_csv(self, results: List[MatchResult], path: str) -> None:
        """Export raw episode logs to CSV for report tables."""
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "agent1","agent2","episode",
                "mean_profit_1","mean_profit_2",
                "mean_price_1","mean_price_2","collusion_index"
            ])
            for r in results:
                for ep in r.episode_logs:
                    w.writerow([
                        r.agent1_name, r.agent2_name, ep.episode,
                        f"{ep.mean_profit_firm1:.2f}",
                        f"{ep.mean_profit_firm2:.2f}",
                        f"{ep.mean_price_firm1:.4f}",
                        f"{ep.mean_price_firm2:.4f}",
                        f"{r.collusion_index:.4f}",
                    ])
        print(f"CSV saved → {path}") 