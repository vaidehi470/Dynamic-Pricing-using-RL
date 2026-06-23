"""
Bertrand Competitive Pricing Environment
Week 2 — Dynamic Pricing with Reinforcement Learning

A custom OpenAI Gymnasium environment that simulates an oligopolistic
market where competing firms set prices each period. Built to the spec
in the Week 2 resource sheet.

Author  : Generated for Week 2 project
Requires: gymnasium>=0.26, numpy
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class BertrandPricingEnv(gym.Env):
    """
    Bertrand Oligopoly Pricing Environment
    =======================================
    Two firms (agents) simultaneously set prices each period.
    Demand follows a linear model: Q_i = a - b*P_i + d*(P_j - P_i)

    Parameters
    ----------
    a            : float  — demand intercept (market size)
    b            : float  — own-price sensitivity
    d            : float  — cross-price sensitivity (substitutability)
    marginal_cost: float  — constant MC for both firms (symmetric)
    n_price_levels: int   — discretisation of [MC, a/b] into this many steps
    max_steps    : int    — episode length (number of pricing periods)
    noise_std    : float  — demand shock std-dev (set 0 for deterministic)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        a: float = 100.0,
        b: float = 1.0,
        d: float = 0.5,
        marginal_cost: float = 20.0,
        n_price_levels: int = 30,        # ≥ 20 per resource-sheet warning
        max_steps: int = 200,
        noise_std: float = 2.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # --- market parameters (all configurable — no hard-coding) ---
        self.a = a
        self.b = b
        self.d = d
        self.mc = marginal_cost
        self.noise_std = noise_std
        self.max_steps = max_steps
        self.render_mode = render_mode

        # --- price grid -----------------------------------------------
        # Competitive (Bertrand Nash) price = MC  →  upper bound = monopoly price
        self.p_min = marginal_cost                    # lower bound: never below MC
        self.p_max = (a + marginal_cost * b) / (2 * b)  # unconstrained monopoly price
        self.n_price_levels = n_price_levels

        # Evenly spaced price levels
        self.price_grid = np.linspace(self.p_min, self.p_max, n_price_levels)

        # --- action & observation spaces ------------------------------
        # Action: choose one of n_price_levels for firm 1
        self.action_space = spaces.Discrete(n_price_levels)

        # Observation (normalised to [0,1]):
        #   [own_last_price, rival_last_price, own_last_profit, rival_last_profit,
        #    steps_remaining_fraction]
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
            dtype=np.float32,
        )

        # --- episode state --------------------------------------------
        self._step = 0
        self._p1 = self.p_min   # firm 1 price (our agent)
        self._p2 = self.p_min   # firm 2 price (opponent / rule-based)
        self._pi1 = 0.0
        self._pi2 = 0.0

        # Nash & monopoly benchmarks (computed analytically)
        self.nash_price = self._compute_nash_price()
        self.monopoly_price = self._compute_monopoly_price()

    # ------------------------------------------------------------------ #
    #  Core gym interface                                                  #
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._step = 0

        # Both firms start at Nash price ± small noise to avoid trivial convergence
        jitter = (self.p_max - self.p_min) * 0.05
        self._p1 = float(np.clip(
            self.nash_price + self.np_random.uniform(-jitter, jitter),
            self.p_min, self.p_max
        ))
        self._p2 = float(np.clip(
            self.nash_price + self.np_random.uniform(-jitter, jitter),
            self.p_min, self.p_max
        ))
        self._pi1, self._pi2 = self._compute_profits(self._p1, self._p2)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        One pricing period.

        action : int — index into self.price_grid for firm 1's price
        Firm 2 plays a simple best-response rule (swap out for a second RL agent
        in multi-agent mode).
        """
        assert self.action_space.contains(action), f"Invalid action {action}"

        # Firm 1 sets price from grid
        p1_new = float(self.price_grid[action])

        # Firm 2: best-response to firm 1's last price (rule-based opponent)
        p2_new = self._firm2_best_response(self._p1)

        # Demand shocks
        shock1 = self.np_random.normal(0, self.noise_std) if self.noise_std > 0 else 0.0
        shock2 = self.np_random.normal(0, self.noise_std) if self.noise_std > 0 else 0.0

        # Profits
        pi1, pi2 = self._compute_profits(p1_new, p2_new, shock1, shock2)

        # Update state
        self._p1, self._p2 = p1_new, p2_new
        self._pi1, self._pi2 = pi1, pi2
        self._step += 1

        # Reward: firm 1's profit (normalised by monopoly benchmark)
        max_possible = self._max_possible_profit()
        reward = float(pi1 / max_possible) if max_possible > 0 else float(pi1)

        terminated = False
        truncated = self._step >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        nash_gap = abs(self._p1 - self.nash_price)
        print(
            f"Step {self._step:>4d} | "
            f"P1={self._p1:6.2f}  P2={self._p2:6.2f} | "
            f"π1={self._pi1:8.2f}  π2={self._pi2:8.2f} | "
            f"Nash gap={nash_gap:.2f}"
        )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    #  Economic model                                                      #
    # ------------------------------------------------------------------ #
    def _demand(self, p_own: float, p_rival: float, shock: float = 0.0) -> float:
        """Linear demand: Q = a - b*P_own + d*(P_rival - P_own) + shock."""
        q = self.a - self.b * p_own + self.d * (p_rival - p_own) + shock
        return max(0.0, q)   # non-negative quantity

    def _compute_profits(
        self,
        p1: float,
        p2: float,
        shock1: float = 0.0,
        shock2: float = 0.0,
    ) -> Tuple[float, float]:
        q1 = self._demand(p1, p2, shock1)
        q2 = self._demand(p2, p1, shock2)
        pi1 = (p1 - self.mc) * q1
        pi2 = (p2 - self.mc) * q2
        return pi1, pi2

    def _compute_nash_price(self) -> float:
        """
        Analytical Bertrand Nash equilibrium price (symmetric duopoly).
        Derived from FOC: P* = (a + b*MC) / (2b + d)
        """
        return (self.a + self.b * self.mc) / (2 * self.b + self.d)

    def _compute_monopoly_price(self) -> float:
        """Unconstrained monopoly price: P_m = (a + b*MC) / (2b)."""
        return (self.a + self.b * self.mc) / (2 * self.b)

    def _firm2_best_response(self, p1: float) -> float:
        """
        Firm 2 best-response function.
        BR(p1) = (a + b*MC + d*p1) / (2b + d)
        Snapped to nearest price grid point.
        """
        br = (self.a + self.b * self.mc + self.d * p1) / (2 * self.b + self.d)
        br = float(np.clip(br, self.p_min, self.p_max))
        idx = int(np.argmin(np.abs(self.price_grid - br)))
        return float(self.price_grid[idx])

    def _max_possible_profit(self) -> float:
        """Monopoly profit — used as normalisation ceiling for reward."""
        pm = self.monopoly_price
        q_m = self._demand(pm, pm)
        return max(1.0, (pm - self.mc) * q_m)

    # ------------------------------------------------------------------ #
    #  Observations & info                                                 #
    # ------------------------------------------------------------------ #
    def _normalise(self, value: float, low: float, high: float) -> float:
        """Min-max normalise to [0, 1]."""
        if high == low:
            return 0.0
        return float(np.clip((value - low) / (high - low), 0.0, 1.0))

    def _get_obs(self) -> np.ndarray:
        """
        Five-dimensional normalised observation:
          [own_price, rival_price, own_profit, rival_profit, time_fraction]
        """
        max_profit = self._max_possible_profit()
        obs = np.array([
            self._normalise(self._p1,  self.p_min, self.p_max),
            self._normalise(self._p2,  self.p_min, self.p_max),
            self._normalise(self._pi1, 0.0, max_profit),
            self._normalise(self._pi2, 0.0, max_profit),
            self._normalise(self._step, 0, self.max_steps),
        ], dtype=np.float32)
        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step"          : self._step,
            "p1"            : self._p1,
            "p2"            : self._p2,
            "profit_firm1"  : self._pi1,
            "profit_firm2"  : self._pi2,
            "nash_price"    : self.nash_price,
            "monopoly_price": self.monopoly_price,
            "nash_gap_firm1": abs(self._p1 - self.nash_price),
        }
