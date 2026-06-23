"""
Week 3 — Rule-Based Baseline Agents
=====================================
Four strategy agents that compete inside BertrandPricingEnv.

Each agent exposes a single method:
    act(obs, info) -> int   (action index into env.price_grid)

Agents implemented
------------------
1. AlwaysNashAgent      — always plays the Nash equilibrium price
2. AlwaysColludeAgent   — always plays the monopoly (collusion) price
3. TitForTatAgent       — mirrors rival's last move; cooperates first
4. RandomAgent          — uniformly random price each period (seeded)

Design notes
------------
• All agents are stateless between episodes — call reset() before each episode.
• act() takes the raw info dict (not the normalised obs) so strategies can
  reason in real price units rather than [0,1] space.
• n_price_levels is passed at construction so every agent shares the same
  discrete action space as the environment.
"""

import numpy as np
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════════════════
#  Base class
# ═══════════════════════════════════════════════════════════════════════

class BaseAgent(ABC):
    """Abstract base — all rule-based agents inherit from this."""

    def __init__(self, name: str, price_grid: np.ndarray):
        self.name = name
        self.price_grid = price_grid
        self.n = len(price_grid)

    def _snap(self, price: float) -> int:
        """Return the grid index closest to a target price."""
        return int(np.argmin(np.abs(self.price_grid - price)))

    def reset(self) -> None:
        """Called at the start of every episode."""
        pass

    @abstractmethod
    def act(self, obs: np.ndarray, info: dict) -> int:
        """Return an action index given the current observation + info."""


# ═══════════════════════════════════════════════════════════════════════
#  Agent 1 — Always Nash
# ═══════════════════════════════════════════════════════════════════════

class AlwaysNashAgent(BaseAgent):
    """
    Plays the Bertrand Nash equilibrium price every single period.

    Why useful as a baseline
    ------------------------
    This is the theoretical competitive benchmark. An RL agent that
    cannot consistently beat Always-Nash has learned nothing useful.
    It also verifies the environment: the Nash price should yield
    near-zero economic profit over the long run when both firms play it.
    """

    def __init__(self, price_grid: np.ndarray, nash_price: float):
        super().__init__("Always-Nash", price_grid)
        self._action = self._snap(nash_price)
        self.nash_price = nash_price

    def act(self, obs: np.ndarray, info: dict) -> int:
        return self._action


# ═══════════════════════════════════════════════════════════════════════
#  Agent 2 — Always Collude
# ═══════════════════════════════════════════════════════════════════════

class AlwaysColludeAgent(BaseAgent):
    """
    Always plays the joint-monopoly (collusion) price.

    Why useful as a baseline
    ------------------------
    This is the profit ceiling — if both firms cooperate perfectly,
    each earns monopoly profit. In practice this is unstable:
    one firm always has an incentive to undercut.

    Resource-sheet warning: always-collude appears to 'win' in the
    first few rounds. Run 1,000+ steps so defection incentives emerge.
    """

    def __init__(self, price_grid: np.ndarray, monopoly_price: float):
        super().__init__("Always-Collude", price_grid)
        self._action = self._snap(monopoly_price)
        self.monopoly_price = monopoly_price

    def act(self, obs: np.ndarray, info: dict) -> int:
        return self._action


# ═══════════════════════════════════════════════════════════════════════
#  Agent 3 — Tit-for-Tat
# ═══════════════════════════════════════════════════════════════════════

class TitForTatAgent(BaseAgent):
    """
    Classic Tit-for-Tat from Axelrod (1980).

    Rules
    -----
    1. Round 1: cooperate (play monopoly / collude price).
    2. Every subsequent round: mirror the rival's previous price exactly
       (snap to nearest grid point).

    Properties (Axelrod's four)
    ---------------------------
    Nice        — never defects first
    Retaliating — immediately punishes undercutting
    Forgiving   — returns to cooperation the moment rival does
    Clear       — trivially simple, opponent can predict it

    Why useful as a baseline
    ------------------------
    TFT is the strongest rule-based strategy in repeated prisoner's
    dilemma tournaments. An RL agent should eventually learn to exploit
    TFT's niceness while avoiding its punishment.
    """

    def __init__(
        self,
        price_grid: np.ndarray,
        monopoly_price: float,
        cooperate_threshold: float = 0.0,
    ):
        super().__init__("Tit-for-Tat", price_grid)
        self.monopoly_price = monopoly_price
        self._cooperate_action = self._snap(monopoly_price)
        self._last_rival_action: int | None = None

    def reset(self) -> None:
        self._last_rival_action = None

    def act(self, obs: np.ndarray, info: dict) -> int:
        # First move: cooperate (play collusion price)
        if self._last_rival_action is None:
            rival_price = info.get("p2", self.monopoly_price)
            self._last_rival_action = self._snap(rival_price)
            return self._cooperate_action

        # All subsequent moves: copy rival's last price
        action = self._last_rival_action

        # Update memory with rival's current price
        rival_price = info.get("p2", self.price_grid[self._last_rival_action])
        self._last_rival_action = self._snap(rival_price)

        return action


# ═══════════════════════════════════════════════════════════════════════
#  Agent 4 — Random
# ═══════════════════════════════════════════════════════════════════════

class RandomAgent(BaseAgent):
    """
    Uniformly random price selection each period.

    Why useful as a baseline
    ------------------------
    The random agent is the floor. An RL agent that fails to beat a
    random policy has a training or reward-shaping problem.

    Resource-sheet warning: always seed the random agent so results
    are reproducible. The seed is logged in every tournament record.
    """

    def __init__(self, price_grid: np.ndarray, seed: int = 42):
        super().__init__("Random", price_grid)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        # Re-seed on every episode reset for episode-level reproducibility
        self._rng = np.random.default_rng(self.seed)

    def act(self, obs: np.ndarray, info: dict) -> int:
        return int(self._rng.integers(0, self.n))
