"""
Week 4 — Tabular Q-Learning Agent
===================================
Hand-coded from scratch — no Stable-Baselines3, no shortcuts.

This file implements:
  1. QLearningAgent   — the full tabular Q-learning agent
  2. QTrainer         — training loop with logging
  3. EpsilonSchedule  — configurable ε-greedy decay

Theory background
-----------------
Q-learning (Watkins & Dayan, 1992) is an off-policy temporal-difference
algorithm. It maintains a table Q[s, a] — the expected discounted future
reward from taking action a in state s and acting optimally thereafter.

The Bellman update (applied after every step):

    Q[s, a] ← Q[s, a] + α · (r + γ · max_a' Q[s', a'] − Q[s, a])

where:
    α  = learning rate  (how fast we update)
    γ  = discount factor (how much we value future rewards)
    r  = reward received this step
    s' = next state
    max_a' Q[s', a'] = best Q-value reachable from next state (greedy)

The term (r + γ · max_a' Q[s', a'] − Q[s, a]) is called the TD error —
the surprise between what we expected and what we actually got.

Resource-sheet warnings addressed
----------------------------------
✓ Reward is already normalised to [0,1] in the environment (π/max_profit)
✓ ε starts at 1.0, decays slowly to 0.05 over 80% of training episodes
✓ State space is discretised to keep Q-table tractable
✓ All hyperparameters configurable and logged for the mid-project review
"""

import numpy as np
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
#  Hyperparameter dataclass  (makes logging and tuning clean)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class QHyperparams:
    """
    All hyperparameters for the Q-learning agent in one place.

    alpha      : learning rate — how large each Bellman update step is.
                 Too high → Q-values oscillate. Too low → very slow learning.
                 Typical range: 0.05–0.3
    gamma      : discount factor — weight of future vs immediate reward.
                 0 = myopic (only care about now).
                 1 = infinite horizon (care equally about all future steps).
                 Typical range: 0.90–0.99 for episodic pricing tasks.
    eps_start  : initial exploration rate. Start at 1.0 so the agent
                 explores the full price grid before exploiting.
    eps_end    : minimum exploration rate. Never go fully greedy —
                 0.05 means 5% random actions even after convergence.
    eps_decay_frac : fraction of total episodes over which ε decays.
                 Resource sheet: use 0.80 (decay over 80% of training).
    n_bins     : number of bins for state discretisation.
                 The raw 5-dim obs is continuous; we bucket each dim
                 into n_bins intervals to keep the Q-table finite.
    """
    alpha          : float = 0.10
    gamma          : float = 0.95
    eps_start      : float = 1.00
    eps_end        : float = 0.05
    eps_decay_frac : float = 0.80
    n_bins         : int   = 10


# ═══════════════════════════════════════════════════════════════════════
#  Epsilon schedule
# ═══════════════════════════════════════════════════════════════════════

class EpsilonSchedule:
    """
    Linear epsilon decay from eps_start to eps_end.

    Decay happens over (decay_frac × total_episodes) episodes.
    After that, epsilon stays flat at eps_end for the remainder
    of training — the agent exploits its learned policy.

    Why linear (not exponential)?
    Linear decay is more predictable and easier to reason about
    during the mid-project review. Exponential is covered in Week 5.
    """

    def __init__(
        self,
        eps_start     : float,
        eps_end       : float,
        total_episodes: int,
        decay_frac    : float = 0.80,
    ):
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.decay_over = int(total_episodes * decay_frac)
        self._step      = 0

    def value(self) -> float:
        if self._step >= self.decay_over:
            return self.eps_end
        frac = self._step / self.decay_over
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def step(self) -> float:
        eps = self.value()
        self._step += 1
        return eps

    def reset(self) -> None:
        self._step = 0


# ═══════════════════════════════════════════════════════════════════════
#  State discretiser
# ═══════════════════════════════════════════════════════════════════════

class StateDiscretiser:
    """
    Converts the continuous 5-dim [0,1] observation into a discrete
    integer tuple usable as a Q-table key.

    Each dimension is binned into n_bins equal intervals.
    The full state space has n_bins^5 possible states.

    With n_bins=10 and n_actions=30:
      Q-table entries = 10^5 × 30 = 3,000,000
      Memory ≈ 3M × 8 bytes ≈ 24 MB  (tractable)

    Resource-sheet note: if memory is an issue, reduce n_bins or
    drop the time-fraction dimension (least informative for pricing).
    """

    def __init__(self, n_bins: int, obs_dim: int = 5):
        self.n_bins  = n_bins
        self.obs_dim = obs_dim
        # Bin edges: n_bins+1 edges create n_bins intervals over [0,1]
        self.edges   = np.linspace(0.0, 1.0, n_bins + 1)

    def encode(self, obs: np.ndarray) -> tuple:
        """Map a [0,1]^5 observation to a tuple of bin indices."""
        bins = []
        for i in range(self.obs_dim):
            # np.searchsorted gives the bin index; clip to [0, n_bins-1]
            idx = int(np.searchsorted(self.edges, obs[i], side='right')) - 1
            bins.append(int(np.clip(idx, 0, self.n_bins - 1)))
        return tuple(bins)

    @property
    def n_states(self) -> int:
        return self.n_bins ** self.obs_dim


# ═══════════════════════════════════════════════════════════════════════
#  Q-Learning Agent
# ═══════════════════════════════════════════════════════════════════════

class QLearningAgent:
    """
    Tabular Q-Learning agent for the BertrandPricingEnv.

    Core algorithm (Watkins & Dayan, 1992)
    ----------------------------------------
    Every step:
      1. Observe state s (discretised)
      2. Choose action a via ε-greedy:
           - with prob ε: random action (explore)
           - with prob 1-ε: argmax_a Q[s,a] (exploit)
      3. Execute a, receive reward r, observe next state s'
      4. Bellman update:
           Q[s,a] ← Q[s,a] + α(r + γ·max_a' Q[s',a'] - Q[s,a])
      5. s ← s'

    The Q-table is a numpy array of shape (n_bins^5, n_actions),
    stored as a dict keyed by state tuple for memory efficiency.
    """

    def __init__(
        self,
        n_actions  : int,
        hp         : QHyperparams,
        seed       : int = 42,
    ):
        self.name      = "Q-Learning"
        self.n_actions = n_actions
        self.hp        = hp
        self.rng       = np.random.default_rng(seed)
        self.disc      = StateDiscretiser(hp.n_bins)

        # Q-table as defaultdict: missing states initialise to zeros
        # (optimistic initialisation can also be tried: fill with 1.0)
        self._Q : dict = {}

        # Training counters
        self.total_steps    = 0
        self.total_episodes = 0

        # For act() — epsilon is set externally by QTrainer during training
        self._epsilon = hp.eps_start

    # ── Q-table access ──────────────────────────────────────────────

    def _get_q(self, state: tuple) -> np.ndarray:
        """Return Q-values for a state, initialising to 0 if unseen."""
        if state not in self._Q:
            self._Q[state] = np.zeros(self.n_actions, dtype=np.float64)
        return self._Q[state]

    # ── Policy ──────────────────────────────────────────────────────

    def act(self, obs: np.ndarray, info: dict, training: bool = True) -> int:
        """
        ε-greedy action selection.

        During training: explore with probability ε, exploit otherwise.
        During evaluation (training=False): always greedy (ε=0).
        """
        state = self.disc.encode(obs)

        if training and self.rng.random() < self._epsilon:
            return int(self.rng.integers(0, self.n_actions))  # explore
        else:
            return int(np.argmax(self._get_q(state)))          # exploit

    # ── Bellman update ───────────────────────────────────────────────

    def update(
        self,
        obs       : np.ndarray,
        action    : int,
        reward    : float,
        next_obs  : np.ndarray,
        terminated: bool,
        truncated : bool,
    ) -> float:
        """
        Apply the Q-learning Bellman update.

        Returns the TD error (useful for logging convergence).

        Bellman equation:
          Q[s,a] ← Q[s,a] + α · TD_error
          TD_error = r + γ · max_a' Q[s',a'] - Q[s,a]

        If the episode ended (terminated or truncated), the future
        value term γ·max_a' Q[s',a'] is dropped (= 0), because there
        is no next state to transition to.
        """
        state      = self.disc.encode(obs)
        next_state = self.disc.encode(next_obs)

        q_current  = self._get_q(state)[action]

        if terminated or truncated:
            # Terminal state: no future rewards
            td_target = reward
        else:
            # Bellman target: immediate reward + discounted best future value
            td_target = reward + self.hp.gamma * np.max(self._get_q(next_state))

        td_error = td_target - q_current

        # In-place update of Q[s, a]
        self._get_q(state)[action] += self.hp.alpha * td_error

        self.total_steps += 1
        return float(td_error)

    def reset(self) -> None:
        """Called at the start of each episode (no internal state to clear)."""
        self.total_episodes += 1

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save Q-table + hyperparams to a .npz file."""
        keys   = list(self._Q.keys())
        values = np.array([self._Q[k] for k in keys])
        np.savez_compressed(
            path,
            keys   = np.array(keys,   dtype=object),
            values = values,
            hp     = np.array([json.dumps(asdict(self.hp))]),
        )
        print(f"Q-table saved → {path}.npz  ({len(keys)} states populated)")

    def load(self, path: str) -> None:
        """Load a previously saved Q-table."""
        data = np.load(path, allow_pickle=True)
        keys   = data['keys']
        values = data['values']
        self._Q = {tuple(k): v for k, v in zip(keys, values)}
        print(f"Q-table loaded ← {path}  ({len(self._Q)} states)")

    @property
    def q_table_size(self) -> int:
        return len(self._Q)
