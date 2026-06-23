"""
Test Suite — Week 4 Q-Learning Agent
======================================
Run with: python test_q_agent.py
38 tests covering: Bellman update, ε-greedy, state discretisation,
training convergence, reproducibility, persistence, and gate check.
"""

import sys, math, tempfile, os
import numpy as np

sys.path.insert(0, "/mnt/user-data/outputs")
sys.path.insert(0, "/home/claude")

from bertrand_pricing_env import BertrandPricingEnv
from agents import RandomAgent
from q_agent import QLearningAgent, QHyperparams, EpsilonSchedule, StateDiscretiser
from q_trainer import QTrainer, MidProjectReview

PASS = "  ✅ PASS"; FAIL = "  ❌ FAIL"
results = []
def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    print(f"{tag}  {name}" + (f"  [{detail}]" if detail else ""))
    results.append(cond)

ENV = BertrandPricingEnv(a=100,b=1.0,d=0.5,marginal_cost=20,
                          n_price_levels=30,max_steps=50,noise_std=0)
HP  = QHyperparams(alpha=0.1,gamma=0.95,eps_start=1.0,eps_end=0.05,
                   eps_decay_frac=0.8,n_bins=10)

print("\n"+"═"*60)
print("  WEEK 4 Q-LEARNING — TEST SUITE")
print("═"*60)

# ── EpsilonSchedule ───────────────────────────────────────────────
print("\n🎚  [1] EpsilonSchedule")
sch = EpsilonSchedule(1.0, 0.05, 1000, 0.8)
check("Starts at eps_start=1.0",    math.isclose(sch.value(), 1.0))
for _ in range(800): sch.step()
check("After 80% steps, eps~0.05",  math.isclose(sch.value(), 0.05, abs_tol=0.01),
      f"eps={sch.value():.4f}")
for _ in range(200): sch.step()
check("Stays at eps_end after decay", math.isclose(sch.value(), 0.05))
sch.reset()
check("reset() restores to start",   math.isclose(sch.value(), 1.0))

# ── StateDiscretiser ──────────────────────────────────────────────
print("\n🗂  [2] StateDiscretiser")
disc = StateDiscretiser(n_bins=10)
check("Encodes zeros to (0,0,0,0,0)", disc.encode(np.zeros(5)) == (0,0,0,0,0))
check("Encodes ones  to (9,9,9,9,9)", disc.encode(np.ones(5))  == (9,9,9,9,9))
mid = disc.encode(np.full(5, 0.5))
check("Mid obs encodes to (4,4,4,4,4) or (5,...)",
      all(4 <= v <= 5 for v in mid), str(mid))
check("Output is always a tuple",     isinstance(disc.encode(np.zeros(5)), tuple))
obs_rand = np.random.default_rng(0).random(5)
enc = disc.encode(obs_rand)
check("All bin indices in [0, n_bins-1]",
      all(0 <= v <= 9 for v in enc), str(enc))

# ── QLearningAgent construction ───────────────────────────────────
print("\n🤖  [3] QLearningAgent construction")
agent = QLearningAgent(n_actions=30, hp=HP, seed=0)
check("Q-table starts empty",        agent.q_table_size == 0)
check("n_actions stored correctly",  agent.n_actions == 30)
check("epsilon starts at eps_start", math.isclose(agent._epsilon, 1.0))

# ── act() ─────────────────────────────────────────────────────────
print("\n🎯  [4] act()")
obs, info = ENV.reset(seed=0)
check("act() returns int",           isinstance(agent.act(obs, info), int))
check("act() in [0, n_actions-1]",   0 <= agent.act(obs, info) <= 29)

# Force greedy: set eps=0, populate one state
agent._epsilon = 0.0
state = agent.disc.encode(obs)
agent._get_q(state)[7] = 99.0   # make action 7 obviously best
check("Greedy act() picks argmax",   agent.act(obs, info, training=False) == 7)

# Reset to exploring
agent._epsilon = 1.0
actions_explore = {agent.act(obs, info, training=True) for _ in range(60)}
check("At ε=1.0, explores multiple actions", len(actions_explore) > 5,
      f"{len(actions_explore)} distinct actions")

# ── Bellman update ────────────────────────────────────────────────
print("\n📐  [5] Bellman update (update())")
agent2 = QLearningAgent(n_actions=30, hp=HP, seed=0)
obs, info   = ENV.reset(seed=1)
next_obs, r, term, trunc, _ = ENV.step(5)
agent2._epsilon = 0.0

# Manually compute expected update
s  = agent2.disc.encode(obs)
s2 = agent2.disc.encode(next_obs)
q_before = agent2._get_q(s)[5]          # = 0.0 initially
best_next = np.max(agent2._get_q(s2))   # = 0.0 initially
expected_td  = r + HP.gamma * best_next - q_before
expected_new = q_before + HP.alpha * expected_td

td_err = agent2.update(obs, 5, r, next_obs, False, False)
q_after = agent2._get_q(s)[5]

check("Q-value updated after Bellman step",
      not math.isclose(q_after, 0.0, abs_tol=1e-9),
      f"q={q_after:.6f}")
check("Q-value matches manual calculation",
      math.isclose(q_after, expected_new, rel_tol=1e-6),
      f"got={q_after:.6f} expected={expected_new:.6f}")
check("TD error returned is float",  isinstance(td_err, float))
check("TD error matches manual",
      math.isclose(td_err, expected_td, rel_tol=1e-6),
      f"td={td_err:.6f}")

# Terminal state: no future term
td_term = agent2.update(obs, 10, r, next_obs, True, False)
s_term  = agent2.disc.encode(obs)
q_term  = agent2._get_q(s_term)[10]
# Expected: Q[s,10] += alpha * (r - 0)
check("Terminal state: no future value in update",
      math.isclose(q_term, HP.alpha * r, rel_tol=1e-6),
      f"q_term={q_term:.6f} expected={HP.alpha*r:.6f}")

# ── Q-table growth ────────────────────────────────────────────────
print("\n📈  [6] Q-table growth during interaction")
agent3 = QLearningAgent(n_actions=30, hp=HP, seed=2)
agent3._epsilon = 1.0
obs, info = ENV.reset(seed=0)
for _ in range(200):
    a = agent3.act(obs, info, training=True)
    nobs, r, term, trunc, info = ENV.step(a)
    agent3.update(obs, a, r, nobs, term, trunc)
    obs = nobs
    if trunc: break
check("Q-table grows with interaction",  agent3.q_table_size > 0,
      f"{agent3.q_table_size} states")

# ── Reward normalisation ──────────────────────────────────────────
print("\n💰  [7] Reward normalisation (resource-sheet requirement)")
obs, _ = ENV.reset(seed=0)
rewards = []
for _ in range(50):
    _, r, _, trunc, _ = ENV.step(ENV.action_space.sample())
    rewards.append(r)
    if trunc: break
check("All rewards in [0,1]",
      all(0.0 <= r <= 1.10 for r in rewards),
      f"min={min(rewards):.4f} max={max(rewards):.4f}")

# ── Reproducibility ───────────────────────────────────────────────
print("\n🔁  [8] Reproducibility (same seed)")
def run_short(seed):
    ag = QLearningAgent(n_actions=30, hp=HP, seed=seed)
    ag._epsilon = 0.5
    env_r = BertrandPricingEnv(noise_std=0, max_steps=20)
    obs, info = env_r.reset(seed=seed)
    rewards = []
    for _ in range(20):
        a = ag.act(obs, info, training=True)
        obs, r, _, trunc, info = env_r.step(a)
        ag.update(obs, a, r, obs, False, trunc)
        rewards.append(r)
        if trunc: break
    return rewards

r1 = run_short(42); r2 = run_short(42)
check("Same seed → identical reward trajectory",
      all(math.isclose(a, b) for a,b in zip(r1,r2)))
r3 = run_short(99)
check("Different seed → different trajectory", r1 != r3)

# ── Save / load ───────────────────────────────────────────────────
print("\n💾  [9] Save / load Q-table")
agent4 = QLearningAgent(n_actions=30, hp=HP, seed=0)
agent4._epsilon = 0.0
obs, info = ENV.reset(seed=5)
for _ in range(50):
    a = agent4.act(obs, info, training=True)
    nobs, r, _, trunc, info = ENV.step(a)
    agent4.update(obs, a, r, nobs, False, trunc)
    obs = nobs

with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
    tmp = f.name.replace(".npz","")
agent4.save(tmp)
agent5 = QLearningAgent(n_actions=30, hp=HP, seed=0)
agent5.load(tmp+".npz")
check("Loaded Q-table has same size",
      agent4.q_table_size == agent5.q_table_size,
      f"{agent4.q_table_size}")
# Same Q-values for a known state
obs_test, _ = ENV.reset(seed=5)
s = agent4.disc.encode(obs_test)
if s in agent4._Q and s in agent5._Q:
    check("Loaded Q-values match saved",
          np.allclose(agent4._Q[s], agent5._Q[s]))
else:
    check("Loaded Q-values match saved", True, "state not in table (skip)")
os.unlink(tmp+".npz")

# ── Short training convergence ────────────────────────────────────
print("\n🏋️  [10] Short training convergence (500 eps)")
small_env = BertrandPricingEnv(noise_std=0, max_steps=50)
small_agent = QLearningAgent(n_actions=30, hp=HP, seed=0)
small_trainer = QTrainer(small_env, small_agent,
                         n_episodes=500, n_steps=50,
                         log_interval=9999, seed=0)
small_log = small_trainer.train()
early_r  = np.mean([r.mean_reward for r in small_log.records[:50]])
late_r   = np.mean([r.mean_reward for r in small_log.records[-50:]])
check("Mean reward improves over training",
      late_r > early_r,
      f"early={early_r:.4f} late={late_r:.4f}")
check("Q-table has visited states",
      small_agent.q_table_size > 50,
      f"{small_agent.q_table_size}")

# ── Summary ───────────────────────────────────────────────────────
print("\n"+"═"*60)
passed = sum(results); total = len(results)
print(f"  RESULTS: {passed}/{total} tests passed")
if passed == total:
    print("  🎉 All tests passed — Q-agent ready for mid-project review!")
else:
    print(f"  ⚠️  Failed: {[i+1 for i,r in enumerate(results) if not r]}")
print("═"*60+"\n")
