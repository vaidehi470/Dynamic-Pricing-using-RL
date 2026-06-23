"""
Test Suite — BertrandPricingEnv
================================
Run with:  python test_bertrand_env.py
All tests are self-contained (no pytest required, though pytest also works).
"""

import sys
import math
import numpy as np

# ── import the environment ──────────────────────────────────────────────
sys.path.insert(0, "/mnt/user-data/outputs")
from bertrand_pricing_env import BertrandPricingEnv

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name: str, condition: bool, detail: str = "") -> None:
    tag = PASS if condition else FAIL
    msg = f"{tag}  {name}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    results.append(condition)

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  BERTRAND PRICING ENV — TEST SUITE")
print("═"*60)

# ── TEST 1: Analytical Nash price ───────────────────────────────────────
print("\n📐 [1] Analytical Nash Equilibrium")
env = BertrandPricingEnv(a=100, b=1.0, d=0.5, marginal_cost=20)
# Expected: (a + b*MC) / (2b + d) = (100 + 20) / 2.5 = 48.0
expected_nash = 48.0
check("Nash price formula",
      math.isclose(env.nash_price, expected_nash, rel_tol=1e-6),
      f"got {env.nash_price:.4f}, expected {expected_nash}")

# Monopoly price: (a + b*MC) / (2b) = 120/2 = 60
expected_mono = 60.0
check("Monopoly price formula",
      math.isclose(env.monopoly_price, expected_mono, rel_tol=1e-6),
      f"got {env.monopoly_price:.4f}, expected {expected_mono}")

# Nash < Monopoly (competition depresses prices vs monopoly)
check("Nash < Monopoly price", env.nash_price < env.monopoly_price)

# ── TEST 2: Action / observation space shapes ───────────────────────────
print("\n📦 [2] Spaces")
n = 30
env = BertrandPricingEnv(n_price_levels=n)
check("Action space is Discrete(30)",   env.action_space.n == n)
check("Obs space shape is (5,)",        env.observation_space.shape == (5,))
check("Obs space dtype float32",        env.observation_space.dtype == np.float32)

# ── TEST 3: reset() ─────────────────────────────────────────────────────
print("\n🔄 [3] reset()")
env = BertrandPricingEnv()
env2 = BertrandPricingEnv()
obs, info = env2.reset(seed=42)
check("obs in observation_space",       env2.observation_space.contains(obs))
check("obs all in [0,1]",              bool(np.all(obs >= 0) and np.all(obs <= 1)))
check("info has nash_price key",       "nash_price" in info)
check("info has p1 key",              "p1" in info)
check("step counter reset to 0",       info["step"] == 0)

# ── TEST 4: step() ──────────────────────────────────────────────────────
print("\n🚶 [4] step()")
env3 = BertrandPricingEnv(max_steps=10, noise_std=0)
obs, _ = env3.reset(seed=0)

# Mid-range action
mid_action = env3.n_price_levels // 2
obs2, reward, terminated, truncated, info2 = env3.step(mid_action)

check("obs after step in space",       env3.observation_space.contains(obs2))
check("reward is float",               isinstance(reward, float))
check("reward >= 0",                   reward >= 0.0,  f"reward={reward:.4f}")
check("not terminated after 1 step",   not terminated)
check("not truncated after 1 step",    not truncated)
check("step counter incremented",      info2["step"] == 1)

# ── TEST 5: Truncation at max_steps ─────────────────────────────────────
print("\n⏱  [5] Episode truncation")
env4 = BertrandPricingEnv(max_steps=5, noise_std=0)
env4.reset(seed=1)
truncated = False
for _ in range(5):
    _, _, _, truncated, _ = env4.step(0)
check("Truncated exactly at max_steps=5", truncated)

# ── TEST 6: Deterministic mode (noise_std=0) ────────────────────────────
print("\n🎲 [6] Determinism (noise_std=0)")
env5 = BertrandPricingEnv(noise_std=0)
env5.reset(seed=7)
_, r1, _, _, i1 = env5.step(15)
env5.reset(seed=7)
_, r2, _, _, i2 = env5.step(15)
check("Same seed → same reward",
      math.isclose(r1, r2, rel_tol=1e-9),
      f"r1={r1:.6f} r2={r2:.6f}")
check("Same seed → same p1",
      math.isclose(i1["p1"], i2["p1"], rel_tol=1e-9))

# ── TEST 7: Price grid configurability ──────────────────────────────────
print("\n⚙️  [7] Configurability")
env6 = BertrandPricingEnv(a=200, b=2.0, d=1.0, marginal_cost=30, n_price_levels=50)
check("Custom a=200 stored",           env6.a == 200)
check("Custom MC=30 stored",          env6.mc == 30)
check("Price grid has 50 levels",     len(env6.price_grid) == 50)
check("p_min equals MC",              math.isclose(env6.p_min, 30.0, rel_tol=1e-6))
check("Price grid min == MC",         math.isclose(env6.price_grid[0], 30.0, rel_tol=1e-6))

# ── TEST 8: Observation normalisation bounds ────────────────────────────
print("\n📏 [8] Observation normalisation")
env7 = BertrandPricingEnv(noise_std=0, max_steps=100)
env7.reset(seed=42)
all_in_bounds = True
for _ in range(100):
    action = env7.action_space.sample()
    obs, _, _, trunc, _ = env7.step(action)
    if not (np.all(obs >= -0.01) and np.all(obs <= 1.01)):
        all_in_bounds = False
        break
    if trunc:
        break
check("All obs in [0,1] over full episode", all_in_bounds)

# ── TEST 9: Profit is non-negative at Nash ──────────────────────────────
print("\n💰 [9] Profits")
env8 = BertrandPricingEnv(noise_std=0)
env8.reset(seed=0)
# Find Nash price index
nash_idx = int(np.argmin(np.abs(env8.price_grid - env8.nash_price)))
_, reward_nash, _, _, info_nash = env8.step(nash_idx)
check("Profit at Nash >= 0",
      info_nash["profit_firm1"] >= 0,
      f"π1={info_nash['profit_firm1']:.2f}")
check("Nash gap small at Nash action",
      info_nash["nash_gap_firm1"] < (env8.p_max - env8.p_min) / env8.n_price_levels + 0.1,
      f"gap={info_nash['nash_gap_firm1']:.4f}")

# ── TEST 10: render() doesn't crash ────────────────────────────────────
print("\n🖥  [10] render()")
env9 = BertrandPricingEnv(render_mode="human", noise_std=0)
env9.reset(seed=0)
try:
    env9.step(10)
    check("render() executes without error", True)
except Exception as e:
    check("render() executes without error", False, str(e))

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
passed = sum(results)
total  = len(results)
print(f"  RESULTS: {passed}/{total} tests passed")
if passed == total:
    print("  🎉 All tests passed — environment is production-ready!")
else:
    failed = [i+1 for i, r in enumerate(results) if not r]
    print(f"  ⚠️  Failed test indices: {failed}")
print("═"*60 + "\n")
