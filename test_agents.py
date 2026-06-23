"""
Test Suite — Week 3 Rule-Based Agents + Tournament
====================================================
Run with:  python test_agents.py
All 30 tests are self-contained (no pytest required).
"""

import sys, math
import numpy as np

sys.path.insert(0, "/mnt/user-data/outputs")
sys.path.insert(0, "/home/claude")
from bertrand_pricing_env import BertrandPricingEnv
from agents import AlwaysNashAgent, AlwaysColludeAgent, TitForTatAgent, RandomAgent
from tournament import Tournament, MatchResult

PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"
results = []

def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    print(f"{tag}  {name}" + (f"  [{detail}]" if detail else ""))
    results.append(cond)

ENV = BertrandPricingEnv(a=100, b=1.0, d=0.5, marginal_cost=20,
                          n_price_levels=30, max_steps=1000, noise_std=0)
GRID = ENV.price_grid
NASH = ENV.nash_price      # 48.0
MONO = ENV.monopoly_price  # 60.0

print("\n" + "═"*60)
print("  WEEK 3 AGENTS — TEST SUITE")
print("═"*60)

# ── Always-Nash ─────────────────────────────────────────────────────
print("\n📐 [1] AlwaysNashAgent")
an = AlwaysNashAgent(GRID, NASH)
check("name is 'Always-Nash'", an.name == "Always-Nash")

obs, info = ENV.reset(seed=0)
action = an.act(obs, info)
chosen_price = GRID[action]
check("Always plays Nash price (±grid resolution)",
      abs(chosen_price - NASH) <= (GRID[1]-GRID[0]),
      f"price={chosen_price:.2f} Nash={NASH}")

# Same action every step
actions = set()
obs, info = ENV.reset(seed=1)
for _ in range(20):
    a = an.act(obs, info)
    obs, _, _, _, info = ENV.step(a)
    actions.add(a)
check("Action never changes across steps", len(actions) == 1)

# ── Always-Collude ───────────────────────────────────────────────────
print("\n🤝 [2] AlwaysColludeAgent")
ac = AlwaysColludeAgent(GRID, MONO)
check("name is 'Always-Collude'", ac.name == "Always-Collude")

obs, info = ENV.reset(seed=0)
action = ac.act(obs, info)
chosen_price = GRID[action]
check("Plays near monopoly price",
      abs(chosen_price - MONO) <= (GRID[1]-GRID[0]),
      f"price={chosen_price:.2f} Mono={MONO}")

check("Collude price > Nash price", chosen_price > NASH)

# ── Tit-for-Tat ─────────────────────────────────────────────────────
print("\n🔄 [3] TitForTatAgent")
tft = TitForTatAgent(GRID, MONO)
check("name is 'Tit-for-Tat'", tft.name == "Tit-for-Tat")

# First move = cooperate (monopoly price)
tft.reset()
obs, info = ENV.reset(seed=0)
a0 = tft.act(obs, info)
check("First move cooperates (monopoly price)",
      abs(GRID[a0] - MONO) <= (GRID[1]-GRID[0]),
      f"price={GRID[a0]:.2f} mono={MONO}")

# After reset → cooperates again
tft.reset()
obs, info = ENV.reset(seed=5)
a_reset = tft.act(obs, info)
check("After reset, first move is cooperate again",
      abs(GRID[a_reset] - MONO) <= (GRID[1]-GRID[0]))

# Mirror move: inject a synthetic info dict so TFT mirroring is isolated
# (the env's built-in firm 2 is a best-response bot, not the TFT rival)
tft.reset()
obs, info = ENV.reset(seed=2)
fake_rival_price = GRID[5]   # a specific grid price we control
fake_info_t0 = dict(info); fake_info_t0["p2"] = fake_rival_price
_  = tft.act(obs, fake_info_t0)          # first move: cooperate; stores fake_rival_price
fake_info_t1 = dict(info); fake_info_t1["p2"] = GRID[10]   # new rival price (will be stored for t+2)
a1 = tft.act(obs, fake_info_t1)
mirrored_price = GRID[a1]
check("Second move mirrors rival's previous price (injected)",
      abs(mirrored_price - fake_rival_price) <= (GRID[1]-GRID[0]),
      f"mirrored={mirrored_price:.2f} rival_t0={fake_rival_price:.2f}")

# ── RandomAgent ─────────────────────────────────────────────────────
print("\n🎲 [4] RandomAgent")
ra = RandomAgent(GRID, seed=42)
check("name is 'Random'", ra.name == "Random")

obs, info = ENV.reset(seed=0)
actions_rand = [ra.act(obs, info) for _ in range(50)]
check("Actions are in valid range [0, n-1]",
      all(0 <= a < len(GRID) for a in actions_rand))
check("Actions are not all the same (truly random)",
      len(set(actions_rand)) > 5)

# Seeded reproducibility
ra1 = RandomAgent(GRID, seed=99)
ra2 = RandomAgent(GRID, seed=99)
obs, info = ENV.reset(seed=0)
acts1 = [ra1.act(obs, info) for _ in range(30)]
acts2 = [ra2.act(obs, info) for _ in range(30)]
check("Same seed → identical action sequence", acts1 == acts2)

# Different seeds → different sequences
ra3 = RandomAgent(GRID, seed=7)
acts3 = [ra3.act(obs, info) for _ in range(30)]
check("Different seeds → different sequences", acts1 != acts3)

# ── Reset contract ───────────────────────────────────────────────────
print("\n🔄 [5] reset() contract")
for Agent, args, name in [
    (AlwaysNashAgent,    (GRID, NASH),  "AlwaysNash"),
    (AlwaysColludeAgent, (GRID, MONO),  "AlwaysCollude"),
    (TitForTatAgent,     (GRID, MONO),  "TitForTat"),
    (RandomAgent,        (GRID, 42),    "Random"),
]:
    try:
        ag = Agent(*args)
        ag.reset()
        check(f"{name}.reset() doesn't crash", True)
    except Exception as e:
        check(f"{name}.reset() doesn't crash", False, str(e))

# ── act() output always valid ────────────────────────────────────────
print("\n📦 [6] act() output validity")
agents_all = [
    AlwaysNashAgent(GRID, NASH),
    AlwaysColludeAgent(GRID, MONO),
    TitForTatAgent(GRID, MONO),
    RandomAgent(GRID, 0),
]
for ag in agents_all:
    ag.reset()
    obs, info = ENV.reset(seed=3)
    all_valid = True
    for _ in range(100):
        a = ag.act(obs, info)
        if not (0 <= a < len(GRID)):
            all_valid = False; break
        obs, _, _, trunc, info = ENV.step(a)
        if trunc: break
    check(f"{ag.name}: all actions in [0,{len(GRID)-1}] over 100 steps", all_valid)

# ── Profit ordering sanity ───────────────────────────────────────────
print("\n💰 [7] Profit ordering (1000 steps, no noise)")
def mean_profit(agent, n_steps=1000):
    agent.reset()
    obs, info = ENV.reset(seed=42)
    profits = []
    for _ in range(n_steps):
        a = agent.act(obs, info)
        obs, _, _, trunc, info = ENV.step(a)
        profits.append(info["profit_firm1"])
        if trunc: break
    return float(np.mean(profits))

pi_nash    = mean_profit(AlwaysNashAgent(GRID, NASH))
pi_collude = mean_profit(AlwaysColludeAgent(GRID, MONO))
pi_random  = mean_profit(RandomAgent(GRID, 42))

check("Collude profit > Nash profit (firm 1 benefits from high price)",
      pi_collude > pi_nash,
      f"collude={pi_collude:.0f}  nash={pi_nash:.0f}")
check("Nash profit > 0 (Nash still profitable vs MC)",
      pi_nash > 0,
      f"nash profit={pi_nash:.0f}")
check("Random profit > 0 on average",
      pi_random > 0,
      f"random profit={pi_random:.0f}")

# ── Tournament smoke test ────────────────────────────────────────────
print("\n🏆 [8] Tournament (smoke test — 3 eps × 50 steps)")
small_env = BertrandPricingEnv(a=100, b=1.0, d=0.5, marginal_cost=20,
                                n_price_levels=30, max_steps=50, noise_std=0)
small_agents = [
    AlwaysNashAgent(small_env.price_grid, small_env.nash_price),
    AlwaysColludeAgent(small_env.price_grid, small_env.monopoly_price),
    TitForTatAgent(small_env.price_grid, small_env.monopoly_price),
    RandomAgent(small_env.price_grid, seed=0),
]
t = Tournament(small_env, small_agents, n_episodes=3, n_steps=50,
               seed=0, verbose=False)
res = t.run()

check("Tournament returns list", isinstance(res, list))
check(f"Correct number of matchups (4P2=12)", len(res) == 12)

check("Each result is MatchResult", all(isinstance(r, MatchResult) for r in res))
check("All matchups have compile() run (collusion_index set)",
      all(0.0 <= r.collusion_index <= 1.0 for r in res))
check("Mean profits are finite floats",
      all(math.isfinite(r.mean_profit_agent1) for r in res))
check("Episode logs populated",
      all(len(r.episode_logs) == 3 for r in res))

# ── CSV export ───────────────────────────────────────────────────────
import tempfile, os, csv
print("\n📄 [9] CSV export")
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
    tmp_path = f.name
t.export_csv(res, tmp_path)
check("CSV file created", os.path.exists(tmp_path))
with open(tmp_path) as f:
    rows = list(csv.reader(f))
check("CSV has header + data rows", len(rows) > 1)
check("CSV header has 8 columns", len(rows[0]) == 8)
os.unlink(tmp_path)

# ── Summary ─────────────────────────────────────────────────────────
print("\n" + "═"*60)
passed = sum(results); total = len(results)
print(f"  RESULTS: {passed}/{total} tests passed")
if passed == total:
    print("  🎉 All tests passed — agents are production-ready!")
else:
    failed = [i+1 for i,r in enumerate(results) if not r]
    print(f"  ⚠️  Failed: {failed}")
print("═"*60 + "\n")
