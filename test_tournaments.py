from bertrand_pricing_env import BertrandPricingEnv
from agents import AlwaysNashAgent, AlwaysColludeAgent, TitForTatAgent, RandomAgent
from tournament_1 import Tournament

# 1. Create the environment (all params optional — defaults match Week 2 spec)
env = BertrandPricingEnv(
    a=100,           # market size         — default
    b=1.0,           # own-price sensitivity — default
    d=0.5,           # cross-price sensitivity — default
    marginal_cost=20, # MC for both firms   — default
    n_price_levels=30, # price grid size    — default
    max_steps=1000,  # steps per episode   — default
    noise_std=2.0,   # demand noise        — default
)

# 2. Create agents — they read Nash/monopoly prices FROM the env automatically
agents = [
    AlwaysNashAgent(env.price_grid, env.nash_price),
    AlwaysColludeAgent(env.price_grid, env.monopoly_price),
    TitForTatAgent(env.price_grid, env.monopoly_price),
    RandomAgent(env.price_grid, seed=42),
]

# 3. Run the tournament
t = Tournament(env, agents, n_episodes=10, n_steps=1000, seed=0, verbose=True)
results = t.run()

# 4. Print the summary table
t.print_summary(results)

# # 5. (Optional) Export to CSV for your report
# t.export_csv(results, "tournament_results.csv")