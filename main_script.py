# -*- coding: utf-8 -*-
# %%
# %% [markdown]
# # Introduction
#
# This work presents a constrained combinatorial optimization approach to the **Sports League Assignment Problem** using **Genetic Algorithms (GAs)**. The objective is to allocate a fixed pool of professional players into a set of 5 structurally valid teams in such a way that the **standard deviation of the teams\' average skill ratings** is minimized—promoting competitive balance across the league.
#
# Each player is defined by three attributes: **position** (one of `GK`, `DEF`, `MID`, `FWD`), **skill rating** (a numerical measure of ability), and **cost** (in million euros). A valid solution must satisfy the following **hard constraints**:
#
# - Each team must consist of exactly **7 players**, with a specific positional structure: **1 GK, 2 DEF, 2 MID, and 2 FWD**
# - Each team must have a **total cost ≤ 750 million €**
# - Each player must be assigned to **exactly one team** (no overlaps)
#
# The **search space** is therefore highly constrained and discrete, and infeasible configurations are explicitly excluded from the solution space. The optimization objective is to identify league configurations where teams are not only valid but also **skill-balanced**, quantified by the **standard deviation of average skill ratings across teams**, which serves as the **fitness function** (to be minimized).
#
# To address this, we implement a domain-adapted **Genetic Algorithm framework** featuring:
#
# - A custom **representation** based on team-to-player mappings
# - Validity-preserving **mutation** and **crossover** operators
# - Multiple **selection mechanisms**
# - Optional **elitism** and population-level diversity handling
#
# This report provides a formal problem definition, details the design of the solution encoding and operators, and presents empirical results comparing different GA configurations. The overall objective is to evaluate how well GA-based metaheuristics can navigate this complex constrained search space and evolve solutions that both satisfy domain constraints and optimize league balance.
#
# In addition to Genetic Algorithms, this project also explores and evaluates alternative optimization strategies, such as **Hill Climbing** and **Simulated Annealing**, which are well-suited for navigating discrete and constrained search spaces. These algorithms offer different trade-offs in terms of exploration, exploitation, and convergence speed. By implementing and benchmarking multiple approaches on the same problem, we aim to gain deeper insights into their relative effectiveness and robustness when applied to complex constrained optimization tasks such as the Sports League Assignment. This comparative analysis enhances the interpretability of results and supports a broader understanding of the strengths and limitations of population-based versus local search-based heuristics.

# %% [markdown]
# ## Cell 1: Setup and Critical Data Loading/Debug

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from solution import LeagueSolution, LeagueHillClimbingSolution, LeagueSASolution
from evolution import genetic_algorithm, hill_climbing, simulated_annealing_for_league
from operators import (
    # Base Mutations (examples, can be used for comparison)
    mutate_swap, 
    mutate_team_shift,
    mutate_shuffle_team, 
    # New/Adapted Mutations
    mutate_swap_constrained,
    mutate_targeted_player_exchange,
    mutate_shuffle_within_team_constrained,
    # Base Crossovers (examples, can be used for comparison)
    crossover_one_point,
    crossover_uniform,
    # New/Adapted Crossovers
    crossover_one_point_prefer_valid,
    crossover_uniform_prefer_valid,
    # Base Selections (examples, can be used for comparison)
    selection_ranking,
    # New/Adapted Selections
    selection_tournament_variable_k,
    selection_boltzmann
)

# Load player data
players_df = pd.read_csv("players.csv", sep=";")
players = players_df.to_dict(orient="records")

# --- DEBUG: Print keys of the first player to check column names ---
# THIS IS THE CRITICAL DEBUG LINE - ENSURE IT RUNS FIRST
if players:
    print("Keys in the first player dictionary:", players[0].keys())
else:
    print("Player data is empty or failed to load.")
# --- END DEBUG ---

# %% [markdown]
# ## Cell 2: Further Setup and Data Inspection

# %%
# Define problem parameters (can be centralized here for clarity)
NUM_TEAMS = 5
TEAM_SIZE = 7
MAX_BUDGET = 750

print("Player data loaded successfully.") 
print(f"Total players: {len(players)}")
players_df.head()


# %% [markdown]
# ## Problem Representation Details (from original notebook)

# %% [markdown]
# ### A. Team-based Representation (Structured Encoding)
#
# Let:
#
# - $P = \{p_1, p_2, \dots, p_{35}\}$ be the set of players  
# - $T = \{t_1, t_2, \dots, t_5\}$ be the set of teams
#
# Define the assignment function:
#
# $$
# A: P \rightarrow T
# $$
#
# such that each player is assigned to exactly one team, and the following constraints are satisfied:
#
# **Team Size:**
#
# $$
# \forall t_j \in T,\quad \left|\{p_i \in P \mid A(p_i) = t_j\}\right| = 7
# $$
#
# **Positional Requirements:** For each team $t_j \in T$:
#
# $$
# \begin{aligned}
# &|\{p_i \in P \mid A(p_i) = t_j \land pos(p_i) = \text{GK}\}| = 1 \\\n# &|\{p_i \in P \mid A(p_i) = t_j \land pos(p_i) = \text{DEF}\}| = 2 \\\n# &|\{p_i \in P \mid A(p_i) = t_j \land pos(p_i) = \text{MID}\}| = 2 \\\n# &|\{p_i \in P \mid A(p_i) = t_j \land pos(p_i) = \text{FWD}\}| = 2
# \end{aligned}
# $$
#
# **Budget Constraint:**
#
# $$
# \forall t_j \in T,\quad \sum_{p_i \in P \mid A(p_i) = t_j} cost(p_i) \leq 750
# $$
#
# **Objective Function:** Minimize the standard deviation of average team skill:
#
# $$
# f(A) = \sigma\left(\left\{\frac{1}{7} \sum_{p_i \in P \mid A(p_i) = t_j} skill(p_i) \,\middle|\, t_j \in T\right\}\right)
# $$

# %% [markdown]
# ### B. Player-assignment Representation (Linear Encoding) - This is what is implemented
#
# Let:
#
# - $P = \{p_1, p_2, \dots, p_{35}\}$ be the set of players  
# - $T = \{0, 1, 2, 3, 4\}$ be team IDs
#
# A solution is represented by a vector:
#
# $$
# \mathbf{a} = [a_1, a_2, \dots, a_{35}] \in T^{35}
# $$
#
# where $a_i$ is the team assignment for player $p_i$.
#
# **Team Definitions:**
#
# $$
# P_j = \{p_i \in P \mid a_i = j\}, \quad \forall j \in T
# $$
#
# **Constraints:**
#
# $$
# |P_j| = 7 \quad \text{and}
# $$
#
# $$
# \begin{aligned}
# &|\{p \in P_j \mid pos(p) = \text{GK}\}| = 1 \\\n# &|\{p \in P_j \mid pos(p) = \text{DEF}\}| = 2 \\\n# &|\{p \in P_j \mid pos(p) = \text{MID}\}| = 2 \\\n# &|\{p \in P_j \mid pos(p) = \text{FWD}\}| = 2 \\\n# &\sum_{p \in P_j} cost(p) \leq 750
# \end{aligned}
# $$
#
# **Objective Function:**
#
# $$
# f(\mathbf{a}) = \sigma\left(\left\{\frac{1}{7} \sum_{p \in P_j} skill(p) \,\middle|\, j \in T\right\}\right)
# $$

# %% [markdown]
# ## 1. Hill Climbing

# %%
print("Running Hill Climbing Algorithm...")
start_time_hc = time.time()
hc_solution, hc_fitness, hc_history = hill_climbing(players, max_iterations=1000, verbose=True)
end_time_hc = time.time()

print(f"Hill Climbing finished in {end_time_hc - start_time_hc:.2f} seconds.")
print(f"Best solution found by Hill Climbing: {hc_solution.assignment}")
print(f"Best fitness: {hc_fitness}")

# Plot Hill Climbing History
plt.figure(figsize=(10, 6))
plt.plot(hc_history, marker='o', linestyle='-')
plt.title('Hill Climbing Convergence')
plt.xlabel('Improvement Step')
plt.ylabel('Fitness (Std Dev of Avg Team Skills)')
plt.grid(True)
plt.show()

# %% [markdown]
# ## 2. Simulated Annealing

# %%
from solution import LeagueSASolution # Ensure this is imported if not already at the top
from evolution import simulated_annealing_for_league # Ensure this is imported

print("Running Simulated Annealing Algorithm...")
start_time_sa = time.time()
# Parameters for SA: initial_temp, final_temp, alpha, iterations_per_temp
sa_solution, sa_fitness, sa_history = simulated_annealing_for_league(
    players,
    initial_temp=1000,
    final_temp=0.1,
    alpha=0.99,
    iterations_per_temp=50, # Number of iterations at each temperature level
    verbose=True
)
end_time_sa = time.time()

print(f"Simulated Annealing finished in {end_time_sa - start_time_sa:.2f} seconds.")
print(f"Best solution found by Simulated Annealing: {sa_solution.assignment}")
print(f"Best fitness: {sa_fitness}")

# Plot Simulated Annealing History
plt.figure(figsize=(10, 6))
plt.plot(sa_history, linestyle='-')
plt.title('Simulated Annealing Convergence')
plt.xlabel('Iteration Step')
plt.ylabel('Fitness (Std Dev of Avg Team Skills)')
plt.grid(True)
plt.show()

# %% [markdown]
# ## 3. Genetic Algorithm with New/Adapted Operators
#
# Now we test the Genetic Algorithm with the newly implemented and adapted operators. We will define several configurations to compare their performance.

# %%
# Define configurations for GA with new operators
# Each config: (mutation_op, crossover_op, selection_op, tournament_k_if_applicable, boltzmann_temp_if_applicable)
ga_configs_new = [
    {
        "name": "GA_Config_1 (SwapConst,1PtPreferV,TournVarK)",
        "mutation_operator_func": mutate_swap_constrained,
        "crossover_operator_func": crossover_one_point_prefer_valid,
        "selection_operator_func": selection_tournament_variable_k,
        "tournament_k": 3,
        "boltzmann_temp": None # Not used
    },
    {
        "name": "GA_Config_2 (TargetExch,UnifPreferV,Ranking)",
        "mutation_operator_func": mutate_targeted_player_exchange,
        "crossover_operator_func": crossover_uniform_prefer_valid,
        "selection_operator_func": selection_ranking,
        "tournament_k": None, # Not used
        "boltzmann_temp": None # Not used
    },
    {
        "name": "GA_Config_3 (ShuffleWithin,1PtPreferV,Boltzmann)",
        "mutation_operator_func": mutate_shuffle_within_team_constrained,
        "crossover_operator_func": crossover_one_point_prefer_valid,
        "selection_operator_func": selection_boltzmann,
        "tournament_k": None, # Not used
        "boltzmann_temp": 50 # Example temperature
    },
    {
        "name": "GA_Config_4 (TargetExch,UnifPreferV,TournVarK_k5)",
        "mutation_operator_func": mutate_targeted_player_exchange,
        "crossover_operator_func": crossover_uniform_prefer_valid,
        "selection_operator_func": selection_tournament_variable_k,
        "tournament_k": 5,
        "boltzmann_temp": None # Not used
    }
]

results_ga_new = []
best_solutions_ga_new = []

# Standard GA parameters (can be tuned further)
GA_GENERATIONS = 50 # Increased generations for potentially better convergence
GA_POPULATION_SIZE = 50
GA_ELITE_SIZE = 5
GA_MUTATION_RATE = 0.25 # Slightly increased mutation rate

print("Running Genetic Algorithm with NEW/ADAPTED operator configurations...")
for config in ga_configs_new:
    start_ga_new = time.time()
    print(f"Running {config['name']}...")
    
    best_ga_sol, history_ga_run = genetic_algorithm(
        players=players,
        population_size=GA_POPULATION_SIZE,
        generations=GA_GENERATIONS,
        mutation_rate=GA_MUTATION_RATE,
        elite_size=GA_ELITE_SIZE,
        mutation_operator_func=config['mutation_operator_func'],
        crossover_operator_func=config['crossover_operator_func'],
        selection_operator_func=config['selection_operator_func'],
        tournament_k=config['tournament_k'] if config['tournament_k'] else 3, # Default k if None
        boltzmann_temp=config['boltzmann_temp'] if config['boltzmann_temp'] else 100, # Default temp if None
        num_teams=NUM_TEAMS, 
        team_size=TEAM_SIZE, 
        max_budget=MAX_BUDGET,
        verbose=False # Set to True for detailed generation logs
    )
    
    if best_ga_sol:
        results_ga_new.append((config['name'], history_ga_run))
        best_solutions_ga_new.append((config['name'], best_ga_sol, best_ga_sol.fitness(players)))
        print(f"{config['name']} finished. Best fitness: {best_ga_sol.fitness(players):.4f}. Time: {time.time() - start_ga_new:.2f}s")
    else:
        print(f"{config['name']} failed to produce a solution.")
    print("----------------------------------------------------")

# Plot GA History for new configs
plt.figure(figsize=(14, 9))
for name, history in results_ga_new:
    plt.plot(history, label=name, marker='.')
plt.title('Genetic Algorithm Convergence - New/Adapted Operator Configurations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness (Std Dev of Avg Team Skills)')
plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))
plt.grid(True)
plt.tight_layout()
plt.show()

# Print best GA solutions from new configs
print("\nBest Solutions from New GA Configurations:")
for name, sol, fit in best_solutions_ga_new:
    print(f"{name}: Fitness = {fit:.4f}")
    # print(f"Solution: {sol.assignment}") # Uncomment to see the assignment

# %% [markdown]
# The standard deviation is related to optimization ability because it quantifies the dispersion or variability of a data set in relation to its mean. In optimization contexts, understanding variability can be crucial to identify the efficiency and accuracy of a process or model. (This was a comment in the original notebook)

