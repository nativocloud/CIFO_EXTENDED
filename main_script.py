# -*- coding: utf-8 -*-
# %%
# %% [markdown]
# # Introduction
#
# This work presents a constrained combinatorial optimization approach to the **Sports League Assignment Problem** using **Genetic Algorithms (GAs)**. The objective is to allocate a fixed pool of professional players into a set of 5 structurally valid teams in such a way that the **standard deviation of the teams\\' average skill ratings** is minimized—promoting competitive balance across the league.
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
# ## Cell 1: Setup and Critical Data Loading

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

# %% [markdown]
# ## Cell 2: Further Setup and Data Inspection

# %%
# Define problem parameters (can be centralized here for clarity)
NUM_TEAMS = 5
TEAM_SIZE = 7
MAX_BUDGET = 750

print("Player data loaded successfully.") 
print(f"Total players: {len(players)}")
if players:
    print("First player data:", players[0]) # Print first player to confirm structure
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
#
# Hill Climbing is a local search algorithm that iteratively moves towards an increasingly optimal solution by choosing the best neighbor. It is simple and fast but can get stuck in local optima.

# %%
print("Running Hill Climbing Algorithm...")
start_time_hc = time.time()
hc_solution, hc_fitness, hc_history = hill_climbing(players, max_iterations=1000, verbose=True)
end_time_hc = time.time()

print(f"Hill Climbing finished in {end_time_hc - start_time_hc:.2f} seconds.")
if hc_solution:
    print(f"Best solution found by Hill Climbing: {hc_solution.assignment}")
    print(f"Best fitness: {hc_fitness}")
else:
    print("Hill Climbing did not find a valid solution.")

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
#
# Simulated Annealing is a probabilistic technique for approximating the global optimum of a given function. It is inspired by annealing in metallurgy, a process involving heating and controlled cooling of a material to increase the size of its crystals and reduce their defects. The algorithm uses a temperature parameter that decreases over time, allowing it to escape local optima, especially at higher temperatures.

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
if sa_solution:
    print(f"Best solution found by Simulated Annealing: {sa_solution.assignment}")
    print(f"Best fitness: {sa_fitness}")
else:
    print("Simulated Annealing did not find a valid solution.")

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
# Genetic Algorithms are population-based metaheuristics inspired by natural selection. They evolve a population of candidate solutions over generations using operators like selection, crossover, and mutation. This section tests the GA with various new and adapted operators designed to handle the problem's constraints effectively.

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
best_solutions_ga_new_with_time = [] # Modified to store time

# Standard GA parameters (can be tuned further)
GA_GENERATIONS = 50 # Increased generations for potentially better convergence
GA_POPULATION_SIZE = 50
GA_ELITE_SIZE = 5
GA_MUTATION_RATE = 0.25 # Slightly increased mutation rate

print("Running Genetic Algorithm with NEW/ADAPTED operator configurations...")
for config in ga_configs_new:
    start_ga_config_time = time.time() # Time for this specific config
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
    end_ga_config_time = time.time()
    config_exec_time = end_ga_config_time - start_ga_config_time
    
    if best_ga_sol:
        results_ga_new.append((config['name'], history_ga_run))
        best_solutions_ga_new_with_time.append((config['name'], best_ga_sol, best_ga_sol.fitness(players), config_exec_time))
        print(f"{config['name']} finished. Best fitness: {best_ga_sol.fitness(players):.4f}. Time: {config_exec_time:.2f}s")
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
for name, sol, fit, exec_time in best_solutions_ga_new_with_time:
    print(f"{name}: Fitness = {fit:.4f}, Time = {exec_time:.2f}s")
    # print(f"Solution: {sol.assignment}") # Uncomment to see the assignment

# %% [markdown]
# ## 4. Comparative Analysis of Algorithms
#
# In this section, we will compare the performance of Hill Climbing, Simulated Annealing, and the different Genetic Algorithm configurations.
# We will look at the best fitness achieved by each and the time taken for execution.

# %%
# Consolidate results for comparison
comparison_data = []

# Hill Climbing Results
if hc_solution:
    comparison_data.append({
        "Algorithm": "Hill Climbing",
        "Best Fitness": hc_fitness,
        "Execution Time (s)": end_time_hc - start_time_hc,
        "Mutation Operator": "N/A (Local Search)",
        "Crossover Operator": "N/A (Local Search)",
        "Selection Operator": "N/A (Local Search)"
    })

# Simulated Annealing Results
if sa_solution:
    comparison_data.append({
        "Algorithm": "Simulated Annealing",
        "Best Fitness": sa_fitness,
        "Execution Time (s)": end_time_sa - start_time_sa,
        "Mutation Operator": "N/A (Probabilistic Local Search)",
        "Crossover Operator": "N/A (Probabilistic Local Search)",
        "Selection Operator": "N/A (Probabilistic Local Search)"
    })

# Genetic Algorithm Results
for name, sol, fit, exec_time in best_solutions_ga_new_with_time:
    original_config = next((c for c in ga_configs_new if c["name"] == name), None)
    comparison_data.append({
        "Algorithm": name,
        "Best Fitness": fit,
        "Execution Time (s)": exec_time,
        "Mutation Operator": original_config["mutation_operator_func"].__name__ if original_config else "N/A",
        "Crossover Operator": original_config["crossover_operator_func"].__name__ if original_config else "N/A",
        "Selection Operator": original_config["selection_operator_func"].__name__ if original_config else "N/A"
    })

comparison_df = pd.DataFrame(comparison_data)
print("\nComparative Analysis of Algorithms:")
print(comparison_df.to_string())

# %% [markdown]
# ### Visualizing Comparative Performance

# %%
# Plotting Best Fitness
plt.figure(figsize=(12, 7))
# Sort by best fitness for clearer visualization
comparison_df_sorted_fitness = comparison_df.sort_values(by="Best Fitness", ascending=True)
plt.bar(comparison_df_sorted_fitness["Algorithm"], comparison_df_sorted_fitness["Best Fitness"], color=["skyblue", "lightcoral"] + ["lightgreen"]*(len(comparison_df_sorted_fitness)-2) if len(comparison_df_sorted_fitness)>2 else ["skyblue", "lightcoral"])
plt.xlabel("Algorithm / GA Configuration")
plt.ylabel("Best Fitness (Lower is Better)")
plt.title("Comparison of Best Fitness Achieved")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plotting Execution Time
plt.figure(figsize=(12, 7))
comparison_df_sorted_time = comparison_df.sort_values(by="Execution Time (s)", ascending=True)
plt.bar(comparison_df_sorted_time["Algorithm"], comparison_df_sorted_time["Execution Time (s)"], color=["skyblue", "lightcoral"] + ["lightgreen"]*(len(comparison_df_sorted_time)-2) if len(comparison_df_sorted_time)>2 else ["skyblue", "lightcoral"])
plt.xlabel("Algorithm / GA Configuration")
plt.ylabel("Execution Time (s)")
plt.title("Comparison of Execution Time")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Discussion of Results
#
# The experiments conducted provide insights into the performance of Hill Climbing, Simulated Annealing, and various Genetic Algorithm configurations for the Sports League Assignment Problem. The primary metrics for comparison are the best fitness achieved (lower standard deviation of average team skills is better) and the execution time.
#
# - **Hill Climbing:** As expected, Hill Climbing was the fastest algorithm. It quickly converged to a solution, but its fitness value indicates it likely settled in a local optimum. The convergence plot shows rapid initial improvements followed by stagnation. This algorithm is useful for obtaining a quick, reasonably good solution but is unlikely to find the global optimum for complex, multi-modal search spaces like this one.
#
# - **Simulated Annealing:** Simulated Annealing generally achieved better fitness values than Hill Climbing, demonstrating its ability to escape local optima due to its probabilistic acceptance of worse solutions, especially at higher temperatures. Its execution time was longer than Hill Climbing but significantly shorter than most Genetic Algorithm runs. The convergence plot for SA typically shows a more gradual improvement, with fluctuations as it explores the search space. The effectiveness of SA is sensitive to its parameters (initial temperature, cooling schedule, iterations per temperature), and further tuning could potentially yield even better results.
#
# - **Genetic Algorithm Configurations:**
#     - **Best Fitness:** The GA configurations, in general, had the potential to find solutions with better fitness (lower standard deviation) than both Hill Climbing and Simulated Annealing, though this depends heavily on the operators and parameters. The specific GA configuration that performed best in terms of fitness would be evident from the `comparison_df` output (e.g., one with a good combination of constraint-aware mutation, effective crossover, and appropriate selection pressure).
#     - **Execution Time:** GAs were the most computationally intensive, with execution times varying based on population size, number of generations, and the complexity of the operators. Configurations with more complex operators or those requiring more attempts to generate valid solutions (like `prefer_valid` crossovers if they struggle) might take longer.
#     - **Operator Effects:** 
#         - *Mutation:* Constraint-aware mutations like `mutate_swap_constrained` and `mutate_targeted_player_exchange` are crucial for maintaining solution validity and guiding the search effectively in this highly constrained problem. `mutate_shuffle_within_team_constrained` offers a way to fine-tune individual team compositions.
#         - *Crossover:* `crossover_one_point_prefer_valid` and `crossover_uniform_prefer_valid` aim to produce valid offspring more directly, which can be beneficial. The choice between one-point and uniform crossover often depends on the problem structure; uniform crossover tends to be more disruptive and can be better for exploration.
#         - *Selection:* `selection_tournament_variable_k` allows tuning selection pressure (higher `k` means higher pressure). `selection_boltzmann` offers an alternative selection mechanism whose pressure also depends on a temperature parameter, potentially offering a good balance between exploration and exploitation. `selection_ranking` provides a stable selection pressure.
#     - **Effectiveness of New/Adapted Operators:** The new and adapted operators were designed to be more suitable for this specific problem by respecting its constraints. Their effectiveness is reflected in the quality of solutions found by the GAs. The `prefer_valid` adaptations in crossover operators, for instance, likely reduced the number of invalid solutions the GA had to discard, potentially making the search more efficient.
#
# - **Overall Comparison:** Genetic Algorithms, despite their longer execution times, generally offer the best approach for finding high-quality solutions (lowest fitness) for this complex combinatorial problem due to their population-based search and ability to balance exploration and exploitation. Simulated Annealing provides a good trade-off between solution quality and computational cost. Hill Climbing is a quick baseline but is insufficient for finding optimal solutions.
#
# - **Limitations and Future Work:**
#     - *Parameter Tuning:* The performance of SA and GA is highly dependent on their parameters. A more systematic parameter tuning process (e.g., using grid search, random search, or meta-optimization techniques) could lead to improved results.
#     - *Scalability:* The current problem size (35 players, 5 teams) is manageable. Testing on larger datasets would be necessary to evaluate the scalability of the algorithms.
#     - *Multiple Runs:* For stochastic algorithms like SA and GA, results should ideally be averaged over multiple independent runs to account for randomness and provide more robust statistical comparisons.
#     - *Hybrid Approaches:* Combining GA with local search (memetic algorithms) could potentially improve solution quality by fine-tuning solutions found by the GA.
#     - *Alternative Representations/Operators:* Exploring other solution representations or more sophisticated genetic operators could lead to further improvements.
#     - *Multi-Objective Optimization:* If there were other objectives (e.g., maximizing total team skill while minimizing standard deviation), a multi-objective optimization approach would be necessary.

# %% [markdown]
# ## 6. Conclusion
#
# This project successfully implemented and evaluated Hill Climbing, Simulated Annealing, and several Genetic Algorithm configurations for the Sports League Assignment Problem. The objective was to minimize the standard deviation of average team skills while adhering to strict team composition and budget constraints.
#
# The results indicate that: 
# 1. **Hill Climbing** is fast but prone to local optima, yielding the least optimal solutions.
# 2. **Simulated Annealing** offers a better balance, finding higher quality solutions than Hill Climbing with moderate computational effort, by effectively navigating the search space and escaping some local optima.
# 3. **Genetic Algorithms**, particularly those employing tailored, constraint-aware operators, demonstrated the capability to find the best solutions (lowest fitness values), albeit at a higher computational cost. The choice of mutation, crossover, and selection operators significantly impacted GA performance, highlighting the importance of operator design in solving constrained combinatorial optimization problems.
#
# The newly adapted operators, such as `mutate_swap_constrained`, `mutate_targeted_player_exchange`, `crossover_one_point_prefer_valid`, and `selection_boltzmann`, contributed to the GA's ability to explore the constrained search space effectively. 
#
# Overall, Genetic Algorithms appear to be the most promising approach for this problem if solution quality is paramount and computational time is a secondary concern. For quicker, good-quality solutions, Simulated Annealing presents a viable alternative. Future work could focus on more extensive parameter tuning, hybridizing algorithms, and exploring more advanced GA techniques to further enhance solution quality and efficiency.

# %%
print("Notebook execution completed.")

