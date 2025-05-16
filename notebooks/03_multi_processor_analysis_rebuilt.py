# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %% [markdown]
# # Introduction
#
# This work presents a constrained combinatorial optimization approach to the **Sports League Assignment Problem** using **Genetic Algorithms (GAs)**. The objective is to allocate a fixed pool of professional players into a set of 5 structurally valid teams in such a way that the **standard deviation of the teams\" average skill ratings** is minimized—promoting competitive balance across the league.
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
import os # Added for path joining
import multiprocessing # Added for multiprocessing

from solution import LeagueSolution, LeagueHillClimbingSolution, LeagueSASolution
from evolution import genetic_algorithm, hill_climbing, simulated_annealing 
from operators import (
    # Base Mutations
    mutate_swap,
    mutate_team_shift,
    mutate_shuffle_team,
    # New/Adapted Mutations
    mutate_swap_constrained,
    mutate_targeted_player_exchange,
    mutate_shuffle_within_team_constrained,
    # Base Crossovers
    crossover_one_point,
    crossover_uniform,
    # New/Adapted Crossovers
    crossover_one_point_prefer_valid,
    crossover_uniform_prefer_valid,
    # Selection Operators
    selection_ranking,
    selection_tournament_variable_k,
    selection_boltzmann
)

# Define number of runs for stochastic algorithms
NUM_RUNS = 1 # Parameter for number of runs (e.g., 1, 5, 10, 30) - SET FOR CURRENT TEST

# Define the directory for saving graphs and results
MULTIPROCESSING_RESULTS_DIR = f"/home/ubuntu/CIFO_EXTENDED_Project/images_mp/run_{NUM_RUNS}_results" # Dynamic based on NUM_RUNS
IMAGES_MP_DIR = MULTIPROCESSING_RESULTS_DIR
# Ensure the directory exists
if not os.path.exists(IMAGES_MP_DIR):
    os.makedirs(IMAGES_MP_DIR)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Created directory: {IMAGES_MP_DIR}")

# Load player data (globally for worker processes to access if needed, though it's better to pass it)
players_df = pd.read_csv("players.csv", sep=";")
PLAYERS_DATA_GLOBAL = players_df.to_dict(orient="records") # Renamed for clarity

# %% [markdown]
# ## Cell 2: Further Setup, Data Inspection, and Experiment Parameters

# %%
all_results_summary = []
# Define problem parameters
NUM_TEAMS = 5
TEAM_SIZE = 7
MAX_BUDGET = 750

# Define number of runs for stochastic algorithms (and now Hill Climbing)
NUM_RUNS = 1 # Parameter for number of runs (e.g., 10, 30) - TEST RUN

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
# &|\{p_i \in P \mid A(p_i) = t_j \land pos(p_i) = \text{GK}\}| = 1 \\
# &|\{p_i \in P \mid A(p_i) = t_j \land pos(p_i) = \text{DEF}\}| = 2 \\
# &|\{p_i \in P \mid A(p_i) = t_j \land pos(p_i) = \text{MID}\}| = 2 \\
# &|\{p_i \in P \mid A(p_i) = t_j \land pos(p_i) = \text{FWD}\}| = 2
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
# &|\{p \in P_j \mid pos(p) = \text{GK}\}| = 1 \\
# &|\{p \in P_j \mid pos(p) = \text{DEF}\}| = 2 \\
# &|\{p \in P_j \mid pos(p) = \text{MID}\}| = 2 \\
# &|\{p \in P_j \mid pos(p) = \text{FWD}\}| = 2 \\
# &\sum_{p \in P_j} cost(p) \leq 750
# \end{aligned}
# $$
#
# **Objective Function:**
#
# $$
# f(\mathbf{a}) = \sigma\left(\left\{\frac{1}{7} \sum_{p \in P_j} skill(p) \,\middle|\, j \in T\right\}\right)
# $$

# %% [markdown]
# ## Main Execution Block

# %%
def main():
    script_total_start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Single-Processor Script execution started.")

    data_load_start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading player data...")
    # Player data is already loaded globally
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Player data loaded successfully. Total players: {len(players_data)}")
    if players_data:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] First player data: {players_data[0]}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All algorithms (HC, SA, GA) will be run {NUM_RUNS} times each.")
    data_load_end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Data loading and setup took {data_load_end_time - data_load_start_time:.2f} seconds.")

# %% [markdown]
#     # ## 1. Hill Climbing
#     #
#     # Hill Climbing is a local search algorithm. While a single run from a specific starting point is deterministic, running it multiple times from different random initial solutions provides a more robust evaluation.

# %%
hc_section_start_time = time.time()
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting Hill Climbing Algorithm ({NUM_RUNS} runs) ---")

hc_all_fitness_values = []
hc_all_exec_times = []
best_hc_solution_overall = None
best_hc_fitness_overall = float("inf")
best_hc_history_overall = [] # For the best run

for i in range(NUM_RUNS):
    print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] HC Run {i+1}/{NUM_RUNS}...")
    start_time_hc_run = time.time()
    
    initial_hc_solution_run = LeagueHillClimbingSolution(players_data, num_teams=NUM_TEAMS, team_size=TEAM_SIZE, max_budget=MAX_BUDGET)
    retry_attempts_hc = 0
    max_retry_hc = 5
    while not initial_hc_solution_run.is_valid() and retry_attempts_hc < max_retry_hc:
        print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] HC Run {i+1}: Initial solution invalid, retrying generation ({retry_attempts_hc+1})...")
        initial_hc_solution_run = LeagueHillClimbingSolution(players_data, num_teams=NUM_TEAMS, team_size=TEAM_SIZE, max_budget=MAX_BUDGET)
        retry_attempts_hc += 1
    
    if not initial_hc_solution_run.is_valid():
        print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] HC Run {i+1} failed to create a valid initial solution after {max_retry_hc} retries. Skipping run.")
        hc_all_fitness_values.append(float('nan'))
        hc_all_exec_times.append(float('nan'))
        continue

    hc_solution_obj_run, hc_fitness_val_run, hc_history_convergence_run = hill_climbing(
        initial_solution=initial_hc_solution_run, 
        players_data=players_data, 
        max_iterations=1000, 
        verbose=False
    )
    end_time_hc_run = time.time()
    hc_exec_time_run = end_time_hc_run - start_time_hc_run
    
    if hc_solution_obj_run:
        hc_all_fitness_values.append(hc_fitness_val_run)
        hc_all_exec_times.append(hc_exec_time_run)
        if hc_fitness_val_run < best_hc_fitness_overall:
            best_hc_fitness_overall = hc_fitness_val_run
            best_hc_solution_overall = hc_solution_obj_run
            best_hc_history_overall = hc_history_convergence_run
    else:
        print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] HC Run {i+1} did not find a valid solution during search.")
        hc_all_fitness_values.append(float('nan'))
        hc_all_exec_times.append(float('nan'))

hc_mean_fitness = np.nanmean(hc_all_fitness_values) if hc_all_fitness_values else float("nan")
hc_std_fitness = np.nanstd(hc_all_fitness_values) if hc_all_fitness_values else float("nan")
hc_mean_exec_time = np.nanmean(hc_all_exec_times) if hc_all_exec_times else float("nan")

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hill Climbing ({NUM_RUNS} runs) processing finished.")
print(f"  Mean Best Fitness: {hc_mean_fitness:.4f}")
print(f"  Std Dev Best Fitness: {hc_std_fitness:.4f}")
print(f"  Mean Execution Time per run: {hc_mean_exec_time:.2f}s")
if best_hc_solution_overall:
    print(f"  Overall Best HC Fitness: {best_hc_fitness_overall:.4f}")
    all_results_summary.append({
        "Algorithm": "Hill Climbing (SP)", 
        "Mean Fitness": hc_mean_fitness, 
        "Std Dev Fitness": hc_std_fitness, 
        "Mean Exec Time (s)": hc_mean_exec_time,
        "Overall Best Fitness": best_hc_fitness_overall,
        "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
    })
    plt.figure(figsize=(10, 6))
    plt.plot(best_hc_history_overall, marker="o", linestyle="-")
    plt.title(f"Hill Climbing Convergence (Best of {NUM_RUNS} Runs - SP)")
    plt.xlabel("Improvement Step")
    plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
    plt.grid(True)
    plt.savefig(os.path.join(IMAGES_SP_DIR, "hc_convergence_sp.png")) # Updated path
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved Hill Climbing convergence plot to {IMAGES_SP_DIR}/hc_convergence_sp.png")
    plt.close()
else:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hill Climbing did not find any valid solution across all runs that produced a best overall.")
    all_results_summary.append({
        "Algorithm": "Hill Climbing (SP)", 
        "Mean Fitness": hc_mean_fitness, 
        "Std Dev Fitness": hc_std_fitness, 
        "Mean Exec Time (s)": hc_mean_exec_time,
        "Overall Best Fitness": float('nan'),
        "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
    })
hc_section_end_time = time.time()
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hill Climbing section took {hc_section_end_time - hc_section_start_time:.2f} seconds.")

# ## 2. Simulated Annealing
#
# Similar to Hill Climbing, SA is run multiple times from different random initial solutions.


# %%
sa_section_start_time = time.time()
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting Simulated Annealing Algorithm ({NUM_RUNS} runs) ---")

sa_all_fitness_values = []
sa_all_exec_times = []
best_sa_solution_overall = None
best_sa_fitness_overall = float("inf")
best_sa_history_overall = [] # For the best run

for i in range(NUM_RUNS):
    print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] SA Run {i+1}/{NUM_RUNS}...")
    start_time_sa_run = time.time()
    
    initial_sa_solution_run = LeagueSASolution(players_data, num_teams=NUM_TEAMS, team_size=TEAM_SIZE, max_budget=MAX_BUDGET)
    retry_attempts_sa = 0
    max_retry_sa = 5
    while not initial_sa_solution_run.is_valid() and retry_attempts_sa < max_retry_sa:
        print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] SA Run {i+1}: Initial solution invalid, retrying generation ({retry_attempts_sa+1})...")
        initial_sa_solution_run = LeagueSASolution(players_data, num_teams=NUM_TEAMS, team_size=TEAM_SIZE, max_budget=MAX_BUDGET)
        retry_attempts_sa += 1
        
    if not initial_sa_solution_run.is_valid():
        print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] SA Run {i+1} failed to create a valid initial solution after {max_retry_sa} retries. Skipping run.")
        sa_all_fitness_values.append(float('nan'))
        sa_all_exec_times.append(float('nan'))
        continue

    sa_solution_obj_run, sa_fitness_val_run, sa_history_convergence_run = simulated_annealing(
        initial_solution=initial_sa_solution_run, 
        players_data=players_data, 
        initial_temp=1000, 
        final_temp=1, 
        alpha=0.99, 
        iterations_per_temp=100,
        verbose=False
    )
    end_time_sa_run = time.time()
    sa_exec_time_run = end_time_sa_run - start_time_sa_run
    
    if sa_solution_obj_run:
        sa_all_fitness_values.append(sa_fitness_val_run)
        sa_all_exec_times.append(sa_exec_time_run)
        if sa_fitness_val_run < best_sa_fitness_overall:
            best_sa_fitness_overall = sa_fitness_val_run
            best_sa_solution_overall = sa_solution_obj_run
            best_sa_history_overall = sa_history_convergence_run
    else:
        print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] SA Run {i+1} did not find a valid solution during search.")
        sa_all_fitness_values.append(float('nan'))
        sa_all_exec_times.append(float('nan'))

sa_mean_fitness = np.nanmean(sa_all_fitness_values) if sa_all_fitness_values else float("nan")
sa_std_fitness = np.nanstd(sa_all_fitness_values) if sa_all_fitness_values else float("nan")
sa_mean_exec_time = np.nanmean(sa_all_exec_times) if sa_all_exec_times else float("nan")

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Simulated Annealing ({NUM_RUNS} runs) processing finished.")
print(f"  Mean Best Fitness: {sa_mean_fitness:.4f}")
print(f"  Std Dev Best Fitness: {sa_std_fitness:.4f}")
print(f"  Mean Execution Time per run: {sa_mean_exec_time:.2f}s")
if best_sa_solution_overall:
    print(f"  Overall Best SA Fitness: {best_sa_fitness_overall:.4f}")
    all_results_summary.append({
        "Algorithm": "Simulated Annealing (SP)", 
        "Mean Fitness": sa_mean_fitness, 
        "Std Dev Fitness": sa_std_fitness, 
        "Mean Exec Time (s)": sa_mean_exec_time,
        "Overall Best Fitness": best_sa_fitness_overall,
        "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
    })
    plt.figure(figsize=(10, 6))
    plt.plot(best_sa_history_overall, marker=".", linestyle="-")
    plt.title(f"Simulated Annealing Convergence (Best of {NUM_RUNS} Runs - SP)")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
    plt.grid(True)
    plt.savefig(os.path.join(IMAGES_SP_DIR, "sa_convergence_sp.png")) # Updated path
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved Simulated Annealing convergence plot to {IMAGES_SP_DIR}/sa_convergence_sp.png")
    plt.close()
else:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Simulated Annealing did not find any valid solution across all runs that produced a best overall.")
    all_results_summary.append({
        "Algorithm": "Simulated Annealing (SP)", 
        "Mean Fitness": sa_mean_fitness, 
        "Std Dev Fitness": sa_std_fitness, 
        "Mean Exec Time (s)": sa_mean_exec_time,
        "Overall Best Fitness": float('nan'),
        "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
    })
sa_section_end_time = time.time()
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Simulated Annealing section took {sa_section_end_time - sa_section_start_time:.2f} seconds.")

# ## 3. Genetic Algorithm Configurations
#
# Multiple GA configurations will be tested, each run multiple times.


# %%
ga_section_start_time = time.time()
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting Genetic Algorithm Configurations ({NUM_RUNS} runs each) ---")

ga_configurations = [
    {"name": "GA_Config_1_SwapConst1PtPreferVTournVarK", "pop_size": 50, "gens": 75, "mut_rate": 0.2, "elite": 5, "mut_op": mutate_swap_constrained, "cross_op": crossover_one_point_prefer_valid, "sel_op": selection_tournament_variable_k, "tourn_k": 3, "boltz_temp": None}, # Original best from SP
    {"name": "GA_Config_2_TargetExchUnifPreferVRank", "pop_size": 50, "gens": 75, "mut_rate": 0.2, "elite": 5, "mut_op": mutate_targeted_player_exchange, "cross_op": crossover_uniform_prefer_valid, "sel_op": selection_ranking, "tourn_k": None, "boltz_temp": None},
    {"name": "GA_Config_3_ShuffleTeamConst1PtPreferVBoltz", "pop_size": 50, "gens": 75, "mut_rate": 0.2, "elite": 5, "mut_op": mutate_shuffle_within_team_constrained, "cross_op": crossover_one_point_prefer_valid, "sel_op": selection_boltzmann, "tourn_k": None, "boltz_temp": 100},
    # Variations with more generations
    {"name": "GA_Config_4_SwapConst1PtPreferVTournVarK_150Gen", "pop_size": 50, "gens": 150, "mut_rate": 0.2, "elite": 5, "mut_op": mutate_swap_constrained, "cross_op": crossover_one_point_prefer_valid, "sel_op": selection_tournament_variable_k, "tourn_k": 3, "boltz_temp": None},
    # Variations with different population sizes
    {"name": "GA_Config_5_SwapConst1PtPreferVTournVarK_100Pop", "pop_size": 100, "gens": 75, "mut_rate": 0.2, "elite": 10, "mut_op": mutate_swap_constrained, "cross_op": crossover_one_point_prefer_valid, "sel_op": selection_tournament_variable_k, "tourn_k": 3, "boltz_temp": None},
    # Variations with different mutation rates
    {"name": "GA_Config_6_SwapConst1PtPreferVTournVarK_MutRate0.1", "pop_size": 50, "gens": 75, "mut_rate": 0.1, "elite": 5, "mut_op": mutate_swap_constrained, "cross_op": crossover_one_point_prefer_valid, "sel_op": selection_tournament_variable_k, "tourn_k": 3, "boltz_temp": None},
    {"name": "GA_Config_7_SwapConst1PtPreferVTournVarK_MutRate0.3", "pop_size": 50, "gens": 75, "mut_rate": 0.3, "elite": 5, "mut_op": mutate_swap_constrained, "cross_op": crossover_one_point_prefer_valid, "sel_op": selection_tournament_variable_k, "tourn_k": 3, "boltz_temp": None},
]

for config in ga_configurations:
    config_name_val = config["name"]
    timestamp_val = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n  [{timestamp_val}] Running GA Configuration: {config_name_val}")
    ga_all_fitness_values_config = []
    ga_all_exec_times_config = []
    best_ga_solution_config_overall = None
    best_ga_fitness_config_overall = float("inf")
    best_ga_history_config_overall = []

    for i in range(NUM_RUNS):
        config_name_val = config["name"]
        timestamp_val = time.strftime('%Y-%m-%d %H:%M:%S')
        run_count_val = f"{i+1}/{NUM_RUNS}"
        print(f"    [{timestamp_val}] GA Run {run_count_val} for {config_name_val}...")
        start_time_ga_run = time.time()
        
        # Call GA with parameters from config
        ga_solution_obj_run, ga_history_convergence_run = genetic_algorithm(
            players_data=players_data,
            population_size=config["pop_size"],
            generations=config["gens"],
            mutation_rate=config["mut_rate"],
            elite_size=config["elite"],
            mutation_operator_func=config["mut_op"],
            crossover_operator_func=config["cross_op"],
            selection_operator_func=config["sel_op"],
            tournament_k=config["tourn_k"] if config["tourn_k"] is not None else 3, # Default for sel_op that needs it
            boltzmann_temp=config["boltz_temp"] if config["boltz_temp"] is not None else 100, # Default for sel_op that needs it
            num_teams=NUM_TEAMS, 
            team_size=TEAM_SIZE, 
            max_budget=MAX_BUDGET,
            verbose=False # Set to True for detailed generation logs
        )
        end_time_ga_run = time.time()
        ga_exec_time_run = end_time_ga_run - start_time_ga_run

        if ga_solution_obj_run:
            ga_fitness_val_run = ga_solution_obj_run.fitness() # MODIFIED: Vectorized call
            ga_all_fitness_values_config.append(ga_fitness_val_run)
            ga_all_exec_times_config.append(ga_exec_time_run)
            if ga_fitness_val_run < best_ga_fitness_config_overall:
                best_ga_fitness_config_overall = ga_fitness_val_run
                best_ga_solution_config_overall = ga_solution_obj_run
                best_ga_history_config_overall = ga_history_convergence_run
        else:
            # This run of GA for the current configuration failed
            # config_name is from the outer loop: for config_name, config_params in ga_configurations.items():
            # i is the current run index from: for i in range(NUM_RUNS):
            current_timestamp_failure = time.strftime('%Y-%m-%d %H:%M:%S')
            # The config_name variable is from the loop: for config_name, config in ga_configurations.items():
            # The 'i' variable is from the loop: for i in range(NUM_RUNS):
            print(f"    [{current_timestamp_failure}] GA Run {i+1}/{NUM_RUNS} for {config_name} did not return a valid solution object.")
            ga_all_fitness_values_config.append(float('nan'))
            # ga_exec_time_run was calculated before this if/else block for the current run
            ga_all_exec_times_config.append(ga_exec_time_run)

    ga_mean_fitness_config = np.nanmean(ga_all_fitness_values_config) if ga_all_fitness_values_config else float("nan")
    ga_std_fitness_config = np.nanstd(ga_all_fitness_values_config) if ga_all_fitness_values_config else float("nan")
    ga_mean_exec_time_config = np.nanmean(ga_all_exec_times_config) if ga_all_exec_times_config else float("nan")

    current_timestamp_ga_summary = time.strftime('%Y-%m-%d %H:%M:%S')
    # In the loop 'for config_name, config_params in ga_configurations.items():'
    # config_name is the key string, config_params is the dictionary of parameters.    print(f"  [{current_timestamp_ga_summary}] GA Configuration {config_name} ({NUM_RUNS} runs) processing finished.")    print(f"    Mean Best Fitness: {ga_mean_fitness_config:.4f}")
    print(f"    Std Dev Best Fitness: {ga_std_fitness_config:.4f}")
    print(f"    Mean Execution Time per run: {ga_mean_exec_time_config:.2f}s")
    if best_ga_solution_config_overall:
        current_timestamp_ga_best = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"    [{current_timestamp_ga_best}] GA Config {config_name_val} - Overall Best Fitness: {best_ga_fitness_config_overall:.4f}")
        all_results_summary.append({
            "Algorithm": f"GA: {config_name_val} (MP-{NUM_RUNS} runs)", # Adapted to MP style
            "Mean Fitness": ga_mean_fitness_config,
            "Std Dev Fitness": ga_std_fitness_config,
            "Mean Exec Time (s)": ga_mean_exec_time_config,
            "Overall Best Fitness": best_ga_fitness_config_overall,
            "Mutation Op": config["mut_op"].__name__, # 'config' is the current dict in the loop
            "Crossover Op": config["cross_op"].__name__,
            "Selection Op": config["sel_op"].__name__
        })
        # Plotting logic for best_ga_history_config_overall (similar to MP script)
        if best_ga_history_config_overall and isinstance(best_ga_history_config_overall[0], (int, float, np.number)):
            plt.figure(figsize=(10, 6))
            plt.plot(best_ga_history_config_overall, marker="o", linestyle="-", markersize=4)
            plt.title(f"GA Convergence: {config_name_val} (Best of {NUM_RUNS} MP Runs)")
            plt.xlabel("Generation")
            plt.ylabel("Best Fitness in Population")
            plt.grid(True)
            plt.savefig(os.path.join(IMAGES_MP_DIR, f"ga_convergence_{config_name_val}_mp_{NUM_RUNS}runs.png"))
            current_timestamp_plot_save = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"    [{current_timestamp_plot_save}] Saved GA convergence plot to {IMAGES_MP_DIR}/ga_convergence_{config_name_val}_mp_{NUM_RUNS}runs.png")
            plt.close()
        elif best_ga_history_config_overall:
            current_timestamp_plot_warn = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"    [{current_timestamp_plot_warn}] GA Config {config_name_val} - Fitness history for best run is not in the expected format for plotting. Type: {type(best_ga_history_config_overall[0])}")
        else:
            current_timestamp_plot_info = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"    [{current_timestamp_plot_info}] No valid fitness history to plot for GA config {config_name_val}.")
    else: 
        current_timestamp_ga_no_sol = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"    [{current_timestamp_ga_no_sol}] GA Config {config_name_val} did not find any valid solution across all runs that produced a best overall.")
        all_results_summary.append({
            "Algorithm": f"GA: {config_name_val} (MP-{NUM_RUNS} runs)",
            "Mean Fitness": ga_mean_fitness_config,
            "Std Dev Fitness": ga_std_fitness_config,
            "Mean Exec Time (s)": ga_mean_exec_time_config,
            "Overall Best Fitness": float("nan"),
            "Mutation Op": config["mut_op"].__name__,
            "Crossover Op": config["cross_op"].__name__,
            "Selection Op": config["sel_op"].__name__
        })
        # Plotting logic for best_ga_history_config_overall (similar to MP script)
        if best_ga_history_config_overall and isinstance(best_ga_history_config_overall[0], (int, float, np.number)): # Check if plottable
            plt.figure(figsize=(10, 6))
            plt.plot(best_ga_history_config_overall, marker="o", linestyle="-", markersize=4)
            plt.title(f"GA Convergence: {config_name} (Best of {NUM_RUNS} MP Runs)")
            plt.xlabel("Generation")
            plt.ylabel("Best Fitness in Population")
            plt.grid(True)
            plt.savefig(os.path.join(IMAGES_MP_DIR, f"ga_convergence_{config_name}_mp_{NUM_RUNS}runs.png"))
            current_timestamp_plot_save = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"    [{current_timestamp_plot_save}] Saved GA convergence plot to {IMAGES_MP_DIR}/ga_convergence_{config_name}_mp_{NUM_RUNS}runs.png")
            plt.close()
        elif best_ga_history_config_overall:
            current_timestamp_plot_warn = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"    [{current_timestamp_plot_warn}] GA Config {config_name} - Fitness history for best run is not in the expected format for plotting. Type: {type(best_ga_history_config_overall[0])}")
        else:
            current_timestamp_plot_info = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"    [{current_timestamp_plot_info}] No valid fitness history to plot for GA config {config_name}.")
    else: # This is if best_ga_solution_config_overall is False
        current_timestamp_ga_no_sol = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"    [{current_timestamp_ga_no_sol}] GA Config {config_name_val} did not find any valid solution across all runs that produced a best overall.")
        all_results_summary.append({
            "Algorithm": f"GA: {config_name} (MP-{NUM_RUNS} runs)",
            "Mean Fitness": ga_mean_fitness_config,
            "Std Dev Fitness": ga_std_fitness_config,
            "Mean Exec Time (s)": ga_mean_exec_time_config,
            "Overall Best Fitness": float("nan"),
            "Mutation Op": config["mut_op"].__name__,
            "Crossover Op": config["cross_op"].__name__,
            "Selection Op": config["sel_op"].__name__
        })
ga_section_end_time = time.time()
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] All Genetic Algorithm configurations took {ga_section_end_time - ga_section_start_time:.2f} seconds.")

# Convert summary to DataFrame and save
summary_df = pd.DataFrame(all_results_summary)
summary_filename = os.path.join(IMAGES_SP_DIR, "all_algorithms_summary_sp.csv") # Updated path
summary_df.to_csv(summary_filename, index=False, sep=";")
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Overall summary saved to {summary_filename}")

script_total_end_time = time.time()
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Single-Processor Script execution finished. Total time: {script_total_end_time - script_total_start_time:.2f} seconds.")

# Execute the main function if the script is run directly (or cell is executed in notebook)
if __name__ == "__main__" or __name__ == "builtins": # For notebook execution
main()




# %% [markdown]
# ## Helper Functions for Multiprocessing Runs

# %%
def run_hill_climbing_trial(run_id, players_data_local, num_teams_local, team_size_local, max_budget_local, max_iterations_hc):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    # Use a slightly different print format to distinguish from SP if needed, or keep consistent
    # For now, keeping it similar to the reference MP script for easier comparison during adaptation
    print(f"  [{timestamp}] HC Run {run_id+1}/{NUM_RUNS} starting...")
    start_time_hc_run = time.time()
    
    # Ensure using the correct class from solution.py (which should be the vectorized one)
    initial_hc_solution_run = LeagueHillClimbingSolution(players_data_local, num_teams=num_teams_local, team_size=team_size_local, max_budget=max_budget_local)
    retry_attempts_hc = 0
    max_retry_hc = 5 # Consistent with reference MP script
    while not initial_hc_solution_run.is_valid() and retry_attempts_hc < max_retry_hc:
        # Verbose print for retries can be kept or removed based on preference
        # print(f"    [{time.strftime("%Y-%m-%d %H:%M:%S")}] HC Run {run_id+1}: Initial solution invalid, retrying generation ({retry_attempts_hc+1})...")
        initial_hc_solution_run = LeagueHillClimbingSolution(players_data_local, num_teams=num_teams_local, team_size=team_size_local, max_budget=max_budget_local)
        retry_attempts_hc += 1
    
    if not initial_hc_solution_run.is_valid():
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{timestamp}] HC Run {run_id+1} failed to create a valid initial solution after {max_retry_hc} retries. Skipping run.")
        return None, float("nan"), float("nan"), [] # Return structure consistent with MP script

    # Ensure using the correct hill_climbing function from evolution.py (vectorized)
    hc_solution_obj_run, hc_fitness_val_run, hc_history_convergence_run = hill_climbing(
        initial_solution=initial_hc_solution_run, 
        players_data=players_data_local, # Pass the local copy
        max_iterations=max_iterations_hc, 
        verbose=False # Keep verbose off for parallel runs to avoid cluttered output
    )
    end_time_hc_run = time.time()
    hc_exec_time_run = end_time_hc_run - start_time_hc_run
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  [{timestamp}] HC Run {run_id+1} finished in {hc_exec_time_run:.2f}s with fitness {hc_fitness_val_run if hc_solution_obj_run else 'N/A'}.")
    return hc_solution_obj_run, hc_fitness_val_run if hc_solution_obj_run else float("nan"), hc_exec_time_run, hc_history_convergence_run

def run_simulated_annealing_trial(run_id, players_data_local, num_teams_local, team_size_local, max_budget_local, sa_params_local):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  [{timestamp}] SA Run {run_id+1}/{NUM_RUNS} starting...")
    start_time_sa_run = time.time()
    initial_sa_solution = LeagueSASolution(players_data_local, num_teams=num_teams_local, team_size=team_size_local, max_budget=max_budget_local)
    retry_attempts_sa = 0
    max_retry_sa = 5
    while not initial_sa_solution.is_valid() and retry_attempts_sa < max_retry_sa:
        # print(f"    [{time.strftime("%Y-%m-%d %H:%M:%S")}] SA Run {run_id+1}: Initial solution invalid, retrying generation ({retry_attempts_sa+1})...")
        initial_sa_solution = LeagueSASolution(players_data_local, num_teams=num_teams_local, team_size=team_size_local, max_budget=max_budget_local)
        retry_attempts_sa += 1

    if not initial_sa_solution.is_valid():
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{timestamp}] SA Run {run_id+1} failed to create a valid initial solution after {max_retry_sa} retries. Skipping run.")
        return None, float("nan"), float("nan"), []

    # Ensure using the correct simulated_annealing function from evolution.py (vectorized)
    sa_solution_run, sa_fitness_run, sa_history_run = simulated_annealing(
        initial_solution=initial_sa_solution,
        players_data=players_data_local, 
        initial_temp=sa_params_local["initial_temp"],
        final_temp=sa_params_local["final_temp"],
        alpha=sa_params_local["alpha"],
        iterations_per_temp=sa_params_local["iterations_per_temp"],
        verbose=False
    )
    end_time_sa_run = time.time()
    sa_exec_time_run = end_time_sa_run - start_time_sa_run
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  [{timestamp}] SA Run {run_id+1} finished in {sa_exec_time_run:.2f}s with fitness {sa_fitness_run if sa_solution_run else 'N/A'}.")
    return sa_solution_run, sa_fitness_run if sa_solution_run else float("nan"), sa_exec_time_run, sa_history_run

def run_genetic_algorithm_trial(run_id, players_data_local, num_teams_local, team_size_local, max_budget_local, ga_params_local, config_name_local):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    # Corrected f-string for GA run print statement
    print(f"  [{timestamp}] GA Run {run_id+1}/{NUM_RUNS} for {config_name_local} starting...")
    start_time_ga_run = time.time()
    
    # Ensure using the correct genetic_algorithm function from evolution.py (vectorized)
    best_solution_ga_run, fitness_history_ga_run = genetic_algorithm(
        players_data=players_data_local, 
        population_size=ga_params_local["population_size"],
        generations=ga_params_local["generations"],
        mutation_rate=ga_params_local["mutation_rate"],
        elite_size=ga_params_local["elitism_size"],
        mutation_operator_func=ga_params_local["mutation_op"], 
        crossover_operator_func=ga_params_local["crossover_op"],
        selection_operator_func=ga_params_local["selection_op"],
        tournament_k=ga_params_local.get("tournament_k", 3), # Consistent with SP and MP ref
        boltzmann_temp=ga_params_local.get("boltzmann_temp", 100), # Consistent with SP and MP ref
        num_teams=num_teams_local, 
        team_size=team_size_local,
        max_budget=max_budget_local,
        verbose=False 
    )
    
    end_time_ga_run = time.time()
    actual_exec_time_ga_run = end_time_ga_run - start_time_ga_run 
    best_fitness_ga_run = best_solution_ga_run.fitness() if best_solution_ga_run else float("nan")
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    # Corrected f-string for GA run print statement
    print(f"  [{timestamp}] GA Run {run_id+1} for {config_name_local} finished in {actual_exec_time_ga_run:.2f}s with fitness {best_fitness_ga_run if best_solution_ga_run else 'N/A'}.")
    return best_solution_ga_run, best_fitness_ga_run, actual_exec_time_ga_run, fitness_history_ga_run





# %% [markdown]
# ## Main Execution Block (Adapted for Multiprocessing)

# %%
def main():
    script_total_start_time = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    # Consistent print statement with MP reference
    print(f"[{timestamp}] Multiprocessing Script for {NUM_RUNS} runs execution started.")

    # Data loading is already handled globally by PLAYERS_DATA_GLOBAL
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Player data loaded. Total players: {len(PLAYERS_DATA_GLOBAL)}")
    if PLAYERS_DATA_GLOBAL:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] First player data: {PLAYERS_DATA_GLOBAL[0]}")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] All algorithms (HC, SA, GA) will be run {NUM_RUNS} times each in parallel (where applicable).")
    
    # Initialize results summary list (this was outside main in SP, now inside for MP structure)
    all_results_summary = [] 
    
    # Determine number of processes to use
    # If NUM_RUNS is 1, it makes sense to use only 1 process, otherwise os.cpu_count() or NUM_RUNS if smaller
    num_processes = 1 if NUM_RUNS == 1 else min(NUM_RUNS, os.cpu_count() if os.cpu_count() else 1)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Using {num_processes} processes for parallel execution.")

    # --- 1. Hill Climbing (Parallel Execution) --- 
    hc_section_start_time = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] --- Starting Hill Climbing Algorithm ({NUM_RUNS} runs in parallel) ---")
    
    # Prepare parameters for each HC run
    # Max iterations for HC can be a parameter, using 1000 as in SP and MP ref
    hc_params_for_runs = [(i, PLAYERS_DATA_GLOBAL, NUM_TEAMS, TEAM_SIZE, MAX_BUDGET, 1000) for i in range(NUM_RUNS)]
    
    # Use multiprocessing Pool for parallel execution
    if NUM_RUNS > 1:
        with multiprocessing.Pool(processes=num_processes) as pool:
            hc_results_parallel = pool.starmap(run_hill_climbing_trial, hc_params_for_runs)
    else: # Single run, no need for pool, call directly for easier debugging if needed
        hc_results_parallel = [run_hill_climbing_trial(*hc_params_for_runs[0])]

    # Process results from parallel HC runs
    hc_all_solutions = [res[0] for res in hc_results_parallel if res is not None and res[0] is not None]
    hc_all_fitness_values = [res[1] for res in hc_results_parallel if res is not None]
    hc_all_exec_times = [res[2] for res in hc_results_parallel if res is not None]
    hc_all_histories = [res[3] for res in hc_results_parallel if res is not None]

    best_hc_fitness_overall = float("inf")
    best_hc_solution_overall = None
    best_hc_history_overall = []
    if hc_all_solutions: # Check if any valid solutions were found
        for i, fit_val in enumerate(hc_all_fitness_values):
            if not np.isnan(fit_val) and fit_val < best_hc_fitness_overall:
                # Ensure indices are valid for all lists before accessing
                if i < len(hc_all_solutions) and hc_all_solutions[i] is not None and \
                   i < len(hc_all_histories) and hc_all_histories[i] is not None:
                    best_hc_fitness_overall = fit_val
                    best_hc_solution_overall = hc_all_solutions[i] 
                    best_hc_history_overall = hc_all_histories[i]
    
    hc_mean_fitness = np.nanmean([f for f in hc_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in hc_all_fitness_values) else float("nan")
    hc_std_fitness = np.nanstd([f for f in hc_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in hc_all_fitness_values) else float("nan")
    hc_mean_exec_time = np.nanmean([t for t in hc_all_exec_times if not np.isnan(t)]) if any(not np.isnan(t) for t in hc_all_exec_times) else float("nan")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Hill Climbing ({NUM_RUNS} runs) processing finished.")
    print(f"  Mean Best Fitness: {hc_mean_fitness:.4f}")
    print(f"  Std Dev Best Fitness: {hc_std_fitness:.4f}")
    print(f"  Mean Execution Time per run: {hc_mean_exec_time:.2f}s")
    if best_hc_solution_overall:
        print(f"  Overall Best HC Fitness: {best_hc_fitness_overall:.4f}")
        all_results_summary.append({
            "Algorithm": f"Hill Climbing (MP-{NUM_RUNS} runs)", 
            "Mean Fitness": hc_mean_fitness, 
            "Std Dev Fitness": hc_std_fitness, 
            "Mean Exec Time (s)": hc_mean_exec_time,
            "Overall Best Fitness": best_hc_fitness_overall,
            "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
        })
        if best_hc_history_overall: # Ensure history is not empty
            plt.figure(figsize=(10, 6))
            plt.plot(best_hc_history_overall, marker="o", linestyle="-")
            plt.title(f"Hill Climbing Convergence (Best of {NUM_RUNS} MP Runs)")
            plt.xlabel("Improvement Step")
            plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
            plt.grid(True)
            # Use IMAGES_MP_DIR for saving
            plt.savefig(os.path.join(IMAGES_MP_DIR, f"hc_convergence_mp_{NUM_RUNS}runs.png"))
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Saved Hill Climbing convergence plot to {IMAGES_MP_DIR}/hc_convergence_mp_{NUM_RUNS}runs.png")
            plt.close()
    else:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Hill Climbing did not find any valid solution across all runs that produced a best overall.")
        all_results_summary.append({
            "Algorithm": f"Hill Climbing (MP-{NUM_RUNS} runs)", 
            "Mean Fitness": hc_mean_fitness, 
            "Std Dev Fitness": hc_std_fitness, 
            "Mean Exec Time (s)": hc_mean_exec_time,
            "Overall Best Fitness": float("nan"), # Consistent with MP ref
            "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
        })
    hc_section_end_time = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Hill Climbing section took {hc_section_end_time - hc_section_start_time:.2f} seconds (wall time for parallel execution).")

    # --- 2. Simulated Annealing (Parallel Execution) --- 
    sa_section_start_time = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] --- Starting Simulated Annealing Algorithm ({NUM_RUNS} runs in parallel) ---")
    
    # SA parameters from MP reference (can be adjusted)
    sa_params = {
        "initial_temp": 1000,
        "final_temp": 0.1, # MP ref used 0.1, SP used 1
        "alpha": 0.99,
        "iterations_per_temp": 50 # MP ref used 50, SP used 100
    }
    sa_params_for_runs = [(i, PLAYERS_DATA_GLOBAL, NUM_TEAMS, TEAM_SIZE, MAX_BUDGET, sa_params) for i in range(NUM_RUNS)]

    if NUM_RUNS > 1:
        with multiprocessing.Pool(processes=num_processes) as pool:
            sa_results_parallel = pool.starmap(run_simulated_annealing_trial, sa_params_for_runs)
    else:
        sa_results_parallel = [run_simulated_annealing_trial(*sa_params_for_runs[0])]

    sa_all_solutions = [res[0] for res in sa_results_parallel if res is not None and res[0] is not None]
    sa_all_fitness_values = [res[1] for res in sa_results_parallel if res is not None]
    sa_all_exec_times = [res[2] for res in sa_results_parallel if res is not None]
    sa_all_histories = [res[3] for res in sa_results_parallel if res is not None]

    best_sa_fitness_overall = float("inf")
    best_sa_solution_overall = None
    best_sa_history_overall = []
    if sa_all_solutions:
        for i, fit_val in enumerate(sa_all_fitness_values):
            if not np.isnan(fit_val) and fit_val < best_sa_fitness_overall:
                if i < len(sa_all_solutions) and sa_all_solutions[i] is not None and \
                   i < len(sa_all_histories) and sa_all_histories[i] is not None:
                    best_sa_fitness_overall = fit_val
                    best_sa_solution_overall = sa_all_solutions[i]
                    best_sa_history_overall = sa_all_histories[i]

    sa_mean_fitness = np.nanmean([f for f in sa_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in sa_all_fitness_values) else float("nan")
    sa_std_fitness = np.nanstd([f for f in sa_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in sa_all_fitness_values) else float("nan")
    sa_mean_exec_time = np.nanmean([t for t in sa_all_exec_times if not np.isnan(t)]) if any(not np.isnan(t) for t in sa_all_exec_times) else float("nan")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Simulated Annealing ({NUM_RUNS} runs) processing finished.")
    print(f"  Mean Best Fitness: {sa_mean_fitness:.4f}")
    print(f"  Std Dev Best Fitness: {sa_std_fitness:.4f}")
    print(f"  Mean Execution Time per run: {sa_mean_exec_time:.2f}s")
    if best_sa_solution_overall:
        print(f"  Overall Best SA Fitness: {best_sa_fitness_overall:.4f}")
        all_results_summary.append({
            "Algorithm": f"Simulated Annealing (MP-{NUM_RUNS} runs)", 
            "Mean Fitness": sa_mean_fitness, 
            "Std Dev Fitness": sa_std_fitness, 
            "Mean Exec Time (s)": sa_mean_exec_time,
            "Overall Best Fitness": best_sa_fitness_overall,
            "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
        })
        if best_sa_history_overall:
            plt.figure(figsize=(10, 6))
            plt.plot(best_sa_history_overall, marker=".", linestyle="-") # Marker consistent with MP ref
            plt.title(f"Simulated Annealing Convergence (Best of {NUM_RUNS} MP Runs)")
            plt.xlabel("Temperature Step / Iteration") # Label consistent with MP ref
            plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
            plt.grid(True)
            plt.savefig(os.path.join(IMAGES_MP_DIR, f"sa_convergence_mp_{NUM_RUNS}runs.png"))
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Saved Simulated Annealing convergence plot to {IMAGES_MP_DIR}/sa_convergence_mp_{NUM_RUNS}runs.png")
            plt.close()
    else:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Simulated Annealing did not find any valid solution across all runs that produced a best overall.")
        all_results_summary.append({
            "Algorithm": f"Simulated Annealing (MP-{NUM_RUNS} runs)", 
            "Mean Fitness": sa_mean_fitness, 
            "Std Dev Fitness": sa_std_fitness, 
            "Mean Exec Time (s)": sa_mean_exec_time,
            "Overall Best Fitness": float("nan"),
            "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
        })
    sa_section_end_time = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Simulated Annealing section took {sa_section_end_time - sa_section_start_time:.2f} seconds (wall time for parallel execution).")

    # --- 3. Genetic Algorithms (Parallel Execution per Configuration) --- 
    ga_section_start_time = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] --- Starting Genetic Algorithms ({NUM_RUNS} runs per config in parallel) ---")

    # GA configurations (can be taken from SP or MP ref, ensuring ops are from current operators.py)
    ga_configurations = {
        "GA_Config_1_SwapConst1PtPreferVTournVarK": {
            "population_size": 50,
            "generations": 100,
            "mutation_op": mutate_swap_constrained, # from operators.py
            "mutation_rate": 0.1,
            "crossover_op": crossover_one_point_prefer_valid, # from operators.py
            "selection_op": selection_tournament_variable_k, # from operators.py
            "tournament_k": 3,
            "elitism_size": 2
        },
        "GA_Config_2_TargetExchUnifPreferVRanking": {
            "population_size": 50,
            "generations": 100,
            "mutation_op": mutate_targeted_player_exchange,
            "mutation_rate": 0.15,
            "crossover_op": crossover_uniform_prefer_valid,
            "selection_op": selection_ranking,
            "elitism_size": 2
        },
        "GA_Config_3_ShuffleWithin1PtPreferVBoltzmann": {
            "population_size": 50,
            "generations": 100,
            "mutation_op": mutate_shuffle_within_team_constrained,
            "mutation_rate": 0.05,
            "crossover_op": crossover_one_point_prefer_valid,
            "selection_op": selection_boltzmann,
            "elitism_size": 1 # MP ref had 1, SP had 2. Using 1 for now.
        },
        "GA_Config_4_TargetExchUnifPreferVTournVarK_k5": {
            "population_size": 50,
            "generations": 100,
            "mutation_op": mutate_targeted_player_exchange, 
            "mutation_rate": 0.1,
            "crossover_op": crossover_uniform_prefer_valid,
            "selection_op": selection_tournament_variable_k,
            "tournament_k": 5,
            "elitism_size": 3
        }
    }

    for config_name, ga_params_dict in ga_configurations.items():
        config_start_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{timestamp}] Processing GA Configuration: {config_name} ({NUM_RUNS} runs in parallel)")

        ga_params_for_runs = [(i, PLAYERS_DATA_GLOBAL, NUM_TEAMS, TEAM_SIZE, MAX_BUDGET, ga_params_dict, config_name) for i in range(NUM_RUNS)]

        if NUM_RUNS > 1:
            with multiprocessing.Pool(processes=num_processes) as pool:
                ga_results_parallel = pool.starmap(run_genetic_algorithm_trial, ga_params_for_runs)
        else:
            ga_results_parallel = [run_genetic_algorithm_trial(*ga_params_for_runs[0])]
        
        ga_all_solutions_config = [res[0] for res in ga_results_parallel if res is not None and res[0] is not None]
        ga_all_fitness_values_config = [res[1] for res in ga_results_parallel if res is not None]
        ga_all_exec_times_config = [res[2] for res in ga_results_parallel if res is not None]
        ga_all_histories_config = [res[3] for res in ga_results_parallel if res is not None]

        best_ga_fitness_config = float("inf")
        best_ga_solution_config = None
        best_ga_history_config = []
        if ga_all_solutions_config:
            for i, fit_val in enumerate(ga_all_fitness_values_config):
                if not np.isnan(fit_val) and fit_val < best_ga_fitness_config:
                    if i < len(ga_all_solutions_config) and ga_all_solutions_config[i] is not None and \
                       i < len(ga_all_histories_config) and ga_all_histories_config[i] is not None:
                        best_ga_fitness_config = fit_val
                        best_ga_solution_config = ga_all_solutions_config[i]
                        best_ga_history_config = ga_all_histories_config[i]

        ga_mean_fitness_config = np.nanmean([f for f in ga_all_fitness_values_config if not np.isnan(f)]) if any(not np.isnan(f) for f in ga_all_fitness_values_config) else float("nan")
        ga_std_fitness_config = np.nanstd([f for f in ga_all_fitness_values_config if not np.isnan(f)]) if any(not np.isnan(f) for f in ga_all_fitness_values_config) else float("nan")
        ga_mean_exec_time_config = np.nanmean([t for t in ga_all_exec_times_config if not np.isnan(t)]) if any(not np.isnan(t) for t in ga_all_exec_times_config) else float("nan")

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # Indentation of this print was one level deeper in MP ref, adjusting to be consistent with HC/SA section summaries
        print(f"  [{timestamp}] GA Config {config_name} ({NUM_RUNS} runs) - Mean Fitness: {ga_mean_fitness_config:.4f}, Std Dev Fitness: {ga_std_fitness_config:.4f}, Mean Exec Time: {ga_mean_exec_time_config:.2f}s")
        if best_ga_solution_config:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"    [{timestamp}] GA Config {config_name} - Overall Best Fitness: {best_ga_fitness_config:.4f}")
            all_results_summary.append({
                "Algorithm": f"GA: {config_name} (MP-{NUM_RUNS} runs)",
                "Mean Fitness": ga_mean_fitness_config,
                "Std Dev Fitness": ga_std_fitness_config,
                "Mean Exec Time (s)": ga_mean_exec_time_config,
                "Overall Best Fitness": best_ga_fitness_config,
                "Mutation Op": ga_params_dict["mutation_op"].__name__,
                "Crossover Op": ga_params_dict["crossover_op"].__name__,
                "Selection Op": ga_params_dict["selection_op"].__name__
            })
            if best_ga_history_config and isinstance(best_ga_history_config[0], (int, float, np.number)):
                plt.figure(figsize=(10, 6))
                plt.plot(best_ga_history_config, marker="o", linestyle="-", markersize=4) # markersize from MP ref
                plt.title(f"GA Convergence: {config_name} (Best of {NUM_RUNS} MP Runs)")
                plt.xlabel("Generation")
                plt.ylabel("Best Fitness in Population")
                plt.grid(True)
                plt.savefig(os.path.join(IMAGES_MP_DIR, f"ga_convergence_{config_name}_mp_{NUM_RUNS}runs.png"))
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"    [{timestamp}] Saved GA convergence plot to {IMAGES_MP_DIR}/ga_convergence_{config_name}_mp_{NUM_RUNS}runs.png")
                plt.close()
            elif best_ga_history_config: # If history exists but not plottable as numbers
                 timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                 print(f"    [{timestamp}] GA Config {config_name} - Fitness history for best run is not in the expected format for plotting (e.g. list of numbers). Type: {type(best_ga_history_config[0])}")
            else: # No history or empty history
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"    [{timestamp}] No valid fitness history to plot for GA config {config_name}.")
        else:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"    [{timestamp}] GA Config {config_name} did not find any valid solution across all runs that produced a best overall.")
            # Append with NaN for best fitness if no solution found
            all_results_summary.append({
                "Algorithm": f"GA: {config_name} (MP-{NUM_RUNS} runs)",
                "Mean Fitness": ga_mean_fitness_config,
                "Std Dev Fitness": ga_std_fitness_config,
                "Mean Exec Time (s)": ga_mean_exec_time_config,
                "Overall Best Fitness": float("nan"),
                "Mutation Op": ga_params_dict["mutation_op"].__name__,
                "Crossover Op": ga_params_dict["crossover_op"].__name__,
                "Selection Op": ga_params_dict["selection_op"].__name__
            })
        config_end_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{timestamp}] GA Configuration {config_name} section took {config_end_time - config_start_time:.2f} seconds.")

    ga_section_end_time = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Genetic Algorithms section took {ga_section_end_time - ga_section_start_time:.2f} seconds (wall time for parallel execution).")

    # --- 4. Comparative Plots and Summary CSV --- 
    if all_results_summary:
        results_df = pd.DataFrame(all_results_summary)
        
        # Comparative Execution Times Plot
        plt.figure(figsize=(14, 8)) # Adjusted figure size for better readability
        plot_df_times = results_df.dropna(subset=["Mean Exec Time (s)"])
        if not plot_df_times.empty:
            alg_names_times = [name.replace(f" (MP-{NUM_RUNS} runs)", "") for name in plot_df_times["Algorithm"]]
            bars = plt.bar(alg_names_times, plot_df_times["Mean Exec Time (s)"], color=plt.cm.viridis(np.linspace(0, 1, len(plot_df_times))))
            plt.xlabel("Algorithm / Configuration")
            plt.ylabel("Mean Execution Time per Run (s)")
            plt.title(f"Comparative Mean Execution Times ({NUM_RUNS} MP Runs)")
            plt.xticks(rotation=60, ha="right") # Rotate labels for better fit
            plt.tight_layout() # Adjust layout
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * yval, f"{yval:.2f}s", ha="center", va="bottom")
            plt.savefig(os.path.join(IMAGES_MP_DIR, f"comparative_times_mp_{NUM_RUNS}runs.png"))
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Saved comparative execution times plot to {IMAGES_MP_DIR}/comparative_times_mp_{NUM_RUNS}runs.png")
            plt.close()
        else:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] No valid data for comparative execution times plot after dropping NaNs.")

        # Comparative Best Fitness Plot
        plt.figure(figsize=(14, 8))
        plot_df_fitness = results_df.dropna(subset=["Overall Best Fitness"])
        if not plot_df_fitness.empty:
            alg_names_fitness = [name.replace(f" (MP-{NUM_RUNS} runs)", "") for name in plot_df_fitness["Algorithm"]]
            bars = plt.bar(alg_names_fitness, plot_df_fitness["Overall Best Fitness"], color=plt.cm.plasma(np.linspace(0, 1, len(plot_df_fitness))))
            plt.xlabel("Algorithm / Configuration")
            plt.ylabel("Overall Best Fitness Achieved")
            plt.title(f"Comparative Overall Best Fitness ({NUM_RUNS} MP Runs)")
            plt.xticks(rotation=60, ha="right")
            plt.tight_layout()
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * yval, f"{yval:.4f}", ha="center", va="bottom")
            plt.savefig(os.path.join(IMAGES_MP_DIR, f"comparative_fitness_mp_{NUM_RUNS}runs.png"))
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Saved comparative best fitness plot to {IMAGES_MP_DIR}/comparative_fitness_mp_{NUM_RUNS}runs.png")
            plt.close()
        else:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] No valid data for comparative best fitness plot after dropping NaNs.")

        # Save Summary CSV
        summary_file_path = os.path.join(IMAGES_MP_DIR, f"all_algorithms_summary_mp_{NUM_RUNS}runs.csv")
        results_df.to_csv(summary_file_path, index=False)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Saved all algorithms summary to {summary_file_path}")
    else:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] No results to plot or save in summary CSV.")

    script_total_end_time = time.time()
    total_script_duration = script_total_end_time - script_total_start_time
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Multiprocessing Script execution finished.") # Added newline for separation
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Total script execution time: {total_script_duration:.2f} seconds.")

# %% [markdown]
# ## Script Execution Trigger

# %%
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Good practice for multiprocessing, esp. on Windows
    main()


