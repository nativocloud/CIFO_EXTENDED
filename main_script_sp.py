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
# ## Cell 1: Setup and Critical Data Loading

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os # Added for path joining

from solution import LeagueSolution, LeagueHillClimbingSolution, LeagueSASolution
# Updated import for refactored evolution.py
from evolution import genetic_algorithm, hill_climbing, simulated_annealing 
from operators import (
    # Base Mutations (examples, can be used for comparison)
    # mutate_swap, 
    # mutate_team_shift,
    # mutate_shuffle_team, 
    # New/Adapted Mutations
    mutate_swap_constrained,
    mutate_targeted_player_exchange,
    mutate_shuffle_within_team_constrained,
    # Base Crossovers (examples, can be used for comparison)
    # crossover_one_point,
    # crossover_uniform,
    # New/Adapted Crossovers
    crossover_one_point_prefer_valid,
    crossover_uniform_prefer_valid,
    # Base Selections (examples, can be used for comparison)
    selection_ranking,
    # New/Adapted Selections
    selection_tournament_variable_k,
    selection_boltzmann
)

# Define the directory for saving graphs
SP_GRAPHS_DIR = "/home/ubuntu/CIFO_EXTENDED_Project/sp_graphs"
# Ensure the directory exists
if not os.path.exists(SP_GRAPHS_DIR):
    os.makedirs(SP_GRAPHS_DIR)

# Load player data
players_df = pd.read_csv("players.csv", sep=";")
players_data = players_df.to_dict(orient="records") # Renamed to players_data for clarity

# %% [markdown]
# ## Cell 2: Further Setup, Data Inspection, and Experiment Parameters

# %%
# Define problem parameters
NUM_TEAMS = 5
TEAM_SIZE = 7
MAX_BUDGET = 750

# Define number of runs for stochastic algorithms (and now Hill Climbing)
NUM_RUNS = 30 # Parameter for number of runs (e.g., 10, 30)

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
    # players_df.head() # This would print to console if not commented
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All algorithms (HC, SA, GA) will be run {NUM_RUNS} times each.")
    data_load_end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Data loading and setup took {data_load_end_time - data_load_start_time:.2f} seconds.")

    all_results_summary = []

    # %% [markdown]
    # ## 1. Hill Climbing
    #
    # Hill Climbing is a local search algorithm. While a single run from a specific starting point is deterministic, running it multiple times from different random initial solutions provides a more robust evaluation.

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
        while not initial_hc_solution_run.is_valid(players_data) and retry_attempts_hc < max_retry_hc:
            print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] HC Run {i+1}: Initial solution invalid, retrying generation ({retry_attempts_hc+1})...")
            initial_hc_solution_run = LeagueHillClimbingSolution(players_data, num_teams=NUM_TEAMS, team_size=TEAM_SIZE, max_budget=MAX_BUDGET)
            retry_attempts_hc += 1
        
        if not initial_hc_solution_run.is_valid(players_data):
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
        plt.savefig(os.path.join(SP_GRAPHS_DIR, "hc_convergence_sp.png"))
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved Hill Climbing convergence plot to sp_graphs/hc_convergence_sp.png")
        # plt.show() # Commented out for script execution
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


    # %% [markdown]
    # ## 2. Simulated Annealing
    #
    # Simulated Annealing is a probabilistic technique. We will run it multiple times to get statistical measures of its performance.

    # %%
    sa_section_start_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting Simulated Annealing Algorithm ({NUM_RUNS} runs) ---")

    sa_all_fitness_values = []
    sa_all_exec_times = []
    best_sa_solution_overall = None
    best_sa_fitness_overall = float("inf")
    best_sa_history_overall = []

    sa_params = {
        "initial_temp": 1000,
        "final_temp": 0.1,
        "alpha": 0.99,
        "iterations_per_temp": 50
    }

    for i in range(NUM_RUNS):
        print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] SA Run {i+1}/{NUM_RUNS}...")
        initial_sa_solution = LeagueSASolution(players_data, num_teams=NUM_TEAMS, team_size=TEAM_SIZE, max_budget=MAX_BUDGET)
        retry_attempts_sa = 0
        max_retry_sa = 5
        while not initial_sa_solution.is_valid(players_data) and retry_attempts_sa < max_retry_sa:
            print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] SA Run {i+1}: Initial solution invalid, retrying generation ({retry_attempts_sa+1})...")
            initial_sa_solution = LeagueSASolution(players_data, num_teams=NUM_TEAMS, team_size=TEAM_SIZE, max_budget=MAX_BUDGET)
            retry_attempts_sa += 1

        if not initial_sa_solution.is_valid(players_data):
            print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] SA Run {i+1} failed to create a valid initial solution after {max_retry_sa} retries. Skipping run.")
            sa_all_fitness_values.append(float('nan'))
            sa_all_exec_times.append(float('nan'))
            continue

        start_time_sa_run = time.time()
        sa_solution_run, sa_fitness_run, sa_history_run = simulated_annealing(
            initial_solution=initial_sa_solution,
            players_data=players_data,
            initial_temp=sa_params["initial_temp"],
            final_temp=sa_params["final_temp"],
            alpha=sa_params["alpha"],
            iterations_per_temp=sa_params["iterations_per_temp"],
            verbose=False
        )
        end_time_sa_run = time.time()
        
        if sa_solution_run:
            sa_all_fitness_values.append(sa_fitness_run)
            sa_all_exec_times.append(end_time_sa_run - start_time_sa_run)
            if sa_fitness_run < best_sa_fitness_overall:
                best_sa_fitness_overall = sa_fitness_run
                best_sa_solution_overall = sa_solution_run
                best_sa_history_overall = sa_history_run
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
        plt.plot(best_sa_history_overall, linestyle="-")
        plt.title(f"Simulated Annealing Convergence (Best of {NUM_RUNS} Runs - SP)")
        plt.xlabel("Iteration Step")
        plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
        plt.grid(True)
        plt.savefig(os.path.join(SP_GRAPHS_DIR, "sa_convergence_sp.png"))
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved Simulated Annealing convergence plot to sp_graphs/sa_convergence_sp.png")
        # plt.show() # Commented out for script execution
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

    # %% [markdown]
    # ## 3. Genetic Algorithm with New/Adapted Operators
    #
    # Genetic Algorithms are also stochastic. We will run each configuration multiple times.

    # %%
    ga_section_start_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting Genetic Algorithms ({NUM_RUNS} runs per config) ---")
    ga_params = {
        "population_size": 50,
        "generations": 100,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8, # Used internally by some crossover operators if they are designed to use it
        "elite_size": 2 # Used internally by GA if designed to use it
    }

    ga_configs_new = [
        {
            "name": "GA_Config_1 (SwapConst,1PtPreferV,TournVarK)",
            "mutation_operator_func": mutate_swap_constrained,
            "crossover_operator_func": crossover_one_point_prefer_valid,
            "selection_operator_func": selection_tournament_variable_k,
            "tournament_k": 3,
            "boltzmann_temp": None 
        },
        {
            "name": "GA_Config_2 (TargetExch,UnifPreferV,Ranking)",
            "mutation_operator_func": mutate_targeted_player_exchange,
            "crossover_operator_func": crossover_uniform_prefer_valid,
            "selection_operator_func": selection_ranking,
            "tournament_k": None, 
            "boltzmann_temp": None 
        },
        {
            "name": "GA_Config_3 (ShuffleWithin,1PtPreferV,Boltzmann)",
            "mutation_operator_func": mutate_shuffle_within_team_constrained,
            "crossover_operator_func": crossover_one_point_prefer_valid,
            "selection_operator_func": selection_boltzmann,
            "tournament_k": None, 
            "boltzmann_temp": 50 
        },
        {
            "name": "GA_Config_4 (TargetExch,UnifPreferV,TournVarK_k5)",
            "mutation_operator_func": mutate_targeted_player_exchange,
            "crossover_operator_func": crossover_uniform_prefer_valid,
            "selection_operator_func": selection_tournament_variable_k,
            "tournament_k": 5, # Different k value
            "boltzmann_temp": None 
        }
    ]

    for ga_config in ga_configs_new:
        ga_config_start_time = time.time()
        print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] Running GA Configuration: {ga_config['name']}")
        ga_all_fitness_values_config = []
        ga_all_exec_times_config = []
        best_ga_solution_config = None
        best_ga_fitness_config = float("inf")
        best_ga_history_config = []

        for i in range(NUM_RUNS):
            print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] GA Run {i+1}/{NUM_RUNS} for {ga_config['name']}...")
            start_time_ga_run = time.time()
            # Call refactored genetic_algorithm function
            # Note: elite_size and crossover_rate are part of ga_params and handled internally by genetic_algorithm if needed
            best_solution_ga_run, history_ga_run = genetic_algorithm(
                players_data=players_data,
                population_size=ga_params["population_size"],
                generations=ga_params["generations"],
                mutation_rate=ga_params["mutation_rate"],
                elite_size=ga_params["elite_size"], # Pass for internal use
                mutation_operator_func=ga_config["mutation_operator_func"],
                crossover_operator_func=ga_config["crossover_operator_func"],
                selection_operator_func=ga_config["selection_operator_func"],
                tournament_k=ga_config.get("tournament_k"),
                boltzmann_temp=ga_config.get("boltzmann_temp"),
                num_teams=NUM_TEAMS,
                team_size=TEAM_SIZE,
                max_budget=MAX_BUDGET,
                verbose=False
            )
            end_time_ga_run = time.time()
            ga_exec_time_run = end_time_ga_run - start_time_ga_run

            if best_solution_ga_run:
                # Fitness is calculated from the best_solution_ga_run object
                current_run_fitness = best_solution_ga_run.fitness(players_data)
                ga_all_fitness_values_config.append(current_run_fitness)
                ga_all_exec_times_config.append(ga_exec_time_run)
                if current_run_fitness < best_ga_fitness_config:
                    best_ga_fitness_config = current_run_fitness
                    best_ga_solution_config = best_solution_ga_run
                    best_ga_history_config = history_ga_run
            else:
                print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] GA Run {i+1} for {ga_config['name']} did not return a solution.")
                ga_all_fitness_values_config.append(float('nan'))
                ga_all_exec_times_config.append(float('nan'))

        ga_mean_fitness_config = np.nanmean(ga_all_fitness_values_config) if ga_all_fitness_values_config else float("nan")
        ga_std_fitness_config = np.nanstd(ga_all_fitness_values_config) if ga_all_fitness_values_config else float("nan")
        ga_mean_exec_time_config = np.nanmean(ga_all_exec_times_config) if ga_all_exec_times_config else float("nan")

        print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] GA Config {ga_config['name']} ({NUM_RUNS} runs) processing finished.")
        print(f"    Mean Best Fitness: {ga_mean_fitness_config:.4f}")
        print(f"    Std Dev Best Fitness: {ga_std_fitness_config:.4f}")
        print(f"    Mean Execution Time per run: {ga_mean_exec_time_config:.2f}s")
        if best_ga_solution_config:
            print(f"    Overall Best GA Fitness for Config: {best_ga_fitness_config:.4f}")
            # print(f"    Overall Best GA Solution for Config: {best_ga_solution_config.assignment}")
            all_results_summary.append({
                "Algorithm": ga_config["name"] + " (SP)",
                "Mean Fitness": ga_mean_fitness_config,
                "Std Dev Fitness": ga_std_fitness_config,
                "Mean Exec Time (s)": ga_mean_exec_time_config,
                "Overall Best Fitness": best_ga_fitness_config,
                "Mutation Op": ga_config["mutation_operator_func"].__name__,
                "Crossover Op": ga_config["crossover_operator_func"].__name__,
                "Selection Op": ga_config["selection_operator_func"].__name__
            })

            plt.figure(figsize=(10, 6))
            plt.plot(best_ga_history_config, linestyle="-")
            plt.title(f"GA Convergence ({ga_config['name']} - Best of {NUM_RUNS} Runs - SP)")
            plt.xlabel("Generation")
            plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
            plt.grid(True)
            # Sanitize filename for GA config
            sanitized_config_name = ga_config['name'].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
            plt.savefig(os.path.join(SP_GRAPHS_DIR, f"ga_convergence_{sanitized_config_name}_sp.png"))
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved GA convergence plot to sp_graphs/ga_convergence_{sanitized_config_name}_sp.png")
            # plt.show() # Commented out for script execution
            plt.close()
        else:
            print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] GA Config {ga_config['name']} did not find any valid solution across all runs that produced a best overall.")
            all_results_summary.append({
                "Algorithm": ga_config["name"] + " (SP)",
                "Mean Fitness": ga_mean_fitness_config,
                "Std Dev Fitness": ga_std_fitness_config,
                "Mean Exec Time (s)": ga_mean_exec_time_config,
                "Overall Best Fitness": float('nan'),
                "Mutation Op": ga_config["mutation_operator_func"].__name__,
                "Crossover Op": ga_config["crossover_operator_func"].__name__,
                "Selection Op": ga_config["selection_operator_func"].__name__
            })
        ga_config_end_time = time.time()
        print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] GA Configuration {ga_config['name']} took {ga_config_end_time - ga_config_start_time:.2f} seconds.")
    ga_section_end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Genetic Algorithms section took {ga_section_end_time - ga_section_start_time:.2f} seconds.")

    # %% [markdown]
    # ## 4. Comparative Analysis

    # %%
    analysis_start_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting Comparative Analysis ---")
    results_df = pd.DataFrame(all_results_summary)
    print("Results Summary Table (SP):")
    print(results_df.to_string())

    # Plotting comparative fitness
    plt.figure(figsize=(12, 7))
    results_df_sorted_fitness = results_df.sort_values(by="Mean Fitness")
    plt.bar(results_df_sorted_fitness["Algorithm"], results_df_sorted_fitness["Mean Fitness"], yerr=results_df_sorted_fitness["Std Dev Fitness"], capsize=5, color="skyblue")
    plt.xlabel("Algorithm Configuration")
    plt.ylabel("Mean Best Fitness (Lower is Better)")
    plt.title("Comparative Mean Best Fitness of Algorithms (SP)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(SP_GRAPHS_DIR, "comparative_fitness_sp.png"))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved comparative fitness plot to sp_graphs/comparative_fitness_sp.png")
    # plt.show() # Commented out for script execution
    plt.close()

    # Plotting comparative execution times
    plt.figure(figsize=(12, 7))
    results_df_sorted_time = results_df.sort_values(by="Mean Exec Time (s)")
    plt.bar(results_df_sorted_time["Algorithm"], results_df_sorted_time["Mean Exec Time (s)"], color="lightcoral")
    plt.xlabel("Algorithm Configuration")
    plt.ylabel("Mean Execution Time (s)")
    plt.title("Comparative Mean Execution Times of Algorithms (SP)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(SP_GRAPHS_DIR, "comparative_times_sp.png"))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved comparative execution times plot to sp_graphs/comparative_times_sp.png")
    # plt.show() # Commented out for script execution
    plt.close()
    analysis_end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Comparative Analysis section took {analysis_end_time - analysis_start_time:.2f} seconds.")

    script_total_end_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Single-Processor Script finished. Total execution time: {script_total_end_time - script_total_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()

