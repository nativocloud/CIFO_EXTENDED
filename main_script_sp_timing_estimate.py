# -*- coding: utf-8 -*-
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
import cProfile # Added for profiling
import pstats # Added for processing profiling results

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

# Define the directory for saving graphs and results
TIMING_ESTIMATE_DIR = "/home/ubuntu/CIFO_EXTENDED_Project/images_sp/timing_estimate"
IMAGES_SP_DIR = TIMING_ESTIMATE_DIR # Adjusted for this script
# Ensure the directory exists
if not os.path.exists(IMAGES_SP_DIR):
    os.makedirs(IMAGES_SP_DIR)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Created directory: {IMAGES_SP_DIR}")

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
NUM_RUNS = 1 # Parameter for number of runs (e.g., 10, 30)

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
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Single-Processor Timing Estimate Script execution started.")

    data_load_start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading player data...")
    # Player data is already loaded globally
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Player data loaded successfully. Total players: {len(players_data)}")
    if players_data:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] First player data: {players_data[0]}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All algorithms (HC, SA, GA) will be run {NUM_RUNS} times each.")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GA configurations will run for 1 generation for timing estimate.")
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
            "Algorithm": "Hill Climbing (SP-Timing)", 
            "Mean Fitness": hc_mean_fitness, 
            "Std Dev Fitness": hc_std_fitness, 
            "Mean Exec Time (s)": hc_mean_exec_time,
            "Overall Best Fitness": best_hc_fitness_overall,
            "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
        })
        plt.figure(figsize=(10, 6))
        plt.plot(best_hc_history_overall, marker="o", linestyle="-")
        plt.title(f"Hill Climbing Convergence (Best of {NUM_RUNS} Runs - SP-Timing)")
        plt.xlabel("Improvement Step")
        plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
        plt.grid(True)
        plt.savefig(os.path.join(IMAGES_SP_DIR, "hc_convergence_sp_timing.png")) # Updated path
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved Hill Climbing convergence plot to {IMAGES_SP_DIR}/hc_convergence_sp_timing.png")
        plt.close()
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hill Climbing did not find any valid solution across all runs that produced a best overall.")
        all_results_summary.append({
            "Algorithm": "Hill Climbing (SP-Timing)", 
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
        while not initial_sa_solution.is_valid() and retry_attempts_sa < max_retry_sa:
            print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] SA Run {i+1}: Initial solution invalid, retrying generation ({retry_attempts_sa+1})...")
            initial_sa_solution = LeagueSASolution(players_data, num_teams=NUM_TEAMS, team_size=TEAM_SIZE, max_budget=MAX_BUDGET)
            retry_attempts_sa += 1

        if not initial_sa_solution.is_valid():
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
            "Algorithm": "Simulated Annealing (SP-Timing)", 
            "Mean Fitness": sa_mean_fitness, 
            "Std Dev Fitness": sa_std_fitness, 
            "Mean Exec Time (s)": sa_mean_exec_time,
            "Overall Best Fitness": best_sa_fitness_overall,
            "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
        })
        plt.figure(figsize=(10, 6))
        plt.plot(best_sa_history_overall, marker=".", linestyle="-")
        plt.title(f"Simulated Annealing Convergence (Best of {NUM_RUNS} Runs - SP-Timing)")
        plt.xlabel("Accepted Improvement Step")
        plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
        plt.grid(True)
        plt.savefig(os.path.join(IMAGES_SP_DIR, "sa_convergence_sp_timing.png")) # Updated path
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved Simulated Annealing convergence plot to {IMAGES_SP_DIR}/sa_convergence_sp_timing.png")
        plt.close()
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Simulated Annealing did not find any valid solution across all runs that produced a best overall.")
        all_results_summary.append({
            "Algorithm": "Simulated Annealing (SP-Timing)", 
            "Mean Fitness": sa_mean_fitness, 
            "Std Dev Fitness": sa_std_fitness, 
            "Mean Exec Time (s)": sa_mean_exec_time,
            "Overall Best Fitness": float('nan'),
            "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
        })
    sa_section_end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Simulated Annealing section took {sa_section_end_time - sa_section_start_time:.2f} seconds.")


    # %% [markdown]
    # ## 3. Genetic Algorithms
    #
    # We will test multiple configurations of Genetic Algorithms, each run multiple times.

    # %%
    ga_section_start_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting Genetic Algorithms ({NUM_RUNS} runs each configuration) ---")

    ga_params_dict = {
        "population_size": 50,
        "generations": 1,  # MODIFIED FOR TIMING ESTIMATE
        "mutation_rate": 0.2,
        "elite_size": 2
    }
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GA Parameters: Pop={ga_params_dict['population_size']}, Gens={ga_params_dict['generations']}, MutRate={ga_params_dict['mutation_rate']}, Elite={ga_params_dict['elite_size']}")
    ga_configs_new = [
        {"name": "GA_Cfg1_AdSwapC_Ad1PtPV_SelTournK3", "mutation_op": mutate_swap_constrained, "crossover_op": crossover_one_point_prefer_valid, "selection_op": lambda p, f, k=3: selection_tournament_variable_k(p, f, k)},
        {"name": "GA_Cfg2_AdTargEx_AdUnifPV_SelRank", "mutation_op": mutate_targeted_player_exchange, "crossover_op": crossover_uniform_prefer_valid, "selection_op": selection_ranking},
        {"name": "GA_Cfg3_AdShufWTC_Ad1PtPV_SelBoltz", "mutation_op": mutate_shuffle_within_team_constrained, "crossover_op": crossover_one_point_prefer_valid, "selection_op": lambda p, f, t=100: selection_boltzmann(p, f, t)},
        {"name": "GA_Cfg4_AdTargEx_AdUnifPV_SelTournK5", "mutation_op": mutate_targeted_player_exchange, "crossover_op": crossover_uniform_prefer_valid, "selection_op": lambda p, f, k=5: selection_tournament_variable_k(p, f, k)},
        {"name": "GA_Cfg5_BaseSwap_Base1Pt_SelRank", "mutation_op": mutate_swap, "crossover_op": crossover_one_point, "selection_op": selection_ranking},
        {"name": "GA_Cfg6_BaseTeamShift_AdUnifPV_SelTournK3", "mutation_op": mutate_team_shift, "crossover_op": crossover_uniform_prefer_valid, "selection_op": lambda p, f, k=3: selection_tournament_variable_k(p, f, k)},
        {"name": "GA_Cfg7_AdSwapC_BaseUnif_SelBoltz", "mutation_op": mutate_swap_constrained, "crossover_op": crossover_uniform, "selection_op": lambda p, f, t=100: selection_boltzmann(p, f, t)},
        {"name": "GA_Cfg8_BaseShuffle_BaseUnif_SelTournK4", "mutation_op": mutate_shuffle_team, "crossover_op": crossover_uniform, "selection_op": lambda p, f, k=4: selection_tournament_variable_k(p, f, k)},
        {"name": "GA_Cfg9_AdTargEx_Base1Pt_SelRank", "mutation_op": mutate_targeted_player_exchange, "crossover_op": crossover_one_point, "selection_op": selection_ranking}
    ]

    for config in ga_configs_new:
        config_name = config["name"]
        print(f"  \n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting GA Configuration: {config_name}")
        ga_all_fitness_values_cfg = []
        ga_all_exec_times_cfg = []
        best_ga_solution_cfg_overall = None
        best_ga_fitness_cfg_overall = float("inf")
        best_ga_history_cfg_overall = []

        for i in range(NUM_RUNS):
            print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] {config_name} - Run {i+1}/{NUM_RUNS}...")
            start_time_ga_run = time.time()
            
            # Initial population generation now happens inside genetic_algorithm function
            best_solution_ga_run, history_ga_run = genetic_algorithm(
                players_data=players_data,
                num_teams=NUM_TEAMS,
                team_size=TEAM_SIZE,
                max_budget=MAX_BUDGET,
                population_size=ga_params_dict["population_size"],
                generations=ga_params_dict["generations"],
                mutation_rate=ga_params_dict["mutation_rate"],
                elite_size=ga_params_dict["elite_size"],
                mutation_operator_func=config["mutation_op"],
                crossover_operator_func=config["crossover_op"],
                selection_operator_func=config["selection_op"],
                verbose=False
            )
            end_time_ga_run = time.time()
            ga_exec_time_run = end_time_ga_run - start_time_ga_run

            if best_solution_ga_run:
                best_fitness_ga_run = best_solution_ga_run.fitness() # Calculate fitness from solution
                ga_all_fitness_values_cfg.append(best_fitness_ga_run)
                ga_all_exec_times_cfg.append(ga_exec_time_run)
                if best_fitness_ga_run < best_ga_fitness_cfg_overall:
                    best_ga_fitness_cfg_overall = best_fitness_ga_run
                    best_ga_solution_cfg_overall = best_solution_ga_run
                    best_ga_history_cfg_overall = history_ga_run
            else:
                print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] {config_name} - Run {i+1} did not find a valid solution.")
                ga_all_fitness_values_cfg.append(float('nan'))
                ga_all_exec_times_cfg.append(float('nan'))

        ga_mean_fitness_cfg = np.nanmean(ga_all_fitness_values_cfg) if ga_all_fitness_values_cfg else float("nan")
        ga_std_fitness_cfg = np.nanstd(ga_all_fitness_values_cfg) if ga_all_fitness_values_cfg else float("nan")
        ga_mean_exec_time_cfg = np.nanmean(ga_all_exec_times_cfg) if ga_all_exec_times_cfg else float("nan")

        print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] GA Configuration: {config_name} ({NUM_RUNS} runs) processing finished.")
        print(f"    Mean Best Fitness: {ga_mean_fitness_cfg:.4f}")
        print(f"    Std Dev Best Fitness: {ga_std_fitness_cfg:.4f}")
        print(f"    Mean Execution Time per run: {ga_mean_exec_time_cfg:.2f}s")

        if best_ga_solution_cfg_overall:
            print(f"    Overall Best Fitness for {config_name}: {best_ga_fitness_cfg_overall:.4f}")
            all_results_summary.append({
                "Algorithm": f"{config_name} (SP-Timing)", 
                "Mean Fitness": ga_mean_fitness_cfg, 
                "Std Dev Fitness": ga_std_fitness_cfg, 
                "Mean Exec Time (s)": ga_mean_exec_time_cfg,
                "Overall Best Fitness": best_ga_fitness_cfg_overall,
                "Mutation Op": config["mutation_op"].__name__,
                "Crossover Op": config["crossover_op"].__name__,
                "Selection Op": config["selection_op"].__name__ if hasattr(config["selection_op"], '__name__') else "lambda_tournament_or_boltzmann"
            })
            if best_ga_history_cfg_overall: # Check if history exists (it should if a solution was found)
                plt.figure(figsize=(12, 7))
                plt.plot(best_ga_history_cfg_overall, marker="o", linestyle="-")
                plt.title(f"GA Convergence: {config_name} (Best of {NUM_RUNS} Runs - SP-Timing)")
                plt.xlabel("Generation")
                plt.ylabel("Best Fitness in Population")
                plt.grid(True)
                plt.savefig(os.path.join(IMAGES_SP_DIR, f"ga_convergence_sp_timing_{config_name}.png")) # Updated path
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved GA convergence plot for {config_name} to {IMAGES_SP_DIR}/ga_convergence_sp_timing_{config_name}.png")
                plt.close()
            else:
                 print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] No convergence history to plot for {config_name} (likely no valid solutions found or history was empty).")
        else:
            print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] GA Configuration {config_name} did not find any valid solution across all runs that produced a best overall.")
            all_results_summary.append({
                "Algorithm": f"{config_name} (SP-Timing)", 
                "Mean Fitness": ga_mean_fitness_cfg, 
                "Std Dev Fitness": ga_std_fitness_cfg, 
                "Mean Exec Time (s)": ga_mean_exec_time_cfg,
                "Overall Best Fitness": float('nan'),
                "Mutation Op": config["mutation_op"].__name__,
                "Crossover Op": config["crossover_op"].__name__,
                "Selection Op": config["selection_op"].__name__ if hasattr(config["selection_op"], '__name__') else "lambda_tournament_or_boltzmann"
            })

    ga_section_end_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Genetic Algorithms section took {ga_section_end_time - ga_section_start_time:.2f} seconds.")


    # %% [markdown]
    # ## 4. Comparative Analysis and Reporting

    # %%
    reporting_start_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Generating Comparative Analysis Report ---")

    summary_df = pd.DataFrame(all_results_summary)
    summary_df_sorted = summary_df.sort_values(by=["Mean Fitness", "Mean Exec Time (s)"])
    csv_filename = os.path.join(IMAGES_SP_DIR, "comparative_results_sp_timing.csv") # Updated path
    summary_df_sorted.to_csv(csv_filename, index=False)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Comparative results saved to {csv_filename}")
    print("\nFinal Comparative Summary (SP-Timing):")
    print(summary_df_sorted.to_string())

    # Plotting comparative fitness
    plt.figure(figsize=(15, 8))
    summary_df_sorted.plot(kind="bar", x="Algorithm", y="Mean Fitness", yerr="Std Dev Fitness", capsize=4, legend=False, color="skyblue")
    plt.title("Mean Best Fitness of Algorithms (SP-Timing)")
    plt.ylabel("Mean Best Fitness (Lower is Better)")
    plt.xlabel("Algorithm Configuration")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_SP_DIR, "comparative_fitness_sp_timing.png")) # Updated path
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved comparative fitness plot to {IMAGES_SP_DIR}/comparative_fitness_sp_timing.png")
    plt.close()

    # Plotting comparative execution times
    plt.figure(figsize=(15, 8))
    summary_df_sorted.plot(kind="bar", x="Algorithm", y="Mean Exec Time (s)", legend=False, color="lightcoral")
    plt.title("Mean Execution Time of Algorithms (SP-Timing)")
    plt.ylabel("Mean Execution Time (seconds)")
    plt.xlabel("Algorithm Configuration")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_SP_DIR, "comparative_times_sp_timing.png")) # Updated path
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved comparative execution time plot to {IMAGES_SP_DIR}/comparative_times_sp_timing.png")
    plt.close()

    reporting_end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Reporting section took {reporting_end_time - reporting_start_time:.2f} seconds.")

    script_total_end_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Single-Processor Timing Estimate Script execution finished.")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total script execution time: {script_total_end_time - script_total_start_time:.2f} seconds.")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    
    # Ensure the directory for profiling results exists
    profiling_dir = os.path.join(TIMING_ESTIMATE_DIR, "profiling_results")
    if not os.path.exists(profiling_dir):
        os.makedirs(profiling_dir)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Created directory: {profiling_dir}")

    stats_file = os.path.join(profiling_dir, "timing_estimate_profile.prof")
    profiler.dump_stats(stats_file)
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Profiling statistics saved to {stats_file}")

    # Optional: Print human-readable stats to console
    # with open(os.path.join(profiling_dir, "timing_estimate_profile_readable.txt"), "w") as f:
    #     ps = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
    #     ps.print_stats()
    # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Readable profiling statistics saved to {profiling_dir}/timing_estimate_profile_readable.txt")

    # For a quick view, print top 20 cumulative time consumers
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Top 20 functions by cumulative time:")
    ps = pstats.Stats(profiler).sort_stats('cumulative')
    ps.print_stats(20)

