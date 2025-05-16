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
# This work presents a constrained combinatorial optimization approach to the **Sports League Assignment Problem** using **Genetic Algorithms (GAs)**, **Hill Climbing (HC)**, and **Simulated Annealing (SA)**. The objective is to allocate a fixed pool of professional players into a set of 5 structurally valid teams in such a way that the **standard deviation of the teams\" average skill ratings** is minimized—promoting competitive balance across the league.
#
# Each player is defined by three attributes: **position** (one of `GK`, `DEF`, `MID`, `FWD`), **skill rating** (a numerical measure of ability), and **cost** (in million euros). A valid solution must satisfy the following **hard constraints**:
#
# - Each team must consist of exactly **7 players**, with a specific positional structure: **1 GK, 2 DEF, 2 MID, and 2 FWD**
# - Each team must have a **total cost ≤ 750 million €**
# - Each player must be assigned to **exactly one team** (no overlaps)
#
# The **search space** is therefore highly constrained and discrete, and infeasible configurations are explicitly excluded from the solution space. The optimization objective is to identify league configurations where teams are not only valid but also **skill-balanced**, quantified by the **standard deviation of average skill ratings across teams**, which serves as the **fitness function** (to be minimized).
#
# This script adapts the single-processor version to utilize **multiprocessing** to run multiple independent trials of each algorithm concurrently, allowing for more robust statistical analysis of their performance within a shorter timeframe.

# %% [markdown]
# ## Cell 1: Setup and Critical Data Loading

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os # Added for path joining
import multiprocessing # Added for multiprocessing
# cProfile and pstats are generally for single-process profiling, might be complex with multiprocessing
# For now, we focus on timing the parallel runs themselves.

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
# Define problem parameters
NUM_TEAMS = 5
TEAM_SIZE = 7
MAX_BUDGET = 750

# Define number of runs for stochastic algorithms
NUM_RUNS = 1 # Parameter for number of runs (e.g., 5, 10, 30)

# %% [markdown]
# ## Helper Functions for Multiprocessing Runs

# %%
def run_hill_climbing_trial(run_id, players_data_local, num_teams_local, team_size_local, max_budget_local, max_iterations_hc):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"  [{timestamp}] HC Run {run_id+1}/{NUM_RUNS} starting...")
    start_time_hc_run = time.time()
    
    initial_hc_solution_run = LeagueHillClimbingSolution(players_data_local, num_teams=num_teams_local, team_size=team_size_local, max_budget=max_budget_local)
    retry_attempts_hc = 0
    max_retry_hc = 5
    while not initial_hc_solution_run.is_valid() and retry_attempts_hc < max_retry_hc:
        initial_hc_solution_run = LeagueHillClimbingSolution(players_data_local, num_teams=num_teams_local, team_size=team_size_local, max_budget=max_budget_local)
        retry_attempts_hc += 1
    
    if not initial_hc_solution_run.is_valid():
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"  [{timestamp}] HC Run {run_id+1} failed to create a valid initial solution after {max_retry_hc} retries. Skipping run.")
        return None, float("nan"), float("nan"), []

    hc_solution_obj_run, hc_fitness_val_run, hc_history_convergence_run = hill_climbing(
        initial_solution=initial_hc_solution_run, 
        players_data=players_data_local, 
        max_iterations=max_iterations_hc, 
        verbose=False
    )
    end_time_hc_run = time.time()
    hc_exec_time_run = end_time_hc_run - start_time_hc_run
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"  [{timestamp}] HC Run {run_id+1} finished in {hc_exec_time_run:.2f}s with fitness {hc_fitness_val_run if hc_solution_obj_run else 'N/A'}.")
    return hc_solution_obj_run, hc_fitness_val_run if hc_solution_obj_run else float("nan"), hc_exec_time_run, hc_history_convergence_run

def run_simulated_annealing_trial(run_id, players_data_local, num_teams_local, team_size_local, max_budget_local, sa_params_local):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"  [{timestamp}] SA Run {run_id+1}/{NUM_RUNS} starting...")
    start_time_sa_run = time.time()
    initial_sa_solution = LeagueSASolution(players_data_local, num_teams=num_teams_local, team_size=team_size_local, max_budget=max_budget_local)
    retry_attempts_sa = 0
    max_retry_sa = 5
    while not initial_sa_solution.is_valid() and retry_attempts_sa < max_retry_sa:
        initial_sa_solution = LeagueSASolution(players_data_local, num_teams=num_teams_local, team_size=team_size_local, max_budget=max_budget_local)
        retry_attempts_sa += 1

    if not initial_sa_solution.is_valid():
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"  [{timestamp}] SA Run {run_id+1} failed to create a valid initial solution after {max_retry_sa} retries. Skipping run.")
        return None, float("nan"), float("nan"), []

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
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"  [{timestamp}] SA Run {run_id+1} finished in {sa_exec_time_run:.2f}s with fitness {sa_fitness_run if sa_solution_run else 'N/A'}.")
    return sa_solution_run, sa_fitness_run if sa_solution_run else float("nan"), sa_exec_time_run, sa_history_run

def run_genetic_algorithm_trial(run_id, players_data_local, num_teams_local, team_size_local, max_budget_local, ga_params_local, config_name_local):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"  [{timestamp}] GA Run {run_id+1}/{NUM_RUNS} for {config_name_local} starting...")
    start_time_ga_run = time.time()
    
    best_solution_ga_run, fitness_history_ga_run = genetic_algorithm(
        players_data=players_data_local, 
        population_size=ga_params_local["population_size"],
        generations=ga_params_local["generations"],
        mutation_rate=ga_params_local["mutation_rate"],
        elite_size=ga_params_local["elitism_size"],
        mutation_operator_func=ga_params_local["mutation_op"], 
        crossover_operator_func=ga_params_local["crossover_op"],
        selection_operator_func=ga_params_local["selection_op"],
        tournament_k=ga_params_local.get("tournament_k", 3),
        boltzmann_temp=ga_params_local.get("boltzmann_temp", 100),
        num_teams=num_teams_local, 
        team_size=team_size_local,
        max_budget=max_budget_local,
        verbose=False 
    )
    
    end_time_ga_run = time.time()
    actual_exec_time_ga_run = end_time_ga_run - start_time_ga_run 
    best_fitness_ga_run = best_solution_ga_run.fitness() if best_solution_ga_run else float("nan")
    
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"  [{timestamp}] GA Run {run_id+1} for {config_name_local} finished in {actual_exec_time_ga_run:.2f}s with fitness {best_fitness_ga_run if best_solution_ga_run else 'N/A'}.")
    return best_solution_ga_run, best_fitness_ga_run, actual_exec_time_ga_run, fitness_history_ga_run


# %% [markdown]
# ## Main Execution Block (Adapted for Multiprocessing)

# %%
def main():
    script_total_start_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Multiprocessing Script for {NUM_RUNS} runs execution started.")

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Player data loaded. Total players: {len(PLAYERS_DATA_GLOBAL)}")
    if PLAYERS_DATA_GLOBAL:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] First player data: {PLAYERS_DATA_GLOBAL[0]}")
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] All algorithms (HC, SA, GA) will be run {NUM_RUNS} times each in parallel (where applicable).")
    
    all_results_summary = []
    num_processes = min(NUM_RUNS, os.cpu_count() if os.cpu_count() else 1)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Using {num_processes} processes for parallel execution.")


    # --- 1. Hill Climbing --- 
    hc_section_start_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] --- Starting Hill Climbing Algorithm ({NUM_RUNS} runs in parallel) ---")
    
    hc_params_for_runs = [(i, PLAYERS_DATA_GLOBAL, NUM_TEAMS, TEAM_SIZE, MAX_BUDGET, 1000) for i in range(NUM_RUNS)]
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        hc_results_parallel = pool.starmap(run_hill_climbing_trial, hc_params_for_runs)

    hc_all_solutions = [res[0] for res in hc_results_parallel if res is not None and res[0] is not None]
    hc_all_fitness_values = [res[1] for res in hc_results_parallel if res is not None]
    hc_all_exec_times = [res[2] for res in hc_results_parallel if res is not None]
    hc_all_histories = [res[3] for res in hc_results_parallel if res is not None]

    best_hc_fitness_overall = float("inf")
    best_hc_solution_overall = None
    best_hc_history_overall = []
    if hc_all_solutions:
        for i, fit_val in enumerate(hc_all_fitness_values):
            if not np.isnan(fit_val) and fit_val < best_hc_fitness_overall:
                if i < len(hc_all_solutions) and i < len(hc_all_histories):
                    best_hc_fitness_overall = fit_val
                    best_hc_solution_overall = hc_all_solutions[i] 
                    best_hc_history_overall = hc_all_histories[i]
    
    hc_mean_fitness = np.nanmean([f for f in hc_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in hc_all_fitness_values) else float("nan")
    hc_std_fitness = np.nanstd([f for f in hc_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in hc_all_fitness_values) else float("nan")
    hc_mean_exec_time = np.nanmean([t for t in hc_all_exec_times if not np.isnan(t)]) if any(not np.isnan(t) for t in hc_all_exec_times) else float("nan")

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
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
        if best_hc_history_overall:
            plt.figure(figsize=(10, 6))
            plt.plot(best_hc_history_overall, marker="o", linestyle="-")
            plt.title(f"Hill Climbing Convergence (Best of {NUM_RUNS} MP Runs)")
            plt.xlabel("Improvement Step")
            plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
            plt.grid(True)
            plt.savefig(os.path.join(IMAGES_MP_DIR, f"hc_convergence_mp_{NUM_RUNS}runs.png"))
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] Saved Hill Climbing convergence plot to {IMAGES_MP_DIR}/hc_convergence_mp_{NUM_RUNS}runs.png")
            plt.close()
    else:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] Hill Climbing did not find any valid solution across all runs that produced a best overall.")
        all_results_summary.append({
            "Algorithm": f"Hill Climbing (MP-{NUM_RUNS} runs)", 
            "Mean Fitness": hc_mean_fitness, 
            "Std Dev Fitness": hc_std_fitness, 
            "Mean Exec Time (s)": hc_mean_exec_time,
            "Overall Best Fitness": float("nan"),
            "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
        })
    hc_section_end_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Hill Climbing section took {hc_section_end_time - hc_section_start_time:.2f} seconds (wall time for parallel execution).")


    # --- 2. Simulated Annealing --- 
    sa_section_start_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] --- Starting Simulated Annealing Algorithm ({NUM_RUNS} runs in parallel) ---")
    
    sa_params = {
        "initial_temp": 1000,
        "final_temp": 0.1,
        "alpha": 0.99,
        "iterations_per_temp": 50
    }
    sa_params_for_runs = [(i, PLAYERS_DATA_GLOBAL, NUM_TEAMS, TEAM_SIZE, MAX_BUDGET, sa_params) for i in range(NUM_RUNS)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        sa_results_parallel = pool.starmap(run_simulated_annealing_trial, sa_params_for_runs)

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
                if i < len(sa_all_solutions) and i < len(sa_all_histories):
                    best_sa_fitness_overall = fit_val
                    best_sa_solution_overall = sa_all_solutions[i]
                    best_sa_history_overall = sa_all_histories[i]

    sa_mean_fitness = np.nanmean([f for f in sa_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in sa_all_fitness_values) else float("nan")
    sa_std_fitness = np.nanstd([f for f in sa_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in sa_all_fitness_values) else float("nan")
    sa_mean_exec_time = np.nanmean([t for t in sa_all_exec_times if not np.isnan(t)]) if any(not np.isnan(t) for t in sa_all_exec_times) else float("nan")

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
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
            plt.plot(best_sa_history_overall, marker=".", linestyle="-")
            plt.title(f"Simulated Annealing Convergence (Best of {NUM_RUNS} MP Runs)")
            plt.xlabel("Temperature Step / Iteration")
            plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
            plt.grid(True)
            plt.savefig(os.path.join(IMAGES_MP_DIR, f"sa_convergence_mp_{NUM_RUNS}runs.png"))
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] Saved Simulated Annealing convergence plot to {IMAGES_MP_DIR}/sa_convergence_mp_{NUM_RUNS}runs.png")
            plt.close()
    else:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
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
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Simulated Annealing section took {sa_section_end_time - sa_section_start_time:.2f} seconds (wall time for parallel execution).")


    # --- 3. Genetic Algorithms --- 
    ga_section_start_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] --- Starting Genetic Algorithms ({NUM_RUNS} runs per config in parallel) ---")

    ga_configurations = {
        "GA_Config_1_SwapConst1PtPreferVTournVarK": {
            "population_size": 50,
            "generations": 100,
            "mutation_op": mutate_swap_constrained,
            "mutation_rate": 0.1,
            "crossover_op": crossover_one_point_prefer_valid,
            "selection_op": selection_tournament_variable_k,
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
            "elitism_size": 1
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
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"  [{timestamp}] Processing GA Configuration: {config_name} ({NUM_RUNS} runs in parallel)")

        ga_params_for_runs = [(i, PLAYERS_DATA_GLOBAL, NUM_TEAMS, TEAM_SIZE, MAX_BUDGET, ga_params_dict, config_name) for i in range(NUM_RUNS)]

        with multiprocessing.Pool(processes=num_processes) as pool:
            ga_results_parallel = pool.starmap(run_genetic_algorithm_trial, ga_params_for_runs)
        
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
                    if i < len(ga_all_solutions_config) and i < len(ga_all_histories_config):
                        best_ga_fitness_config = fit_val
                        best_ga_solution_config = ga_all_solutions_config[i]
                        best_ga_history_config = ga_all_histories_config[i]

        ga_mean_fitness_config = np.nanmean([f for f in ga_all_fitness_values_config if not np.isnan(f)]) if any(not np.isnan(f) for f in ga_all_fitness_values_config) else float("nan")
        ga_std_fitness_config = np.nanstd([f for f in ga_all_fitness_values_config if not np.isnan(f)]) if any(not np.isnan(f) for f in ga_all_fitness_values_config) else float("nan")
        ga_mean_exec_time_config = np.nanmean([t for t in ga_all_exec_times_config if not np.isnan(t)]) if any(not np.isnan(t) for t in ga_all_exec_times_config) else float("nan")

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"    [{timestamp}] GA Config {config_name} ({NUM_RUNS} runs) - Mean Fitness: {ga_mean_fitness_config:.4f}, Std Dev Fitness: {ga_std_fitness_config:.4f}, Mean Exec Time: {ga_mean_exec_time_config:.2f}s")
        if best_ga_solution_config:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
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
                plt.plot(best_ga_history_config, marker="o", linestyle="-", markersize=4)
                plt.title(f"GA Convergence: {config_name} (Best of {NUM_RUNS} MP Runs)")
                plt.xlabel("Generation")
                plt.ylabel("Best Fitness in Population")
                plt.grid(True)
                plt.savefig(os.path.join(IMAGES_MP_DIR, f"ga_convergence_{config_name}_mp_{NUM_RUNS}runs.png"))
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"    [{timestamp}] Saved GA convergence plot to {IMAGES_MP_DIR}/ga_convergence_{config_name}_mp_{NUM_RUNS}runs.png")
                plt.close()
            elif best_ga_history_config:
                 timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                 print(f"    [{timestamp}] GA Config {config_name} - Fitness history for best run is not in the expected format for plotting (e.g. list of numbers). Type: {type(best_ga_history_config[0])}")
            else:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"    [{timestamp}] No valid fitness history to plot for GA config {config_name}.")
        else:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"    [{timestamp}] GA Config {config_name} did not find any valid solution across all runs that produced a best overall.")
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
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"    [{timestamp}] GA Config {config_name} section took {config_end_time - config_start_time:.2f} seconds (wall time for parallel execution).")

    ga_section_end_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Genetic Algorithms section took {ga_section_end_time - ga_section_start_time:.2f} seconds (wall time for all configs).")

# %% [markdown]
#     # ## Results Summary and Comparison

    # %%
    results_df = pd.DataFrame(all_results_summary)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] --- Overall Results Summary ({NUM_RUNS} MP Runs) ---")
    print(results_df.to_string())
    results_df.to_csv(os.path.join(IMAGES_MP_DIR, f"all_algorithms_summary_mp_{NUM_RUNS}runs.csv"), index=False)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Saved results summary to {IMAGES_MP_DIR}/all_algorithms_summary_mp_{NUM_RUNS}runs.csv")

    if not results_df.empty:
        plt.figure(figsize=(14, 8))
        plot_df_times = results_df.dropna(subset=["Mean Exec Time (s)"])
        if not plot_df_times.empty:
            alg_names_times = [name.replace(f" (MP-{NUM_RUNS} runs)", "") for name in plot_df_times["Algorithm"]]
            bars = plt.bar(alg_names_times, plot_df_times["Mean Exec Time (s)"], color=plt.cm.viridis(np.linspace(0, 1, len(plot_df_times))))
            plt.xlabel("Algorithm / Configuration")
            plt.ylabel("Mean Execution Time per Run (s)")
            plt.title(f"Comparative Mean Execution Times ({NUM_RUNS} MP Runs)")
            plt.xticks(rotation=60, ha="right")
            plt.tight_layout()
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * yval, f'{yval:.2f}s', ha='center', va='bottom')
            plt.savefig(os.path.join(IMAGES_MP_DIR, f"comparative_times_mp_{NUM_RUNS}runs.png"))
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] Saved comparative execution times plot.")
            plt.close()
        else:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] No valid data for comparative execution times plot after dropping NaNs.")

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
                plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * yval, f'{yval:.4f}', ha='center', va='bottom')
            plt.savefig(os.path.join(IMAGES_MP_DIR, f"comparative_fitness_mp_{NUM_RUNS}runs.png"))
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] Saved comparative best fitness plot.")
            plt.close()
        else:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] No valid data for comparative best fitness plot after dropping NaNs.")

    else:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] No results to plot for summary.")

    # --- 4. Save Summary Results ---
    if all_results_summary:
        results_df = pd.DataFrame(all_results_summary)
        summary_file_path = os.path.join(IMAGES_MP_DIR, f"all_algorithms_summary_mp_{NUM_RUNS}runs.csv")
        results_df.to_csv(summary_file_path, index=False)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] Saved all algorithms summary to {summary_file_path}")
    else:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] No results to save in summary CSV.")

    script_total_end_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] Multiprocessing Script execution finished.")
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Total script execution time: {script_total_end_time - script_total_start_time:.2f} seconds.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
