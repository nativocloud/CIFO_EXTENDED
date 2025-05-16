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

# Load player data (globally for worker processes to access if needed)
players_df = pd.read_csv("players.csv", sep=";")
PLAYERS_DATA_GLOBAL = players_df.to_dict(orient="records") # Renamed for clarity in multiprocessing context

# %% [markdown]
# ## Cell 2: Further Setup, Data Inspection, and Experiment Parameters

# %%
all_results_summary = []
# Define problem parameters
NUM_TEAMS = 5
TEAM_SIZE = 7
MAX_BUDGET = 750

# %% [markdown]
# ## Helper Functions for Multiprocessing Runs

# %%
def run_hill_climbing_trial(run_id, players_data_local, num_teams_local, team_size_local, max_budget_local, max_iterations_hc):
    """
    Execute a single Hill Climbing trial with the given parameters.
    
    Args:
        run_id: The ID of this run (for logging)
        players_data_local: The player data to use
        num_teams_local: Number of teams to create
        team_size_local: Size of each team
        max_budget_local: Maximum budget per team
        max_iterations_hc: Maximum iterations for hill climbing
        
    Returns:
        Tuple of (solution_object, fitness_value, execution_time, convergence_history)
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"  [{timestamp}] HC Run {run_id+1}/{NUM_RUNS} starting...")
    start_time_hc_run = time.time()
    
    # Create initial solution
    initial_hc_solution_run = LeagueHillClimbingSolution(players_data_local, num_teams=num_teams_local, team_size=team_size_local, max_budget=max_budget_local)
    retry_attempts_hc = 0
    max_retry_hc = 5
    
    # Retry if initial solution is invalid
    while not initial_hc_solution_run.is_valid() and retry_attempts_hc < max_retry_hc:
        initial_hc_solution_run = LeagueHillClimbingSolution(players_data_local, num_teams=num_teams_local, team_size=team_size_local, max_budget=max_budget_local)
        retry_attempts_hc += 1
    
    # Skip run if we couldn't create a valid initial solution
    if not initial_hc_solution_run.is_valid():
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"  [{timestamp}] HC Run {run_id+1} failed to create a valid initial solution after {max_retry_hc} retries. Skipping run.")
        return None, float("nan"), float("nan"), []

    # Run hill climbing
    hc_solution_obj_run, hc_fitness_val_run, hc_history_convergence_run = hill_climbing(
        initial_solution=initial_hc_solution_run, 
        players_data=players_data_local, 
        max_iterations=max_iterations_hc, 
        verbose=False
    )
    
    # Calculate execution time
    end_time_hc_run = time.time()
    hc_exec_time_run = end_time_hc_run - start_time_hc_run
    
    # Log completion
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"  [{timestamp}] HC Run {run_id+1} finished in {hc_exec_time_run:.2f}s with fitness {hc_fitness_val_run if hc_solution_obj_run else 'N/A'}.")
    
    return hc_solution_obj_run, hc_fitness_val_run if hc_solution_obj_run else float("nan"), hc_exec_time_run, hc_history_convergence_run

def run_simulated_annealing_trial(run_id, players_data_local, num_teams_local, team_size_local, max_budget_local, sa_params_local):
    """
    Execute a single Simulated Annealing trial with the given parameters.
    
    Args:
        run_id: The ID of this run (for logging)
        players_data_local: The player data to use
        num_teams_local: Number of teams to create
        team_size_local: Size of each team
        max_budget_local: Maximum budget per team
        sa_params_local: Dictionary of SA parameters (initial_temp, final_temp, alpha, iterations_per_temp)
        
    Returns:
        Tuple of (solution_object, fitness_value, execution_time, convergence_history)
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"  [{timestamp}] SA Run {run_id+1}/{NUM_RUNS} starting...")
    start_time_sa_run = time.time()
    
    # Create initial solution
    initial_sa_solution = LeagueSASolution(players_data_local, num_teams=num_teams_local, team_size=team_size_local, max_budget=max_budget_local)
    retry_attempts_sa = 0
    max_retry_sa = 5
    
    # Retry if initial solution is invalid
    while not initial_sa_solution.is_valid() and retry_attempts_sa < max_retry_sa:
        initial_sa_solution = LeagueSASolution(players_data_local, num_teams=num_teams_local, team_size=team_size_local, max_budget=max_budget_local)
        retry_attempts_sa += 1

    # Skip run if we couldn't create a valid initial solution
    if not initial_sa_solution.is_valid():
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"  [{timestamp}] SA Run {run_id+1} failed to create a valid initial solution after {max_retry_sa} retries. Skipping run.")
        return None, float("nan"), float("nan"), []

    # Run simulated annealing
    sa_solution_run, sa_fitness_run, sa_history_run = simulated_annealing(
        initial_solution=initial_sa_solution,
        players_data=players_data_local, 
        initial_temp=sa_params_local["initial_temp"],
        final_temp=sa_params_local["final_temp"],
        alpha=sa_params_local["alpha"],
        iterations_per_temp=sa_params_local["iterations_per_temp"],
        verbose=False
    )
    
    # Calculate execution time
    end_time_sa_run = time.time()
    sa_exec_time_run = end_time_sa_run - start_time_sa_run
    
    # Log completion
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"  [{timestamp}] SA Run {run_id+1} finished in {sa_exec_time_run:.2f}s with fitness {sa_fitness_run if sa_solution_run else 'N/A'}.")
    
    return sa_solution_run, sa_fitness_run if sa_solution_run else float("nan"), sa_exec_time_run, sa_history_run

def run_genetic_algorithm_trial(run_id, players_data_local, num_teams_local, team_size_local, max_budget_local, ga_config):
    """
    Execute a single Genetic Algorithm trial with the given parameters.
    
    Args:
        run_id: The ID of this run (for logging)
        players_data_local: The player data to use
        num_teams_local: Number of teams to create
        team_size_local: Size of each team
        max_budget_local: Maximum budget per team
        ga_config: Dictionary containing GA configuration parameters
        
    Returns:
        Tuple of (solution_object, fitness_value, execution_time, convergence_history)
    """
    config_name = ga_config["name"]
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"  [{timestamp}] GA Run {run_id+1}/{NUM_RUNS} for {config_name} starting...")
    start_time_ga_run = time.time()
    
    # Run genetic algorithm
    best_solution_ga_run, fitness_history_ga_run = genetic_algorithm(
        players_data=players_data_local, 
        population_size=ga_config["pop_size"],
        generations=ga_config["gens"],
        mutation_rate=ga_config["mut_rate"],
        elite_size=ga_config["elite"],
        mutation_operator_func=ga_config["mut_op"], 
        crossover_operator_func=ga_config["cross_op"],
        selection_operator_func=ga_config["sel_op"],
        tournament_k=ga_config["tourn_k"] if ga_config["tourn_k"] is not None else 3,
        boltzmann_temp=ga_config["boltz_temp"] if ga_config["boltz_temp"] is not None else 100,
        num_teams=num_teams_local, 
        team_size=team_size_local,
        max_budget=max_budget_local,
        verbose=False 
    )
    
    # Calculate execution time
    end_time_ga_run = time.time()
    actual_exec_time_ga_run = end_time_ga_run - start_time_ga_run 
    
    # Calculate fitness if solution exists
    best_fitness_ga_run = best_solution_ga_run.fitness() if best_solution_ga_run else float("nan")
    
    # Log completion
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"  [{timestamp}] GA Run {run_id+1} for {config_name} finished in {actual_exec_time_ga_run:.2f}s with fitness {best_fitness_ga_run if best_solution_ga_run else 'N/A'}.")
    
    return best_solution_ga_run, best_fitness_ga_run, actual_exec_time_ga_run, fitness_history_ga_run

# %% [markdown]
# ## Main Execution Block (Multiprocessing)

# %%
def main():
    """Main execution function for the multiprocessing script."""
    script_total_start_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Multiprocessing Script for {NUM_RUNS} runs execution started.")

    # Log player data info
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Player data loaded. Total players: {len(PLAYERS_DATA_GLOBAL)}")
    if PLAYERS_DATA_GLOBAL:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] First player data: {PLAYERS_DATA_GLOBAL[0]}")
    
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] All algorithms (HC, SA, GA) will be run {NUM_RUNS} times each in parallel (where applicable).")
    
    all_results_summary = []
    
    # Determine number of processes to use
    num_processes = min(NUM_RUNS, os.cpu_count() if os.cpu_count() else 1)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Using {num_processes} processes for parallel execution.")

    # --- 1. Hill Climbing --- 
    hc_section_start_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] --- Starting Hill Climbing Algorithm ({NUM_RUNS} runs in parallel) ---")
    
    # Prepare parameters for each HC run
    hc_params_for_runs = [(i, PLAYERS_DATA_GLOBAL, NUM_TEAMS, TEAM_SIZE, MAX_BUDGET, 1000) for i in range(NUM_RUNS)]
    
    # Execute HC runs in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        hc_results_parallel = pool.starmap(run_hill_climbing_trial, hc_params_for_runs)

    # Extract results
    hc_all_solutions = [res[0] for res in hc_results_parallel if res is not None and res[0] is not None]
    hc_all_fitness_values = [res[1] for res in hc_results_parallel if res is not None]
    hc_all_exec_times = [res[2] for res in hc_results_parallel if res is not None]
    hc_all_histories = [res[3] for res in hc_results_parallel if res is not None]

    # Find best HC solution
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
    
    # Calculate HC statistics
    hc_mean_fitness = np.nanmean([f for f in hc_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in hc_all_fitness_values) else float("nan")
    hc_std_fitness = np.nanstd([f for f in hc_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in hc_all_fitness_values) else float("nan")
    hc_mean_exec_time = np.nanmean([t for t in hc_all_exec_times if not np.isnan(t)]) if any(not np.isnan(t) for t in hc_all_exec_times) else float("nan")

    # Log HC results
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
        
        # Plot HC convergence
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
    
    # Prepare SA parameters
    sa_params = {
        "initial_temp": 1000,
        "final_temp": 0.1,
        "alpha": 0.99,
        "iterations_per_temp": 50
    }
    
    # Prepare parameters for each SA run
    sa_params_for_runs = [(i, PLAYERS_DATA_GLOBAL, NUM_TEAMS, TEAM_SIZE, MAX_BUDGET, sa_params) for i in range(NUM_RUNS)]

    # Execute SA runs in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        sa_results_parallel = pool.starmap(run_simulated_annealing_trial, sa_params_for_runs)

    # Extract results
    sa_all_solutions = [res[0] for res in sa_results_parallel if res is not None and res[0] is not None]
    sa_all_fitness_values = [res[1] for res in sa_results_parallel if res is not None]
    sa_all_exec_times = [res[2] for res in sa_results_parallel if res is not None]
    sa_all_histories = [res[3] for res in sa_results_parallel if res is not None]

    # Find best SA solution
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

    # Calculate SA statistics
    sa_mean_fitness = np.nanmean([f for f in sa_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in sa_all_fitness_values) else float("nan")
    sa_std_fitness = np.nanstd([f for f in sa_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in sa_all_fitness_values) else float("nan")
    sa_mean_exec_time = np.nanmean([t for t in sa_all_exec_times if not np.isnan(t)]) if any(not np.isnan(t) for t in sa_all_exec_times) else float("nan")

    # Log SA results
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
        
        # Plot SA convergence
        if best_sa_history_overall:
            plt.figure(figsize=(10, 6))
            plt.plot(best_sa_history_overall, marker=".", linestyle="-")
            plt.title(f"Simulated Annealing Convergence (Best of {NUM_RUNS} MP Runs)")
            plt.xlabel("Iteration")
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

    # --- 3. Genetic Algorithm Configurations ---
    ga_section_start_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] --- Starting Genetic Algorithm Configurations ({NUM_RUNS} runs each in parallel) ---")

    # Define GA configurations
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

    # Process each GA configuration
    for ga_config in ga_configurations:
        config_name = ga_config["name"]
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n  [{timestamp}] Running GA Configuration: {config_name}")
        
        # Prepare parameters for each GA run
        ga_params_for_runs = [(i, PLAYERS_DATA_GLOBAL, NUM_TEAMS, TEAM_SIZE, MAX_BUDGET, ga_config) for i in range(NUM_RUNS)]
        
        # Execute GA runs in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            ga_results_parallel = pool.starmap(run_genetic_algorithm_trial, ga_params_for_runs)
        
        # Extract results
        ga_all_solutions = [res[0] for res in ga_results_parallel if res is not None and res[0] is not None]
        ga_all_fitness_values = [res[1] for res in ga_results_parallel if res is not None]
        ga_all_exec_times = [res[2] for res in ga_results_parallel if res is not None]
        ga_all_histories = [res[3] for res in ga_results_parallel if res is not None]
        
        # Find best GA solution for this configuration
        best_ga_fitness_config = float("inf")
        best_ga_solution_config = None
        best_ga_history_config = []
        
        if ga_all_solutions:
            for i, fit_val in enumerate(ga_all_fitness_values):
                if not np.isnan(fit_val) and fit_val < best_ga_fitness_config:
                    if i < len(ga_all_solutions) and i < len(ga_all_histories):
                        best_ga_fitness_config = fit_val
                        best_ga_solution_config = ga_all_solutions[i]
                        best_ga_history_config = ga_all_histories[i]
        
        # Calculate GA statistics for this configuration
        ga_mean_fitness = np.nanmean([f for f in ga_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in ga_all_fitness_values) else float("nan")
        ga_std_fitness = np.nanstd([f for f in ga_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in ga_all_fitness_values) else float("nan")
        ga_mean_exec_time = np.nanmean([t for t in ga_all_exec_times if not np.isnan(t)]) if any(not np.isnan(t) for t in ga_all_exec_times) else float("nan")
        
        # Log GA results for this configuration
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"  [{timestamp}] GA Configuration {config_name} ({NUM_RUNS} runs) processing finished.")
        print(f"    Mean Best Fitness: {ga_mean_fitness:.4f}")
        print(f"    Std Dev Best Fitness: {ga_std_fitness:.4f}")
        print(f"    Mean Execution Time per run: {ga_mean_exec_time:.2f}s")
        
        if best_ga_solution_config:
            print(f"    Overall Best GA Fitness for {config_name}: {best_ga_fitness_config:.4f}")
            all_results_summary.append({
                "Algorithm": f"GA: {config_name} (MP-{NUM_RUNS} runs)",
                "Mean Fitness": ga_mean_fitness,
                "Std Dev Fitness": ga_std_fitness,
                "Mean Exec Time (s)": ga_mean_exec_time,
                "Overall Best Fitness": best_ga_fitness_config,
                "Mutation Op": ga_config["mut_op"].__name__,
                "Crossover Op": ga_config["cross_op"].__name__,
                "Selection Op": ga_config["sel_op"].__name__
            })
            
            # Plot GA convergence for this configuration
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
                print(f"    [{timestamp}] GA Config {config_name} - Fitness history for best run is not in the expected format for plotting.")
            else:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"    [{timestamp}] No valid fitness history to plot for GA config {config_name}.")
        else:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"    [{timestamp}] GA Config {config_name} did not find any valid solution across all runs that produced a best overall.")
            all_results_summary.append({
                "Algorithm": f"GA: {config_name} (MP-{NUM_RUNS} runs)",
                "Mean Fitness": ga_mean_fitness,
                "Std Dev Fitness": ga_std_fitness,
                "Mean Exec Time (s)": ga_mean_exec_time,
                "Overall Best Fitness": float("nan"),
                "Mutation Op": ga_config["mut_op"].__name__,
                "Crossover Op": ga_config["cross_op"].__name__,
                "Selection Op": ga_config["sel_op"].__name__
            })
    
    ga_section_end_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] All Genetic Algorithm configurations took {ga_section_end_time - ga_section_start_time:.2f} seconds (wall time for parallel execution).")

    # --- 4. Save Summary Results ---
    # Convert summary to DataFrame and save
    summary_df = pd.DataFrame(all_results_summary)
    summary_csv_path = os.path.join(IMAGES_MP_DIR, f"all_algorithms_summary_mp_{NUM_RUNS}runs.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Saved summary results to {summary_csv_path}")

    # --- 5. Comparative Plots ---
    # Plot comparative fitness
    plt.figure(figsize=(14, 8))
    algorithms = summary_df["Algorithm"].tolist()
    mean_fitness = summary_df["Mean Fitness"].tolist()
    std_fitness = summary_df["Std Dev Fitness"].tolist()
    
    # Create bar positions
    bar_positions = np.arange(len(algorithms))
    bar_width = 0.6
    
    # Create bars with error bars
    plt.bar(bar_positions, mean_fitness, bar_width, 
            yerr=std_fitness, 
            capsize=5, 
            color='skyblue', 
            edgecolor='black',
            alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Algorithm')
    plt.ylabel('Mean Fitness (Std Dev of Avg Team Skills)')
    plt.title(f'Comparative Mean Fitness Across Algorithms (MP-{NUM_RUNS} runs)')
    plt.xticks(bar_positions, algorithms, rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.savefig(os.path.join(IMAGES_MP_DIR, f"comparative_fitness_mp_{NUM_RUNS}runs.png"))
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Saved comparative fitness plot to {IMAGES_MP_DIR}/comparative_fitness_mp_{NUM_RUNS}runs.png")
    plt.close()
    
    # Plot comparative execution times
    plt.figure(figsize=(14, 8))
    exec_times = summary_df["Mean Exec Time (s)"].tolist()
    
    # Create bars
    plt.bar(bar_positions, exec_times, bar_width, 
            color='lightgreen', 
            edgecolor='black',
            alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Algorithm')
    plt.ylabel('Mean Execution Time (seconds)')
    plt.title(f'Comparative Mean Execution Times Across Algorithms (MP-{NUM_RUNS} runs)')
    plt.xticks(bar_positions, algorithms, rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.savefig(os.path.join(IMAGES_MP_DIR, f"comparative_times_mp_{NUM_RUNS}runs.png"))
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Saved comparative execution times plot to {IMAGES_MP_DIR}/comparative_times_mp_{NUM_RUNS}runs.png")
    plt.close()
    
    # Log script completion
    script_total_end_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Multiprocessing Script execution completed in {script_total_end_time - script_total_start_time:.2f} seconds.")

# Execute main function if script is run directly
if __name__ == "__main__":
    main()
