# -*- coding: utf-8 -*-
# %%
# %% [markdown]
# # Introduction
#
# This work presents a constrained combinatorial optimization approach to the **Sports League Assignment Problem** using **Genetic Algorithms (GAs)** and **Hill Climbing (HC)**. The objective is to allocate a fixed pool of professional players into a set of 5 structurally valid teams in such a way that the **standard deviation of the teams\" average skill ratings** is minimized—promoting competitive balance across the league.
#
# Each player is defined by three attributes: **position** (one of `GK`, `DEF`, `MID`, `FWD`), **skill rating** (a numerical measure of ability), and **cost** (in million euros). A valid solution must satisfy the following **hard constraints**:
#
# - Each team must consist of exactly **7 players**, with a specific positional structure: **1 GK, 2 DEF, 2 MID, and 2 FWD**
# - Each team must have a **total cost ≤ 750 million €**
# - Each player must be assigned to **exactly one team** (no overlaps)
#
# The **search space** is therefore highly constrained and discrete, and infeasible configurations are explicitly excluded from the solution space. The optimization objective is to identify league configurations where teams are not only valid but also **skill-balanced**, quantified by the **standard deviation of average skill ratings across teams**, which serves as the **fitness function** (to be minimized).
#
# This script utilizes **multiprocessing** to run multiple independent trials of each algorithm configuration concurrently, allowing for robust statistical analysis of their performance within a shorter timeframe. This version specifically focuses on parameter variation for Hill Climbing and two selected Genetic Algorithm configurations.

# %% [markdown]
# ## Cell 1: Setup and Critical Data Loading

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os # Added for path joining
import multiprocessing # Added for multiprocessing

from solution import LeagueSolution, LeagueHillClimbingSolution, LeagueSASolution # LeagueSASolution might not be used directly but evolution.py might import it.
from evolution import genetic_algorithm, hill_climbing # Removed simulated_annealing as it's not part of this script's focus
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
# MODIFIED FOR FINAL PARAMETER VARIATION
MULTIPROCESSING_RESULTS_DIR = "/home/ubuntu/CIFO_EXTENDED_Project/images_mp/final_param_var_results"
IMAGES_MP_DIR = MULTIPROCESSING_RESULTS_DIR # Adjusted for this script
# Ensure the directory exists
if not os.path.exists(IMAGES_MP_DIR):
    os.makedirs(IMAGES_MP_DIR)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
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
# MODIFIED FOR 30 RUNS PER PARAMETER VARIATION
NUM_RUNS = 30 

# %% [markdown]
# ## Helper Functions for Multiprocessing Runs

# %%
def run_hill_climbing_trial(run_id, players_data_local, num_teams_local, team_size_local, max_budget_local, max_iterations_hc, variation_name_hc):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  [{timestamp}] HC Run {run_id+1}/{NUM_RUNS} for {variation_name_hc} starting (MaxIter: {max_iterations_hc})...")
    start_time_hc_run = time.time()
    
    initial_hc_solution_run = LeagueHillClimbingSolution(players_data_local, num_teams=num_teams_local, team_size=team_size_local, max_budget=max_budget_local)
    retry_attempts_hc = 0
    max_retry_hc = 5
    while not initial_hc_solution_run.is_valid() and retry_attempts_hc < max_retry_hc:
        initial_hc_solution_run = LeagueHillClimbingSolution(players_data_local, num_teams=num_teams_local, team_size=team_size_local, max_budget=max_budget_local)
        retry_attempts_hc += 1
    
    if not initial_hc_solution_run.is_valid():
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{timestamp}] HC Run {run_id+1} for {variation_name_hc} failed to create a valid initial solution after {max_retry_hc} retries. Skipping run.")
        return None, float("nan"), float("nan"), []

    hc_solution_obj_run, hc_fitness_val_run, hc_history_convergence_run = hill_climbing(
        initial_solution=initial_hc_solution_run, 
        players_data=players_data_local, 
        max_iterations=max_iterations_hc, 
        verbose=False
    )
    end_time_hc_run = time.time()
    hc_exec_time_run = end_time_hc_run - start_time_hc_run
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  [{timestamp}] HC Run {run_id+1} for {variation_name_hc} finished in {hc_exec_time_run:.2f}s with fitness {hc_fitness_val_run if hc_solution_obj_run else 'N/A'}.")
    return hc_solution_obj_run, hc_fitness_val_run if hc_solution_obj_run else float("nan"), hc_exec_time_run, hc_history_convergence_run

def run_genetic_algorithm_trial(run_id, players_data_local, num_teams_local, team_size_local, max_budget_local, ga_params_local, config_name_local, variation_name_ga):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  [{timestamp}] GA Run {run_id+1}/{NUM_RUNS} for {config_name_local} - {variation_name_ga} starting (Pop: {ga_params_local['population_size']}, Gen: {ga_params_local['generations']}, MutRate: {ga_params_local['mutation_rate']})...")
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
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  [{timestamp}] GA Run {run_id+1} for {config_name_local} - {variation_name_ga} finished in {actual_exec_time_ga_run:.2f}s with fitness {best_fitness_ga_run if best_solution_ga_run else 'N/A'}.")
    return best_solution_ga_run, best_fitness_ga_run, actual_exec_time_ga_run, fitness_history_ga_run


# %% [markdown]
# ## Main Execution Block (Adapted for Parameter Variation)

# %%
def main():
    script_total_start_time = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Multiprocessing Script for Parameter Variation ({NUM_RUNS} runs per variation) execution started.")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Player data loaded. Total players: {len(PLAYERS_DATA_GLOBAL)}")
    if PLAYERS_DATA_GLOBAL:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] First player data: {PLAYERS_DATA_GLOBAL[0]}")
    
    all_results_summary = []
    max_processes = 15 
    num_processes = min(NUM_RUNS, os.cpu_count() if os.cpu_count() else 1, max_processes)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Using {num_processes} processes for parallel execution.")

    # --- Hill Climbing Parameter Variations ---
    hc_max_iterations_variations = [500, 1000, 1500]
    for max_iter_hc in hc_max_iterations_variations:
        hc_variation_name = f"HC_MaxIter_{max_iter_hc}"
        hc_section_start_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] --- Starting Hill Climbing: {hc_variation_name} ({NUM_RUNS} runs in parallel) ---")
        
        hc_params_for_runs = [(i, PLAYERS_DATA_GLOBAL, NUM_TEAMS, TEAM_SIZE, MAX_BUDGET, max_iter_hc, hc_variation_name) for i in range(NUM_RUNS)]
        
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
                    if i < len(hc_all_solutions) and hc_all_solutions[i] is not None and i < len(hc_all_histories):
                        best_hc_fitness_overall = fit_val
                        best_hc_solution_overall = hc_all_solutions[i] 
                        best_hc_history_overall = hc_all_histories[i]
        
        hc_mean_fitness = np.nanmean([f for f in hc_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in hc_all_fitness_values) else float("nan")
        hc_std_fitness = np.nanstd([f for f in hc_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in hc_all_fitness_values) else float("nan")
        hc_mean_exec_time = np.nanmean([t for t in hc_all_exec_times if not np.isnan(t)]) if any(not np.isnan(t) for t in hc_all_exec_times) else float("nan")

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Hill Climbing ({hc_variation_name}, {NUM_RUNS} runs) processing finished.")
        print(f"  Mean Best Fitness: {hc_mean_fitness:.4f}")
        print(f"  Std Dev Best Fitness: {hc_std_fitness:.4f}")
        print(f"  Mean Execution Time per run: {hc_mean_exec_time:.2f}s")
        if best_hc_solution_overall:
            print(f"  Overall Best HC Fitness for {hc_variation_name}: {best_hc_fitness_overall:.4f}")
            all_results_summary.append({
                "Algorithm": hc_variation_name, 
                "Mean Fitness": hc_mean_fitness, 
                "Std Dev Fitness": hc_std_fitness, 
                "Mean Exec Time (s)": hc_mean_exec_time,
                "Overall Best Fitness": best_hc_fitness_overall,
                "Params": f"MaxIter={max_iter_hc}",
                "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
            })
            if best_hc_history_overall:
                plt.figure(figsize=(10, 6))
                plt.plot(best_hc_history_overall, marker="o", linestyle="-")
                plt.title(f"Hill Climbing Convergence ({hc_variation_name} - Best of {NUM_RUNS} MP Runs)")
                plt.xlabel("Improvement Step")
                plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
                plt.grid(True)
                plt.savefig(os.path.join(IMAGES_MP_DIR, f"hc_convergence_{hc_variation_name}_{NUM_RUNS}runs.png"))
                plt.close()
        else:
            all_results_summary.append({
                "Algorithm": hc_variation_name, 
                "Mean Fitness": hc_mean_fitness, 
                "Std Dev Fitness": hc_std_fitness, 
                "Mean Exec Time (s)": hc_mean_exec_time,
                "Overall Best Fitness": float("nan"),
                "Params": f"MaxIter={max_iter_hc}",
                "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
            })
        hc_section_end_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Hill Climbing section for {hc_variation_name} took {hc_section_end_time - hc_section_start_time:.2f} seconds.")

    # --- Genetic Algorithm Parameter Variations ---
    ga_base_configs = {
        "GA_Config_1": {
            "mutation_op": mutate_swap_constrained,
            "crossover_op": crossover_one_point_prefer_valid,
            "selection_op": selection_tournament_variable_k,
            "elitism_size": 2,
            "tournament_k": 3,
            "base_params": {"population_size": 50, "generations": 100, "mutation_rate": 0.1}
        },
        "GA_Config_4": {
            "mutation_op": mutate_targeted_player_exchange,
            "crossover_op": crossover_uniform_prefer_valid,
            "selection_op": selection_tournament_variable_k,
            "elitism_size": 3,
            "tournament_k": 5,
            "base_params": {"population_size": 50, "generations": 100, "mutation_rate": 0.1}
        }
    }

    ga_param_variations = {
        "mutation_rate": [0.05, 0.15, 0.25],
        "population_size": [30, 75],
        "generations": [75, 150]
    }

    for ga_config_name, ga_config_details in ga_base_configs.items():
        # Run with base parameters first
        current_ga_params = ga_config_details["base_params"].copy()
        current_ga_params.update({
            "mutation_op": ga_config_details["mutation_op"],
            "crossover_op": ga_config_details["crossover_op"],
            "selection_op": ga_config_details["selection_op"],
            "elitism_size": ga_config_details["elitism_size"],
            "tournament_k": ga_config_details.get("tournament_k"),
            "boltzmann_temp": ga_config_details.get("boltzmann_temp", 100) # Default if not specified
        })
        ga_variation_name = f"{ga_config_name}_BaseParams"
        ga_section_start_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] --- Starting GA: {ga_variation_name} ({NUM_RUNS} runs in parallel) ---")
        ga_params_for_runs = [(i, PLAYERS_DATA_GLOBAL, NUM_TEAMS, TEAM_SIZE, MAX_BUDGET, current_ga_params, ga_config_name, ga_variation_name) for i in range(NUM_RUNS)]
        with multiprocessing.Pool(processes=num_processes) as pool:
            ga_results_parallel = pool.starmap(run_genetic_algorithm_trial, ga_params_for_runs)
        
        # Process results for this base GA config (similar to HC processing)
        ga_all_solutions = [res[0] for res in ga_results_parallel if res is not None and res[0] is not None]
        ga_all_fitness_values = [res[1] for res in ga_results_parallel if res is not None]
        ga_all_exec_times = [res[2] for res in ga_results_parallel if res is not None]
        ga_all_histories = [res[3] for res in ga_results_parallel if res is not None]

        best_ga_fitness_overall = float("inf")
        best_ga_solution_overall = None
        best_ga_history_overall = []
        if ga_all_solutions:
            for i, fit_val in enumerate(ga_all_fitness_values):
                if not np.isnan(fit_val) and fit_val < best_ga_fitness_overall:
                    if i < len(ga_all_solutions) and ga_all_solutions[i] is not None and i < len(ga_all_histories):
                        best_ga_fitness_overall = fit_val
                        best_ga_solution_overall = ga_all_solutions[i]
                        best_ga_history_overall = ga_all_histories[i]
        
        ga_mean_fitness = np.nanmean([f for f in ga_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in ga_all_fitness_values) else float("nan")
        ga_std_fitness = np.nanstd([f for f in ga_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in ga_all_fitness_values) else float("nan")
        ga_mean_exec_time = np.nanmean([t for t in ga_all_exec_times if not np.isnan(t)]) if any(not np.isnan(t) for t in ga_all_exec_times) else float("nan")
        
        all_results_summary.append({
            "Algorithm": ga_variation_name,
            "Mean Fitness": ga_mean_fitness,
            "Std Dev Fitness": ga_std_fitness,
            "Mean Exec Time (s)": ga_mean_exec_time,
            "Overall Best Fitness": best_ga_fitness_overall if best_ga_solution_overall else float("nan"),
            "Params": f"Pop={current_ga_params['population_size']},Gen={current_ga_params['generations']},MutRate={current_ga_params['mutation_rate']}",
            "Mutation Op": ga_config_details["mutation_op"].__name__, 
            "Crossover Op": ga_config_details["crossover_op"].__name__,
            "Selection Op": ga_config_details["selection_op"].__name__
        })
        if best_ga_solution_overall and best_ga_history_overall:
            plt.figure(figsize=(10, 6))
            plt.plot(best_ga_history_overall, marker='.', linestyle='-')
            plt.title(f"GA Convergence ({ga_variation_name} - Best of {NUM_RUNS} MP Runs)")
            plt.xlabel("Generation")
            plt.ylabel("Best Fitness in Population")
            plt.grid(True)
            plt.savefig(os.path.join(IMAGES_MP_DIR, f"ga_convergence_{ga_variation_name}_{NUM_RUNS}runs.png"))
            plt.close()
        ga_section_end_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] GA section for {ga_variation_name} took {ga_section_end_time - ga_section_start_time:.2f} seconds.")

        # Loop through parameter variations for the current GA config
        for param_name, param_values in ga_param_variations.items():
            for p_val in param_values:
                # Create a copy of base params and update the varied one
                varied_ga_params = ga_config_details["base_params"].copy()
                varied_ga_params[param_name] = p_val
                varied_ga_params.update({
                    "mutation_op": ga_config_details["mutation_op"],
                    "crossover_op": ga_config_details["crossover_op"],
                    "selection_op": ga_config_details["selection_op"],
                    "elitism_size": ga_config_details["elitism_size"],
                    "tournament_k": ga_config_details.get("tournament_k"),
                    "boltzmann_temp": ga_config_details.get("boltzmann_temp", 100)
                })
                
                ga_variation_name_specific = f"{ga_config_name}_{param_name}_{p_val}"
                ga_section_start_time = time.time()
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] --- Starting GA: {ga_variation_name_specific} ({NUM_RUNS} runs in parallel) ---")
                ga_params_for_runs = [(i, PLAYERS_DATA_GLOBAL, NUM_TEAMS, TEAM_SIZE, MAX_BUDGET, varied_ga_params, ga_config_name, ga_variation_name_specific) for i in range(NUM_RUNS)]
                with multiprocessing.Pool(processes=num_processes) as pool:
                    ga_results_parallel = pool.starmap(run_genetic_algorithm_trial, ga_params_for_runs)
                
                # Process results for this specific GA variation
                ga_all_solutions = [res[0] for res in ga_results_parallel if res is not None and res[0] is not None]
                ga_all_fitness_values = [res[1] for res in ga_results_parallel if res is not None]
                ga_all_exec_times = [res[2] for res in ga_results_parallel if res is not None]
                ga_all_histories = [res[3] for res in ga_results_parallel if res is not None]

                best_ga_fitness_overall = float("inf")
                best_ga_solution_overall = None
                best_ga_history_overall = []
                if ga_all_solutions:
                    for i, fit_val in enumerate(ga_all_fitness_values):
                        if not np.isnan(fit_val) and fit_val < best_ga_fitness_overall:
                            if i < len(ga_all_solutions) and ga_all_solutions[i] is not None and i < len(ga_all_histories):
                                best_ga_fitness_overall = fit_val
                                best_ga_solution_overall = ga_all_solutions[i]
                                best_ga_history_overall = ga_all_histories[i]
                
                ga_mean_fitness = np.nanmean([f for f in ga_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in ga_all_fitness_values) else float("nan")
                ga_std_fitness = np.nanstd([f for f in ga_all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in ga_all_fitness_values) else float("nan")
                ga_mean_exec_time = np.nanmean([t for t in ga_all_exec_times if not np.isnan(t)]) if any(not np.isnan(t) for t in ga_all_exec_times) else float("nan")
                
                all_results_summary.append({
                    "Algorithm": ga_variation_name_specific,
                    "Mean Fitness": ga_mean_fitness,
                    "Std Dev Fitness": ga_std_fitness,
                    "Mean Exec Time (s)": ga_mean_exec_time,
                    "Overall Best Fitness": best_ga_fitness_overall if best_ga_solution_overall else float("nan"),
                    "Params": f"Pop={varied_ga_params['population_size']},Gen={varied_ga_params['generations']},MutRate={varied_ga_params['mutation_rate']}",
                    "Mutation Op": ga_config_details["mutation_op"].__name__, 
                    "Crossover Op": ga_config_details["crossover_op"].__name__,
                    "Selection Op": ga_config_details["selection_op"].__name__
                })
                if best_ga_solution_overall and best_ga_history_overall:
                    plt.figure(figsize=(10, 6))
                    plt.plot(best_ga_history_overall, marker='.', linestyle='-')
                    plt.title(f"GA Convergence ({ga_variation_name_specific} - Best of {NUM_RUNS} MP Runs)")
                    plt.xlabel("Generation")
                    plt.ylabel("Best Fitness in Population")
                    plt.grid(True)
                    plt.savefig(os.path.join(IMAGES_MP_DIR, f"ga_convergence_{ga_variation_name_specific}_{NUM_RUNS}runs.png"))
                    plt.close()
                ga_section_end_time = time.time()
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] GA section for {ga_variation_name_specific} took {ga_section_end_time - ga_section_start_time:.2f} seconds.")

    # --- Save Summary Results --- 
    summary_df = pd.DataFrame(all_results_summary)
    summary_csv_path = os.path.join(IMAGES_MP_DIR, f"all_algorithms_summary_final_param_var_{NUM_RUNS}runs.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Saved summary of all algorithm variations to {summary_csv_path}")

    script_total_end_time = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Total script execution time: {script_total_end_time - script_total_start_time:.2f} seconds.")

if __name__ == "__main__":
    multiprocessing.freeze_support() # Good practice for scripts that might be frozen into executables
    main()

