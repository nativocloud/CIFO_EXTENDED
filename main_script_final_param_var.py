# -*- coding: utf-8 -*-
# %%
# %% [markdown]
# # Introduction
#
# This script is for detailed parameter variation analysis of Hill Climbing and two selected Genetic Algorithm configurations (GA_Config_1 and GA_Config_4).
# Each parameter variation will be run 30 times using multiprocessing for robust statistical analysis.

# %% [markdown]
# ## Cell 1: Setup and Critical Data Loading

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import multiprocessing

from solution import LeagueSolution, LeagueHillClimbingSolution # LeagueSASolution removed
from evolution import genetic_algorithm, hill_climbing # simulated_annealing removed
from operators import (
    mutate_swap_constrained,
    mutate_targeted_player_exchange,
    crossover_one_point_prefer_valid,
    crossover_uniform_prefer_valid,
    selection_tournament_variable_k
)

# Define the directory for saving graphs and results
FINAL_PARAM_VAR_RESULTS_DIR = "/home/ubuntu/CIFO_EXTENDED_Project/images_mp/final_param_var_results"
IMAGES_MP_DIR = FINAL_PARAM_VAR_RESULTS_DIR # Adjusted for this script
# Ensure the directory exists
if not os.path.exists(IMAGES_MP_DIR):
    os.makedirs(IMAGES_MP_DIR)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Created directory: {IMAGES_MP_DIR}")

# Load player data
players_df = pd.read_csv("players.csv", sep=";")
PLAYERS_DATA_GLOBAL = players_df.to_dict(orient="records")

# %% [markdown]
# ## Cell 2: Further Setup, Data Inspection, and Experiment Parameters

# %%
# Define problem parameters
NUM_TEAMS = 5
TEAM_SIZE = 7
MAX_BUDGET = 750

# Define number of runs for stochastic algorithms
NUM_RUNS = 30 # Each parameter variation will be run 30 times

# %% [markdown]
# ## Helper Functions for Multiprocessing Runs (Modified for specific algorithm calls)

# %%
def run_hill_climbing_trial(run_id, players_data_local, num_teams_local, team_size_local, max_budget_local, max_iterations_hc, config_name_hc):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"  [{timestamp}] HC Run {run_id+1}/{NUM_RUNS} for {config_name_hc} (MaxIter: {max_iterations_hc}) starting...")
    start_time_hc_run = time.time()
    
    initial_hc_solution_run = LeagueHillClimbingSolution(players_data_local, num_teams=num_teams_local, team_size=team_size_local, max_budget=max_budget_local)
    retry_attempts_hc = 0
    max_retry_hc = 5
    while not initial_hc_solution_run.is_valid() and retry_attempts_hc < max_retry_hc:
        initial_hc_solution_run = LeagueHillClimbingSolution(players_data_local, num_teams=num_teams_local, team_size=team_size_local, max_budget=max_budget_local)
        retry_attempts_hc += 1
    
    if not initial_hc_solution_run.is_valid():
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"  [{timestamp}] HC Run {run_id+1} for {config_name_hc} failed to create a valid initial solution. Skipping run.")
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
    print(f"  [{timestamp}] HC Run {run_id+1} for {config_name_hc} finished in {hc_exec_time_run:.2f}s with fitness {hc_fitness_val_run if hc_solution_obj_run else 'N/A'}.")
    return hc_solution_obj_run, hc_fitness_val_run if hc_solution_obj_run else float("nan"), hc_exec_time_run, hc_history_convergence_run

def run_genetic_algorithm_trial(run_id, players_data_local, num_teams_local, team_size_local, max_budget_local, ga_params_local, config_name_local):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"  [{timestamp}] GA Run {run_id+1}/{NUM_RUNS} for {config_name_local} starting (Pop: {ga_params_local['population_size']}, Gen: {ga_params_local['generations']}, MutRate: {ga_params_local['mutation_rate']})...")
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
# ## Main Execution Block for Parameter Variation

# %%
def main():
    script_total_start_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Parameter Variation Script for HC and 2 GAs ({NUM_RUNS} runs each) execution started.")

    all_results_summary = []
    num_processes = min(NUM_RUNS, os.cpu_count() if os.cpu_count() else 1) # Can be set lower if memory is an issue for many parallel GAs
    if num_processes > 4: # Cap at 4 to avoid excessive resource usage for GAs
        num_processes = 4
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Using up to {num_processes} processes for parallel execution of each configuration's {NUM_RUNS} runs.")

    # --- Hill Climbing Parameter Variations ---
    hc_max_iterations_variations = [500, 1000, 1500]
    for max_iter_hc in hc_max_iterations_variations:
        config_name = f"HC_Iter_{max_iter_hc}"
        section_start_time = time.time()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{timestamp}] --- Starting {config_name} ({NUM_RUNS} runs in parallel) ---")
        
        hc_params_for_runs = [(i, PLAYERS_DATA_GLOBAL, NUM_TEAMS, TEAM_SIZE, MAX_BUDGET, max_iter_hc, config_name) for i in range(NUM_RUNS)]
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results_parallel = pool.starmap(run_hill_climbing_trial, hc_params_for_runs)

        all_solutions = [res[0] for res in results_parallel if res is not None and res[0] is not None]
        all_fitness_values = [res[1] for res in results_parallel if res is not None]
        all_exec_times = [res[2] for res in results_parallel if res is not None]
        all_histories = [res[3] for res in results_parallel if res is not None]

        best_fitness_overall = float("inf")
        best_history_overall = []
        if all_solutions:
            valid_fitness_indices = [j for j, fit_val in enumerate(all_fitness_values) if not np.isnan(fit_val)]
            if valid_fitness_indices:
                best_run_index = min(valid_fitness_indices, key=lambda j: all_fitness_values[j])
                best_fitness_overall = all_fitness_values[best_run_index]
                best_history_overall = all_histories[best_run_index]
        
        mean_fitness = np.nanmean([f for f in all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in all_fitness_values) else float("nan")
        std_fitness = np.nanstd([f for f in all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in all_fitness_values) else float("nan")
        mean_exec_time = np.nanmean([t for t in all_exec_times if not np.isnan(t)]) if any(not np.isnan(t) for t in all_exec_times) else float("nan")

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {config_name} ({NUM_RUNS} runs) processing finished.")
        print(f"  Mean Best Fitness: {mean_fitness:.4f}")
        print(f"  Std Dev Best Fitness: {std_fitness:.4f}")
        print(f"  Mean Execution Time per run: {mean_exec_time:.2f}s")
        print(f"  Overall Best Fitness for {config_name}: {best_fitness_overall:.4f}")
        
        all_results_summary.append({
            "Algorithm Config": config_name, 
            "Mean Fitness": mean_fitness, 
            "Std Dev Fitness": std_fitness, 
            "Mean Exec Time (s)": mean_exec_time,
            "Overall Best Fitness": best_fitness_overall,
            "Params": f"MaxIter={max_iter_hc}"
        })
        if best_history_overall:
            plt.figure(figsize=(10, 6))
            plt.plot(best_history_overall, marker="o", linestyle="-")
            plt.title(f"{config_name} Convergence (Best of {NUM_RUNS} MP Runs)")
            plt.xlabel("Improvement Step")
            plt.ylabel("Fitness")
            plt.grid(True)
            plt.savefig(os.path.join(IMAGES_MP_DIR, f"{config_name.lower()}_convergence_mp_{NUM_RUNS}runs.png"))
            plt.close()
        section_end_time = time.time()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {config_name} section took {section_end_time - section_start_time:.2f} seconds.")

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

    param_variations = {
        "mutation_rate": [0.05, 0.15, 0.25],
        "population_size": [30, 75],
        "generations": [75, 150]
    }

    for ga_name, ga_config_details in ga_base_configs.items():
        # Run base configuration first
        current_params = ga_config_details["base_params"].copy()
        config_name_ga = f"{ga_name}_Base"
        
        ga_run_params = {
            **current_params,
            "mutation_op": ga_config_details["mutation_op"],
            "crossover_op": ga_config_details["crossover_op"],
            "selection_op": ga_config_details["selection_op"],
            "elitism_size": ga_config_details["elitism_size"],
            "tournament_k": ga_config_details["tournament_k"]
        }
        
        section_start_time = time.time()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{timestamp}] --- Starting {config_name_ga} ({NUM_RUNS} runs in parallel) ---")
        ga_params_for_runs = [(i, PLAYERS_DATA_GLOBAL, NUM_TEAMS, TEAM_SIZE, MAX_BUDGET, ga_run_params, config_name_ga) for i in range(NUM_RUNS)]
        with multiprocessing.Pool(processes=num_processes) as pool:
            results_parallel = pool.starmap(run_genetic_algorithm_trial, ga_params_for_runs)
        # ... (analysis and plotting logic as for HC, adapted for GA) ...
        all_solutions = [res[0] for res in results_parallel if res is not None and res[0] is not None]
        all_fitness_values = [res[1] for res in results_parallel if res is not None]
        all_exec_times = [res[2] for res in results_parallel if res is not None]
        all_histories = [res[3] for res in results_parallel if res is not None]

        best_fitness_overall = float("inf")
        best_history_overall = []
        if all_solutions:
            valid_fitness_indices = [j for j, fit_val in enumerate(all_fitness_values) if not np.isnan(fit_val)]
            if valid_fitness_indices:
                best_run_index = min(valid_fitness_indices, key=lambda j: all_fitness_values[j])
                best_fitness_overall = all_fitness_values[best_run_index]
                best_history_overall = all_histories[best_run_index]
        
        mean_fitness = np.nanmean([f for f in all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in all_fitness_values) else float("nan")
        std_fitness = np.nanstd([f for f in all_fitness_values if not np.isnan(f)]) if any(not np.isnan(f) for f in all_fitness_values) else float("nan")
        mean_exec_time = np.nanmean([t for t in all_exec_times if not np.isnan(t)]) if any(not np.isnan(t) for t in all_exec_times) else float("nan")

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {config_name_ga} ({NUM_RUNS} runs) processing finished.")
        print(f"  Mean Best Fitness: {mean_fitness:.4f}")
        print(f"  Std Dev Best Fitness: {std_fitness:.4f}")
        print(f"  Mean Execution Time per run: {mean_exec_time:.2f}s")
        print(f"  Overall Best Fitness for {config_name_ga}: {best_fitness_overall:.4f}")
        
        all_results_summary.append({
            "Algorithm Config": config_name_ga, 
            "Mean Fitness": mean_fitness, 
            "Std Dev Fitness": std_fitness, 
            "Mean Exec Time (s)": mean_exec_time,
            "Overall Best Fitness": best_fitness_overall,
            "Params": f"Pop={current_params['population_size']},Gen={current_params['generations']},MutRate={current_params['mutation_rate']}"
        })
        if best_history_overall:
            plt.figure(figsize=(10, 6))
            plt.plot(best_history_overall, marker="o", linestyle="-")
            plt.title(f"{config_name_ga} Convergence (Best of {NUM_RUNS} MP Runs)")
            plt.xlabel("Generation")
            plt.ylabel("Best Fitness in Population")
            plt.grid(True)
            plt.savefig(os.path.join(IMAGES_MP_DIR, f"{config_name_ga.lower().replace(' ', '_')}_convergence_mp_{NUM_RUNS}runs.png"))
            plt.close()
        section_end_time = time.time()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {config_name_ga} section took {section_end_time - section_start_time:.2f} seconds.")

        # Iterate through parameter variations
        for param_name, values_to_test in param_variations.items():
            for value in values_to_test:
                current_params_varied = ga_config_details["base_params"].copy()
                current_params_varied[param_name] = value
                config_name_ga_var = f"{ga_name}_{param_name}_{value}"

                ga_run_params_varied = {
                    **current_params_varied,
                    "mutation_op": ga_config_details["mutation_op"],
                    "crossover_op": ga_config_details["crossover_op"],
                    "selection_op": ga_config_details["selection_op"],
                    "elitism_size": ga_config_details["elitism_size"],
                    "tournament_k": ga_config_details["tournament_k"]
                }
                section_start_time = time.time()
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n[{timestamp}] --- Starting {config_name_ga_var} ({NUM_RUNS} runs in parallel) ---")
                ga_params_for_runs_var = [(i, PLAYERS_DATA_GLOBAL, NUM_TEAMS, TEAM_SIZE, MAX_BUDGET, ga_run_params_varied, config_name_ga_var) for i in range(NUM_RUNS)]
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results_parallel_var = pool.starmap(run_genetic_algorithm_trial, ga_params_for_runs_var)
                # ... (analysis and plotting logic as for HC, adapted for GA) ...
                all_solutions_var = [res[0] for res in results_parallel_var if res is not None and res[0] is not None]
                all_fitness_values_var = [res[1] for res in results_parallel_var if res is not None]
                all_exec_times_var = [res[2] for res in results_parallel_var if res is not None]
                all_histories_var = [res[3] for res in results_parallel_var if res is not None]

                best_fitness_overall_var = float("inf")
                best_history_overall_var = []
                if all_solutions_var:
                    valid_fitness_indices_var = [j for j, fit_val in enumerate(all_fitness_values_var) if not np.isnan(fit_val)]
                    if valid_fitness_indices_var:
                        best_run_index_var = min(valid_fitness_indices_var, key=lambda j: all_fitness_values_var[j])
                        best_fitness_overall_var = all_fitness_values_var[best_run_index_var]
                        best_history_overall_var = all_histories_var[best_run_index_var]
                
                mean_fitness_var = np.nanmean([f for f in all_fitness_values_var if not np.isnan(f)]) if any(not np.isnan(f) for f in all_fitness_values_var) else float("nan")
                std_fitness_var = np.nanstd([f for f in all_fitness_values_var if not np.isnan(f)]) if any(not np.isnan(f) for f in all_fitness_values_var) else float("nan")
                mean_exec_time_var = np.nanmean([t for t in all_exec_times_var if not np.isnan(t)]) if any(not np.isnan(t) for t in all_exec_times_var) else float("nan")

                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] {config_name_ga_var} ({NUM_RUNS} runs) processing finished.")
                print(f"  Mean Best Fitness: {mean_fitness_var:.4f}")
                print(f"  Std Dev Best Fitness: {std_fitness_var:.4f}")
                print(f"  Mean Execution Time per run: {mean_exec_time_var:.2f}s")
                print(f"  Overall Best Fitness for {config_name_ga_var}: {best_fitness_overall_var:.4f}")
                
                all_results_summary.append({
                    "Algorithm Config": config_name_ga_var, 
                    "Mean Fitness": mean_fitness_var, 
                    "Std Dev Fitness": std_fitness_var, 
                    "Mean Exec Time (s)": mean_exec_time_var,
                    "Overall Best Fitness": best_fitness_overall_var,
                    "Params": f"Pop={current_params_varied['population_size']},Gen={current_params_varied['generations']},MutRate={current_params_varied['mutation_rate']}"
                })
                if best_history_overall_var:
                    plt.figure(figsize=(10, 6))
                    plt.plot(best_history_overall_var, marker="o", linestyle="-")
                    plt.title(f"{config_name_ga_var} Convergence (Best of {NUM_RUNS} MP Runs)")
                    plt.xlabel("Generation")
                    plt.ylabel("Best Fitness in Population")
                    plt.grid(True)
                    plt.savefig(os.path.join(IMAGES_MP_DIR, f"{config_name_ga_var.lower().replace(' ', '_')}_convergence_mp_{NUM_RUNS}runs.png"))
                    plt.close()
                section_end_time = time.time()
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] {config_name_ga_var} section took {section_end_time - section_start_time:.2f} seconds.")

    # Save summary of all results
    summary_df = pd.DataFrame(all_results_summary)
    summary_filename = os.path.join(IMAGES_MP_DIR, f"all_algorithms_summary_final_param_var_{NUM_RUNS}runs.csv")
    summary_df.to_csv(summary_filename, index=False)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] Saved summary of all parameter variation results to {summary_filename}")

    script_total_end_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] Total script execution time: {script_total_end_time - script_total_start_time:.2f} seconds.")

if __name__ == "__main__":
    # This check is important for multiprocessing on some platforms (like Windows)
    # It ensures that the main() function is only called when the script is executed directly, not when imported by a child process.
    multiprocessing.freeze_support() # Good practice, though may not be strictly necessary on Linux for this script
    main()

# %%
