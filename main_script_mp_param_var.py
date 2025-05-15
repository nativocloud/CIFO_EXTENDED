# -*- coding: utf-8 -*-
# %%
# %% [markdown]
# # Introduction
#
# This work presents a constrained combinatorial optimization approach to the **Sports League Assignment Problem** using **Genetic Algorithms (GAs)**, **Hill Climbing (HC)**, and **Simulated Annealing (SA)**. The objective is to allocate a fixed pool of professional players into a set of 5 structurally valid teams in such a way that the **standard deviation of the teams\" average skill ratings** is minimized—promoting competitive balance across the league.
#
# This script is specifically for **parameter variation experiments** for selected Genetic Algorithm configurations.

# %% [markdown]
# ## Cell 1: Setup and Critical Data Loading

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os # Added for path joining
import multiprocessing # Added for multiprocessing
import copy # For deep copying dictionaries

from solution import LeagueSolution # LeagueHillClimbingSolution, LeagueSASolution are not used here
from evolution import genetic_algorithm # hill_climbing, simulated_annealing are not used here
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

# Define the directory for saving graphs and results for parameter variation
PARAM_VAR_RESULTS_DIR = "/home/ubuntu/CIFO_EXTENDED_Project/images_mp/param_var_results"
IMAGES_MP_DIR = PARAM_VAR_RESULTS_DIR # Adjusted for this script
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

# Define number of runs for stochastic algorithms (for parameter variation, 5 runs per variation)
NUM_RUNS = 5

# %% [markdown]
# ## Helper Functions for Multiprocessing Runs (Only GA needed)

# %%
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
        tournament_k=ga_params_local.get("tournament_k", 3), # Default k for tournament
        boltzmann_temp=ga_params_local.get("boltzmann_temp", 100), # Default temp for boltzmann
        num_teams=num_teams_local, 
        team_size=team_size_local,
        max_budget=max_budget_local,
        verbose=False # Keep verbose off for multiple runs
    )
    
    end_time_ga_run = time.time()
    actual_exec_time_ga_run = end_time_ga_run - start_time_ga_run 
    best_fitness_ga_run = best_solution_ga_run.fitness() if best_solution_ga_run else float("nan")
    
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"  [{timestamp}] GA Run {run_id+1} for {config_name_local} finished in {actual_exec_time_ga_run:.2f}s with fitness {best_fitness_ga_run if best_solution_ga_run else 'N/A'}.")
    return best_solution_ga_run, best_fitness_ga_run, actual_exec_time_ga_run, fitness_history_ga_run


# %% [markdown]
# ## Main Execution Block (Adapted for GA Parameter Variation)

# %%
def main():
    script_total_start_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] GA Parameter Variation Script for {NUM_RUNS} runs per variation execution started.")

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Player data loaded. Total players: {len(PLAYERS_DATA_GLOBAL)}")
    if PLAYERS_DATA_GLOBAL:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] First player data: {PLAYERS_DATA_GLOBAL[0]}")
    
    all_results_summary = []
    num_processes = min(NUM_RUNS, os.cpu_count() if os.cpu_count() else 1) # Parallelize the 5 runs for each variation
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Using {num_processes} processes for parallel execution of each variation's {NUM_RUNS} runs.")

    # --- Genetic Algorithms Parameter Variation --- 
    ga_section_start_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] --- Starting Genetic Algorithms Parameter Variation ({NUM_RUNS} runs per variation) ---")

    base_ga_configs = {
        "GA_Config_1_Base": {
            "population_size": 50,
            "generations": 100,
            "mutation_rate": 0.1,
            "elitism_size": 2,
            "mutation_op": mutate_swap_constrained,
            "crossover_op": crossover_one_point_prefer_valid,
            "selection_op": selection_tournament_variable_k,
            "tournament_k": 3,
            "base_name": "GA_Config_1"
        },
        "GA_Config_4_Base": {
            "population_size": 50,
            "generations": 100,
            "mutation_rate": 0.1,
            "elitism_size": 3,
            "mutation_op": mutate_targeted_player_exchange,
            "crossover_op": crossover_uniform_prefer_valid,
            "selection_op": selection_tournament_variable_k,
            "tournament_k": 5,
            "base_name": "GA_Config_4"
        }
    }

    parameter_variations = {
        "mutation_rate": [0.05, 0.15, 0.25],
        "population_size": [30, 75],
        "generations": [75, 150]
    }

    experiment_configs_to_run = []

    for base_config_key, base_params in base_ga_configs.items():
        # Run the base configuration itself (already done in 30 runs, but good for comparison if needed, or skip)
        # For this script, we focus on variations. The base 0.1 mut, 50 pop, 100 gen is the reference.

        # Variations for Mutation Rate
        for mr_var in parameter_variations["mutation_rate"]:
            if mr_var == base_params["mutation_rate"]: continue # Skip if it's the base value
            current_params = copy.deepcopy(base_params)
            current_params["mutation_rate"] = mr_var
            config_name = f"{base_params['base_name']}_MutRate_{mr_var}"
            experiment_configs_to_run.append((config_name, current_params))

        # Variations for Population Size
        for ps_var in parameter_variations["population_size"]:
            if ps_var == base_params["population_size"]: continue
            current_params = copy.deepcopy(base_params)
            current_params["population_size"] = ps_var
            config_name = f"{base_params['base_name']}_PopSize_{ps_var}"
            experiment_configs_to_run.append((config_name, current_params))
        
        # Variations for Generations
        for gen_var in parameter_variations["generations"]:
            if gen_var == base_params["generations"]: continue
            current_params = copy.deepcopy(base_params)
            current_params["generations"] = gen_var
            config_name = f"{base_params['base_name']}_NumGen_{gen_var}"
            experiment_configs_to_run.append((config_name, current_params))

    for config_name, ga_params_for_variation in experiment_configs_to_run:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{timestamp}] Processing GA Variation: {config_name}")
        print(f"  Parameters: Pop={ga_params_for_variation['population_size']}, Gen={ga_params_for_variation['generations']}, MutRate={ga_params_for_variation['mutation_rate']}")

        ga_run_params_list = [(i, PLAYERS_DATA_GLOBAL, NUM_TEAMS, TEAM_SIZE, MAX_BUDGET, ga_params_for_variation, config_name) for i in range(NUM_RUNS)]
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            ga_results_parallel_variation = pool.starmap(run_genetic_algorithm_trial, ga_run_params_list)

        ga_all_solutions_var = [res[0] for res in ga_results_parallel_variation if res is not None and res[0] is not None]
        ga_all_fitness_values_var = [res[1] for res in ga_results_parallel_variation if res is not None]
        ga_all_exec_times_var = [res[2] for res in ga_results_parallel_variation if res is not None]
        ga_all_histories_var = [res[3] for res in ga_results_parallel_variation if res is not None]

        best_ga_fitness_overall_var = float("inf")
        best_ga_solution_overall_var = None
        best_ga_history_overall_var = []

        if ga_all_solutions_var:
            for i, fit_val in enumerate(ga_all_fitness_values_var):
                if not np.isnan(fit_val) and fit_val < best_ga_fitness_overall_var:
                    if i < len(ga_all_solutions_var) and i < len(ga_all_histories_var):
                        best_ga_fitness_overall_var = fit_val
                        best_ga_solution_overall_var = ga_all_solutions_var[i]
                        best_ga_history_overall_var = ga_all_histories_var[i]
        
        ga_mean_fitness_var = np.nanmean([f for f in ga_all_fitness_values_var if not np.isnan(f)]) if any(not np.isnan(f) for f in ga_all_fitness_values_var) else float("nan")
        ga_std_fitness_var = np.nanstd([f for f in ga_all_fitness_values_var if not np.isnan(f)]) if any(not np.isnan(f) for f in ga_all_fitness_values_var) else float("nan")
        ga_mean_exec_time_var = np.nanmean([t for t in ga_all_exec_times_var if not np.isnan(t)]) if any(not np.isnan(t) for t in ga_all_exec_times_var) else float("nan")

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] GA Variation {config_name} ({NUM_RUNS} runs) processing finished.")
        print(f"  Mean Best Fitness: {ga_mean_fitness_var:.4f}")
        print(f"  Std Dev Best Fitness: {ga_std_fitness_var:.4f}")
        print(f"  Mean Execution Time per run: {ga_mean_exec_time_var:.2f}s")
        
        mutation_op_name = ga_params_for_variation["mutation_op"].__name__
        crossover_op_name = ga_params_for_variation["crossover_op"].__name__
        selection_op_name = ga_params_for_variation["selection_op"].__name__

        if best_ga_solution_overall_var:
            print(f"  Overall Best Fitness for {config_name}: {best_ga_fitness_overall_var:.4f}")
            all_results_summary.append({
                "Algorithm": config_name, 
                "Mean Fitness": ga_mean_fitness_var, 
                "Std Dev Fitness": ga_std_fitness_var, 
                "Mean Exec Time (s)": ga_mean_exec_time_var,
                "Overall Best Fitness": best_ga_fitness_overall_var,
                "Mutation Op": mutation_op_name, 
                "Crossover Op": crossover_op_name, 
                "Selection Op": selection_op_name,
                "PopSize": ga_params_for_variation['population_size'],
                "NumGen": ga_params_for_variation['generations'],
                "MutRate": ga_params_for_variation['mutation_rate']
            })
            if best_ga_history_overall_var:
                plt.figure(figsize=(10, 6))
                plt.plot(best_ga_history_overall_var)
                plt.title(f"GA Convergence - {config_name} (Best of {NUM_RUNS} MP Runs)")
                plt.xlabel("Generation")
                plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
                plt.grid(True)
                plt.savefig(os.path.join(IMAGES_MP_DIR, f"ga_convergence_{config_name}_mp_{NUM_RUNS}runs.png"))
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] Saved GA convergence plot to {IMAGES_MP_DIR}/ga_convergence_{config_name}_mp_{NUM_RUNS}runs.png")
                plt.close()
        else:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] GA Variation {config_name} did not find any valid solution across all runs that produced a best overall.")
            all_results_summary.append({
                "Algorithm": config_name, 
                "Mean Fitness": ga_mean_fitness_var, 
                "Std Dev Fitness": ga_std_fitness_var, 
                "Mean Exec Time (s)": ga_mean_exec_time_var,
                "Overall Best Fitness": float("nan"),
                "Mutation Op": mutation_op_name, 
                "Crossover Op": crossover_op_name, 
                "Selection Op": selection_op_name,
                "PopSize": ga_params_for_variation['population_size'],
                "NumGen": ga_params_for_variation['generations'],
                "MutRate": ga_params_for_variation['mutation_rate']
            })

    ga_section_end_time = time.time()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] Genetic Algorithms Parameter Variation section took {ga_section_end_time - ga_section_start_time:.2f} seconds (wall time for all variations).")

    # --- Overall Results Summary --- 
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] --- Overall Results Summary ({NUM_RUNS} MP Runs per Variation) ---")
    if all_results_summary:
        all_results_summary_df = pd.DataFrame(all_results_summary)
        print(all_results_summary_df.to_string())
        summary_filename = os.path.join(IMAGES_MP_DIR, f"all_ga_variations_summary_mp_{NUM_RUNS}runs.csv")
        all_results_summary_df.to_csv(summary_filename, index=False)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] Saved results summary to {summary_filename}")

        # Plotting comparative execution times
        plt.figure(figsize=(12, 8))
        all_results_summary_df_sorted_time = all_results_summary_df.sort_values(by="Mean Exec Time (s)")
        plt.bar(all_results_summary_df_sorted_time["Algorithm"], all_results_summary_df_sorted_time["Mean Exec Time (s)"], color="skyblue")
        plt.xlabel("GA Configuration Variation")
        plt.ylabel("Mean Execution Time (s)")
        plt.title(f"Comparative Mean Execution Times of GA Variations ({NUM_RUNS} MP Runs)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_MP_DIR, f"comparative_times_ga_variations_mp_{NUM_RUNS}runs.png"))
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] Saved comparative execution times plot.")
        plt.close()

        # Plotting comparative best fitness (lower is better)
        plt.figure(figsize=(12, 8))
        all_results_summary_df_sorted_fitness = all_results_summary_df.sort_values(by="Mean Fitness") # Sort by Mean Fitness for comparison
        plt.bar(all_results_summary_df_sorted_fitness["Algorithm"], all_results_summary_df_sorted_fitness["Mean Fitness"], color="lightcoral", yerr=all_results_summary_df_sorted_fitness["Std Dev Fitness"], capsize=5)
        plt.xlabel("GA Configuration Variation")
        plt.ylabel("Mean Best Fitness (Lower is Better)")
        plt.title(f"Comparative Mean Best Fitness of GA Variations ({NUM_RUNS} MP Runs)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_MP_DIR, f"comparative_fitness_ga_variations_mp_{NUM_RUNS}runs.png"))
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] Saved comparative best fitness plot.")
        plt.close()
    else:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] No results to summarize or plot.")

    script_total_end_time = time.time()
    total_script_time = script_total_end_time - script_total_start_time
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] GA Parameter Variation Script execution finished.")
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] Total script execution time: {total_script_time:.2f} seconds.")

if __name__ == "__main__":
    multiprocessing.freeze_support() # Good practice for multiprocessing, esp. on Windows
    main()


