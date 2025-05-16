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

# %% [markdown]
# # Sports League Assignment Problem - Single Processor Analysis
#
# This notebook presents a modular implementation of metaheuristic algorithms for the Sports League Assignment Problem.
# Each algorithm (Hill Climbing, Simulated Annealing, Genetic Algorithm) can be executed independently or together.

# %% [markdown]
# ## Setup and Data Loading

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import scipy.stats as stats

# Add src directory to path for imports using relative path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from src.solution.solution import LeagueSolution, LeagueHillClimbingSolution, LeagueSASolution
from src.evolution.evolution import genetic_algorithm, hill_climbing, simulated_annealing 
from src.operators.operators import (
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

# Define the directories for saving graphs and results using relative path
RESULTS_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
PHASE_DIR = os.path.join(RESULTS_BASE_DIR, "phase_1_sp")
IMAGES_DIR = os.path.join(PHASE_DIR, "images")
DATA_DIR_RESULTS = os.path.join(PHASE_DIR, "data")

# Ensure directories exist
for dir_path in [RESULTS_BASE_DIR, PHASE_DIR, IMAGES_DIR, DATA_DIR_RESULTS]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Created directory: {dir_path}")

# Load player data using relative path
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
players_df = pd.read_csv(os.path.join(DATA_DIR, "players.csv"), sep=";")
players_data = players_df.to_dict(orient="records")

# %% [markdown]
# ## Problem Parameters and Global Variables

# %%
# Define problem parameters
NUM_TEAMS = 5
TEAM_SIZE = 7
MAX_BUDGET = 750

# Define number of runs for stochastic algorithms
NUM_RUNS = 1  # Default value, can be changed

# Dictionary to store all results for comparison
all_results = {
    "HC": {"configs": [], "fitness_values": [], "exec_times": [], "histories": []},
    "SA": {"configs": [], "fitness_values": [], "exec_times": [], "histories": []},
    "GA": {"configs": [], "fitness_values": [], "exec_times": [], "histories": []}
}

# %% [markdown]
# ## Utility Functions

# %%
def print_algorithm_results(algo_name, config_name, mean_fitness, std_fitness, mean_exec_time, best_fitness):
    """Print formatted results for an algorithm run."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {algo_name} {config_name} ({NUM_RUNS} runs) processing finished.")
    print(f"  Mean Best Fitness: {mean_fitness:.4f}")
    print(f"  Std Dev Best Fitness: {std_fitness:.4f}")
    print(f"  Mean Execution Time per run: {mean_exec_time:.2f}s")
    print(f"  Overall Best Fitness: {best_fitness:.4f}")

def save_convergence_plot(history, algo_name, config_name, xlabel="Iteration"):
    """Save convergence plot for an algorithm run."""
    plt.figure(figsize=(10, 6))
    plt.plot(history, marker="o", linestyle="-")
    plt.title(f"{algo_name} Convergence ({config_name}) - Best of {NUM_RUNS} Runs")
    plt.xlabel(xlabel)
    plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
    plt.grid(True)
    
    filename = f"{algo_name.lower().replace(' ', '_')}_{config_name.lower().replace(' ', '_')}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved {algo_name} convergence plot to {filepath}")
    plt.close()
    
    return filepath

def compare_convergence_plots(histories_dict, algo_name, xlabel="Iteration"):
    """Create and save a comparative convergence plot for multiple configurations of the same algorithm."""
    plt.figure(figsize=(12, 8))
    
    for config_name, history in histories_dict.items():
        plt.plot(history, label=config_name)
    
    plt.title(f"{algo_name} Convergence Comparison - Best of {NUM_RUNS} Runs")
    plt.xlabel(xlabel)
    plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
    plt.grid(True)
    plt.legend()
    
    filename = f"{algo_name.lower().replace(' ', '_')}_comparison.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved {algo_name} comparison plot to {filepath}")
    plt.close()
    
    return filepath

def compare_all_algorithms():
    """Create and save a comparative plot of all algorithms."""
    plt.figure(figsize=(14, 10))
    
    # Plot HC results
    for i, config in enumerate(all_results["HC"]["configs"]):
        plt.plot(all_results["HC"]["histories"][i], 
                 label=f"HC - {config}", 
                 linestyle="-")
    
    # Plot SA results
    for i, config in enumerate(all_results["SA"]["configs"]):
        plt.plot(all_results["SA"]["histories"][i][:100:10],  # Sample points to avoid overcrowding
                 label=f"SA - {config}", 
                 linestyle="--")
    
    # Plot GA results
    for i, config in enumerate(all_results["GA"]["configs"]):
        plt.plot(all_results["GA"]["histories"][i], 
                 label=f"GA - {config}", 
                 linestyle="-.")
    
    plt.title(f"All Algorithms Convergence Comparison - Best of {NUM_RUNS} Runs")
    plt.xlabel("Iteration (Normalized)")
    plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    filename = "all_algorithms_comparison.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved all algorithms comparison plot to {filepath}")
    plt.close()
    
    return filepath

def perform_statistical_tests():
    """Perform statistical tests to compare algorithm performance."""
    # Prepare data for statistical tests
    algorithm_names = []
    fitness_values = []
    
    # Collect fitness values for each algorithm configuration
    for algo in ["HC", "SA", "GA"]:
        for i, config in enumerate(all_results[algo]["configs"]):
            algorithm_names.append(f"{algo} - {config}")
            fitness_values.append(all_results[algo]["fitness_values"][i])
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame({
        'Algorithm': algorithm_names,
        'Fitness': fitness_values
    })
    
    # Save results to CSV
    results_filepath = os.path.join(DATA_DIR_RESULTS, "statistical_comparison.csv")
    results_df.to_csv(results_filepath, index=False)
    
    # Perform ANOVA if we have enough data points
    if len(algorithm_names) >= 3:
        try:
            # One-way ANOVA
            groups = [values for values in fitness_values if len(values) > 0]
            if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                f_stat, p_value = stats.f_oneway(*groups)
                
                print("\n--- Statistical Analysis ---")
                print(f"One-way ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
                
                if p_value < 0.05:
                    print("There is a statistically significant difference between algorithm performances.")
                else:
                    print("No statistically significant difference detected between algorithm performances.")
                
                # If we have enough data, perform post-hoc tests
                if len(groups) >= 3:
                    # Flatten data for pairwise tests
                    all_data = []
                    group_labels = []
                    for i, group in enumerate(groups):
                        all_data.extend(group)
                        group_labels.extend([algorithm_names[i]] * len(group))
                    
                    # Tukey HSD test
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd
                    tukey_results = pairwise_tukeyhsd(all_data, group_labels, alpha=0.05)
                    print("\nTukey HSD Post-hoc Test:")
                    print(tukey_results)
        except Exception as e:
            print(f"Statistical test error: {e}")
            print("Not enough data for statistical analysis. Run with NUM_RUNS > 1 for statistical tests.")
    else:
        print("\nNot enough algorithm configurations for statistical comparison.")
    
    return results_filepath

# %% [markdown]
# ## Hill Climbing Implementation

# %%
def run_hill_climbing(num_runs=NUM_RUNS, verbose=False):
    """Run Hill Climbing algorithm with specified number of runs."""
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting Hill Climbing Algorithm ({num_runs} runs) ---")
    
    hc_section_start_time = time.time()
    
    # Configuration name
    config_name = "Standard"
    
    hc_all_fitness_values = []
    hc_all_exec_times = []
    best_hc_solution_overall = None
    best_hc_fitness_overall = float("inf")
    best_hc_history_overall = []
    
    for i in range(num_runs):
        print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] HC Run {i+1}/{num_runs}...")
        start_time_hc_run = time.time()
        
        initial_hc_solution_run = LeagueHillClimbingSolution(players_data, num_teams=NUM_TEAMS, team_size=TEAM_SIZE, max_budget=MAX_BUDGET)
        retry_attempts_hc = 0
        max_retry_hc = 5
        while not initial_hc_solution_run.is_valid() and retry_attempts_hc < max_retry_hc:
            if verbose:
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
            verbose=verbose
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

    # Print results
    if best_hc_solution_overall:
        print_algorithm_results("Hill Climbing", config_name, hc_mean_fitness, hc_std_fitness, 
                               hc_mean_exec_time, best_hc_fitness_overall)
        
        # Save convergence plot
        save_convergence_plot(best_hc_history_overall, "Hill Climbing", config_name, "Improvement Step")
        
        # Store results for comparison
        all_results["HC"]["configs"].append(config_name)
        all_results["HC"]["fitness_values"].append(hc_all_fitness_values)
        all_results["HC"]["exec_times"].append(hc_all_exec_times)
        all_results["HC"]["histories"].append(best_hc_history_overall)
        
        # Add to summary for final comparison
        result_summary = {
            "Algorithm": f"Hill Climbing - {config_name}", 
            "Mean Fitness": hc_mean_fitness, 
            "Std Dev Fitness": hc_std_fitness, 
            "Mean Exec Time (s)": hc_mean_exec_time,
            "Overall Best Fitness": best_hc_fitness_overall,
            "Mutation Op": "N/A", 
            "Crossover Op": "N/A", 
            "Selection Op": "N/A"
        }
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hill Climbing did not find any valid solution across all runs.")
        result_summary = {
            "Algorithm": f"Hill Climbing - {config_name}", 
            "Mean Fitness": hc_mean_fitness, 
            "Std Dev Fitness": hc_std_fitness, 
            "Mean Exec Time (s)": hc_mean_exec_time,
            "Overall Best Fitness": float('nan'),
            "Mutation Op": "N/A", 
            "Crossover Op": "N/A", 
            "Selection Op": "N/A"
        }
    
    hc_section_end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hill Climbing section took {hc_section_end_time - hc_section_start_time:.2f} seconds.")
    
    return result_summary, best_hc_solution_overall, best_hc_history_overall

# %% [markdown]
# ## Hill Climbing Execution and Analysis
# 
# Run this cell to execute the Hill Climbing algorithm and analyze its results.

# %%
# Execute Hill Climbing
hc_results, hc_best_solution, hc_best_history = run_hill_climbing(num_runs=NUM_RUNS)

# %% [markdown]
# ## Simulated Annealing Implementation

# %%
def run_simulated_annealing(num_runs=NUM_RUNS, initial_temp=1000, final_temp=1, alpha=0.99, 
                           iterations_per_temp=100, config_name="Standard", verbose=False):
    """Run Simulated Annealing algorithm with specified parameters."""
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting Simulated Annealing Algorithm ({num_runs} runs) ---")
    print(f"  Configuration: {config_name} (Initial Temp: {initial_temp}, Final Temp: {final_temp}, Alpha: {alpha}, Iterations/Temp: {iterations_per_temp})")
    
    sa_section_start_time = time.time()
    
    sa_all_fitness_values = []
    sa_all_exec_times = []
    best_sa_solution_overall = None
    best_sa_fitness_overall = float("inf")
    best_sa_history_overall = []
    
    for i in range(num_runs):
        print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] SA Run {i+1}/{num_runs}...")
        start_time_sa_run = time.time()
        
        initial_sa_solution_run = LeagueSASolution(players_data, num_teams=NUM_TEAMS, team_size=TEAM_SIZE, max_budget=MAX_BUDGET)
        retry_attempts_sa = 0
        max_retry_sa = 5
        while not initial_sa_solution_run.is_valid() and retry_attempts_sa < max_retry_sa:
            if verbose:
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
            initial_temp=initial_temp, 
            final_temp=final_temp, 
            alpha=alpha, 
            iterations_per_temp=iterations_per_temp,
            verbose=verbose
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

    # Print results
    if best_sa_solution_overall:
        print_algorithm_results("Simulated Annealing", config_name, sa_mean_fitness, sa_std_fitness, 
                               sa_mean_exec_time, best_sa_fitness_overall)
        
        # Save convergence plot
        save_convergence_plot(best_sa_history_overall, "Simulated Annealing", config_name, "Iteration")
        
        # Store results for comparison
        all_results["SA"]["configs"].append(config_name)
        all_results["SA"]["fitness_values"].append(sa_all_fitness_values)
        all_results["SA"]["exec_times"].append(sa_all_exec_times)
        all_results["SA"]["histories"].append(best_sa_history_overall)
        
        # Add to summary for final comparison
        result_summary = {
            "Algorithm": f"Simulated Annealing - {config_name}", 
            "Mean Fitness": sa_mean_fitness, 
            "Std Dev Fitness": sa_std_fitness, 
            "Mean Exec Time (s)": sa_mean_exec_time,
            "Overall Best Fitness": best_sa_fitness_overall,
            "Initial Temp": initial_temp,
            "Final Temp": final_temp,
            "Alpha": alpha,
            "Iterations/Temp": iterations_per_temp
        }
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Simulated Annealing did not find any valid solution across all runs.")
        result_summary = {
            "Algorithm": f"Simulated Annealing - {config_name}", 
            "Mean Fitness": sa_mean_fitness, 
            "Std Dev Fitness": sa_std_fitness, 
            "Mean Exec Time (s)": sa_mean_exec_time,
            "Overall Best Fitness": float('nan'),
            "Initial Temp": initial_temp,
            "Final Temp": final_temp,
            "Alpha": alpha,
            "Iterations/Temp": iterations_per_temp
        }
    
    sa_section_end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Simulated Annealing section took {sa_section_end_time - sa_section_start_time:.2f} seconds.")
    
    return result_summary, best_sa_solution_overall, best_sa_history_overall

# %% [markdown]
# ## Simulated Annealing Execution and Analysis
# 
# Run this cell to execute the Simulated Annealing algorithm with different configurations and analyze results.

# %%
# Execute Simulated Annealing with standard configuration
sa_results_standard, sa_best_solution_standard, sa_best_history_standard = run_simulated_annealing(
    num_runs=NUM_RUNS,
    initial_temp=1000,
    final_temp=1,
    alpha=0.99,
    iterations_per_temp=100,
    config_name="Standard"
)

# Optional: Run with alternative configurations
# Uncomment to run additional configurations

# # Fast cooling configuration
# sa_results_fast, sa_best_solution_fast, sa_best_history_fast = run_simulated_annealing(
#     num_runs=NUM_RUNS,
#     initial_temp=1000,
#     final_temp=1,
#     alpha=0.95,  # Faster cooling
#     iterations_per_temp=100,
#     config_name="Fast Cooling"
# )

# # More iterations configuration
# sa_results_more_iter, sa_best_solution_more_iter, sa_best_history_more_iter = run_simulated_annealing(
#     num_runs=NUM_RUNS,
#     initial_temp=1000,
#     final_temp=1,
#     alpha=0.99,
#     iterations_per_temp=200,  # More iterations per temperature
#     config_name="More Iterations"
# )

# %% [markdown]
# ## Compare Simulated Annealing Configurations
# 
# Run this cell to compare different SA configurations if multiple were executed.

# %%
# Compare SA configurations if we have more than one
if len(all_results["SA"]["configs"]) > 1:
    # Create dictionary of histories for comparison
    sa_histories_dict = {config: all_results["SA"]["histories"][i] 
                         for i, config in enumerate(all_results["SA"]["configs"])}
    
    # Generate comparative plot
    sa_comparison_plot = compare_convergence_plots(sa_histories_dict, "Simulated Annealing")
    
    # Statistical comparison if we have enough runs
    if NUM_RUNS > 1:
        print("\n--- Statistical Comparison of SA Configurations ---")
        sa_configs = all_results["SA"]["configs"]
        sa_fitness_values = all_results["SA"]["fitness_values"]
        
        # ANOVA if we have at least 3 configurations
        if len(sa_configs) >= 3:
            try:
                f_stat, p_value = stats.f_oneway(*sa_fitness_values)
                print(f"One-way ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
                
                if p_value < 0.05:
                    print("There is a statistically significant difference between SA configurations.")
                else:
                    print("No statistically significant difference detected between SA configurations.")
            except:
                print("Not enough data for statistical analysis. Run with NUM_RUNS > 1 for statistical tests.")
        # T-test if we have exactly 2 configurations
        elif len(sa_configs) == 2:
            try:
                t_stat, p_value = stats.ttest_ind(sa_fitness_values[0], sa_fitness_values[1], equal_var=False)
                print(f"Independent t-test: t={t_stat:.4f}, p={p_value:.4f}")
                
                if p_value < 0.05:
                    print(f"There is a statistically significant difference between {sa_configs[0]} and {sa_configs[1]}.")
                else:
                    print(f"No statistically significant difference detected between {sa_configs[0]} and {sa_configs[1]}.")
            except:
                print("Not enough data for statistical analysis. Run with NUM_RUNS > 1 for statistical tests.")
else:
    print("Only one SA configuration was run. No comparison needed.")

# %% [markdown]
# ## Genetic Algorithm Implementation

# %%
def run_genetic_algorithm(num_runs=NUM_RUNS, population_size=50, generations=30, mutation_rate=0.2, elite_size=5,
                         mutation_operator_func=mutate_swap_constrained, crossover_operator_func=crossover_one_point_prefer_valid,
                         selection_operator_func=selection_tournament_variable_k, tournament_k=3, boltzmann_temp=100,
                         config_name="Default", verbose=False):
    """Run Genetic Algorithm with specified parameters."""
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Running GA Configuration: {config_name} ({num_runs} runs) ---")
    
    ga_section_start_time = time.time()
    
    ga_all_fitness_values = []
    ga_all_exec_times = []
    best_ga_solution_config_overall = None
    best_ga_fitness_config_overall = float("inf")
    best_ga_history_config_overall = []
    
    for i in range(num_runs):
        print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] GA Run {i+1}/{num_runs} for {config_name}...")
        start_time_ga_run = time.time()
        
        ga_solution_obj_run, ga_history_convergence_run = genetic_algorithm(
            players_data=players_data,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
            mutation_operator_func=mutation_operator_func,
            crossover_operator_func=crossover_operator_func,
            selection_operator_func=selection_operator_func,
            tournament_k=tournament_k,
            boltzmann_temp=boltzmann_temp,
            num_teams=NUM_TEAMS,
            team_size=TEAM_SIZE,
            max_budget=MAX_BUDGET,
            verbose=verbose
        )
        end_time_ga_run = time.time()
        ga_exec_time_run = end_time_ga_run - start_time_ga_run
        
        if ga_solution_obj_run:
            ga_fitness_val_run = ga_solution_obj_run.fitness()
            ga_all_fitness_values.append(ga_fitness_val_run)
            ga_all_exec_times.append(ga_exec_time_run)
            
            if ga_fitness_val_run < best_ga_fitness_config_overall:
                best_ga_fitness_config_overall = ga_fitness_val_run
                best_ga_solution_config_overall = ga_solution_obj_run
                best_ga_history_config_overall = ga_history_convergence_run
        else:
            print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] GA Run {i+1} for {config_name} did not find a valid solution.")
            ga_all_fitness_values.append(float('nan'))
            ga_all_exec_times.append(float('nan'))
    
    ga_mean_fitness = np.nanmean(ga_all_fitness_values) if ga_all_fitness_values else float("nan")
    ga_std_fitness = np.nanstd(ga_all_fitness_values) if ga_all_fitness_values else float("nan")
    ga_mean_exec_time = np.nanmean(ga_all_exec_times) if ga_all_exec_times else float("nan")
    
    # Print results
    if best_ga_solution_config_overall:
        print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] GA Configuration {config_name} ({num_runs} runs) processing finished.")
        print(f"    Mean Best Fitness: {ga_mean_fitness:.4f}")
        print(f"    Std Dev Best Fitness: {ga_std_fitness:.4f}")
        print(f"    Mean Execution Time per run: {ga_mean_exec_time:.2f}s")
        print(f"    Overall Best GA Fitness for {config_name}: {best_ga_fitness_config_overall:.4f}")
        
        # Save convergence plot
        save_convergence_plot(best_ga_history_config_overall, "Genetic Algorithm", config_name, "Generation")
        
        # Store results for comparison
        all_results["GA"]["configs"].append(config_name)
        all_results["GA"]["fitness_values"].append(ga_all_fitness_values)
        all_results["GA"]["exec_times"].append(ga_all_exec_times)
        all_results["GA"]["histories"].append(best_ga_history_config_overall)
        
        # Get operator names for summary
        mutation_op_name = mutation_operator_func.__name__
        crossover_op_name = crossover_operator_func.__name__
        selection_op_name = selection_operator_func.__name__
        
        # Add to summary for final comparison
        result_summary = {
            "Algorithm": f"Genetic Algorithm - {config_name}", 
            "Mean Fitness": ga_mean_fitness, 
            "Std Dev Fitness": ga_std_fitness, 
            "Mean Exec Time (s)": ga_mean_exec_time,
            "Overall Best Fitness": best_ga_fitness_config_overall,
            "Mutation Op": mutation_op_name,
            "Crossover Op": crossover_op_name,
            "Selection Op": selection_op_name,
            "Population Size": population_size,
            "Generations": generations,
            "Mutation Rate": mutation_rate,
            "Elite Size": elite_size
        }
    else:
        print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] GA Configuration {config_name} did not find any valid solution across all runs.")
        result_summary = {
            "Algorithm": f"Genetic Algorithm - {config_name}", 
            "Mean Fitness": ga_mean_fitness, 
            "Std Dev Fitness": ga_std_fitness, 
            "Mean Exec Time (s)": ga_mean_exec_time,
            "Overall Best Fitness": float('nan'),
            "Mutation Op": mutation_operator_func.__name__,
            "Crossover Op": crossover_operator_func.__name__,
            "Selection Op": selection_operator_func.__name__
        }
    
    ga_section_end_time = time.time()
    print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] GA Configuration {config_name} took {ga_section_end_time - ga_section_start_time:.2f} seconds.")
    
    return result_summary, best_ga_solution_config_overall, best_ga_history_config_overall

# %% [markdown]
# ## Genetic Algorithm Execution and Analysis
# 
# Run this cell to execute the Genetic Algorithm with different configurations and analyze results.

# %%
# Define GA configurations
ga_configs = [
    {
        "name": "GA_Config_1_SwapConst1PtPreferVTournVarK",
        "population_size": 50,
        "generations": 30,
        "mutation_rate": 0.2,
        "elite_size": 5,
        "mutation_operator_func": mutate_swap_constrained,
        "crossover_operator_func": crossover_one_point_prefer_valid,
        "selection_operator_func": selection_tournament_variable_k,
        "tournament_k": 3,
        "boltzmann_temp": 100
    },
    {
        "name": "GA_Config_2_TargetExchUnifPreferVRank",
        "population_size": 50,
        "generations": 30,
        "mutation_rate": 0.2,
        "elite_size": 5,
        "mutation_operator_func": mutate_targeted_player_exchange,
        "crossover_operator_func": crossover_uniform_prefer_valid,
        "selection_operator_func": selection_ranking,
        "tournament_k": 3,
        "boltzmann_temp": 100
    }
]

# Execute GA configurations
ga_results = []
ga_best_solutions = []
ga_best_histories = []

print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting Genetic Algorithm Configurations ({NUM_RUNS} runs each) ---")

ga_all_start_time = time.time()

for config in ga_configs:
    result, solution, history = run_genetic_algorithm(
        num_runs=NUM_RUNS,
        population_size=config["population_size"],
        generations=config["generations"],
        mutation_rate=config["mutation_rate"],
        elite_size=config["elite_size"],
        mutation_operator_func=config["mutation_operator_func"],
        crossover_operator_func=config["crossover_operator_func"],
        selection_operator_func=config["selection_operator_func"],
        tournament_k=config["tournament_k"],
        boltzmann_temp=config["boltzmann_temp"],
        config_name=config["name"]
    )
    ga_results.append(result)
    ga_best_solutions.append(solution)
    ga_best_histories.append(history)

ga_all_end_time = time.time()
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All Genetic Algorithm configurations took {ga_all_end_time - ga_all_start_time:.2f} seconds.")

# %% [markdown]
# ## Compare Genetic Algorithm Configurations
# 
# Run this cell to compare different GA configurations.

# %%
# Compare GA configurations if we have more than one
if len(all_results["GA"]["configs"]) > 1:
    # Create dictionary of histories for comparison
    ga_histories_dict = {config: all_results["GA"]["histories"][i] 
                         for i, config in enumerate(all_results["GA"]["configs"])}
    
    # Generate comparative plot
    ga_comparison_plot = compare_convergence_plots(ga_histories_dict, "Genetic Algorithm", "Generation")
    
    # Statistical comparison if we have enough runs
    if NUM_RUNS > 1:
        print("\n--- Statistical Comparison of GA Configurations ---")
        ga_configs = all_results["GA"]["configs"]
        ga_fitness_values = all_results["GA"]["fitness_values"]
        
        # ANOVA if we have at least 3 configurations
        if len(ga_configs) >= 3:
            try:
                f_stat, p_value = stats.f_oneway(*ga_fitness_values)
                print(f"One-way ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
                
                if p_value < 0.05:
                    print("There is a statistically significant difference between GA configurations.")
                else:
                    print("No statistically significant difference detected between GA configurations.")
            except:
                print("Not enough data for statistical analysis. Run with NUM_RUNS > 1 for statistical tests.")
        # T-test if we have exactly 2 configurations
        elif len(ga_configs) == 2:
            try:
                t_stat, p_value = stats.ttest_ind(ga_fitness_values[0], ga_fitness_values[1], equal_var=False)
                print(f"Independent t-test: t={t_stat:.4f}, p={p_value:.4f}")
                
                if p_value < 0.05:
                    print(f"There is a statistically significant difference between {ga_configs[0]} and {ga_configs[1]}.")
                else:
                    print(f"No statistically significant difference detected between {ga_configs[0]} and {ga_configs[1]}.")
            except:
                print("Not enough data for statistical analysis. Run with NUM_RUNS > 1 for statistical tests.")
else:
    print("Only one GA configuration was run. No comparison needed.")

# %% [markdown]
# ## Cross-Algorithm Comparison
# 
# Run this cell to compare all algorithms (HC, SA, GA) and their configurations.

# %%
# Collect all results for final comparison
all_results_summary = []

# Add HC results
for i, config in enumerate(all_results["HC"]["configs"]):
    all_results_summary.append({
        "Algorithm": f"Hill Climbing - {config}",
        "Mean Fitness": np.nanmean(all_results["HC"]["fitness_values"][i]) if all_results["HC"]["fitness_values"][i] else float("nan"),
        "Std Dev Fitness": np.nanstd(all_results["HC"]["fitness_values"][i]) if all_results["HC"]["fitness_values"][i] else float("nan"),
        "Mean Exec Time (s)": np.nanmean(all_results["HC"]["exec_times"][i]) if all_results["HC"]["exec_times"][i] else float("nan"),
        "Overall Best Fitness": min([x for x in all_results["HC"]["fitness_values"][i] if not np.isnan(x)], default=float("nan")),
        "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
    })

# Add SA results
for i, config in enumerate(all_results["SA"]["configs"]):
    all_results_summary.append({
        "Algorithm": f"Simulated Annealing - {config}",
        "Mean Fitness": np.nanmean(all_results["SA"]["fitness_values"][i]) if all_results["SA"]["fitness_values"][i] else float("nan"),
        "Std Dev Fitness": np.nanstd(all_results["SA"]["fitness_values"][i]) if all_results["SA"]["fitness_values"][i] else float("nan"),
        "Mean Exec Time (s)": np.nanmean(all_results["SA"]["exec_times"][i]) if all_results["SA"]["exec_times"][i] else float("nan"),
        "Overall Best Fitness": min([x for x in all_results["SA"]["fitness_values"][i] if not np.isnan(x)], default=float("nan")),
        "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
    })

# Add GA results
for i, config in enumerate(all_results["GA"]["configs"]):
    all_results_summary.append({
        "Algorithm": f"Genetic Algorithm - {config}",
        "Mean Fitness": np.nanmean(all_results["GA"]["fitness_values"][i]) if all_results["GA"]["fitness_values"][i] else float("nan"),
        "Std Dev Fitness": np.nanstd(all_results["GA"]["fitness_values"][i]) if all_results["GA"]["fitness_values"][i] else float("nan"),
        "Mean Exec Time (s)": np.nanmean(all_results["GA"]["exec_times"][i]) if all_results["GA"]["exec_times"][i] else float("nan"),
        "Overall Best Fitness": min([x for x in all_results["GA"]["fitness_values"][i] if not np.isnan(x)], default=float("nan")),
        "Mutation Op": ga_results[i]["Mutation Op"] if i < len(ga_results) else "N/A",
        "Crossover Op": ga_results[i]["Crossover Op"] if i < len(ga_results) else "N/A",
        "Selection Op": ga_results[i]["Selection Op"] if i < len(ga_results) else "N/A"
    })

# Convert to DataFrame and save
summary_df = pd.DataFrame(all_results_summary)
summary_filename = os.path.join(DATA_DIR_RESULTS, "all_algorithms_summary.csv")
summary_df.to_csv(summary_filename, index=False, sep=";")
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Overall summary saved to {summary_filename}")

# Generate comparative plot for all algorithms
if all_results["HC"]["histories"] and all_results["SA"]["histories"] and all_results["GA"]["histories"]:
    all_algorithms_plot = compare_all_algorithms()
    
    # Perform statistical tests if we have enough runs
    if NUM_RUNS > 1:
        statistical_results = perform_statistical_tests()
        print(f"Statistical comparison saved to {statistical_results}")

# Display summary table
print("\n--- Algorithm Performance Summary ---")
display_cols = ["Algorithm", "Overall Best Fitness", "Mean Fitness", "Mean Exec Time (s)"]
print(summary_df[display_cols].sort_values("Overall Best Fitness").to_string(index=False))

# %% [markdown]
# ## Main Execution Function

# %%
def main(run_hc=True, run_sa=True, run_ga=True, num_runs=1):
    """Main execution function to run all algorithms."""
    global NUM_RUNS
    NUM_RUNS = num_runs
    
    script_total_start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Single-Processor Script execution started.")

    data_load_start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading player data...")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Player data loaded successfully. Total players: {len(players_data)}")
    if players_data:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] First player data: {players_data[0]}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All algorithms (HC, SA, GA) will be run {NUM_RUNS} times each.")
    data_load_end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Data loading and setup took {data_load_end_time - data_load_start_time:.2f} seconds.")
    
    # Run Hill Climbing if requested
    if run_hc:
        hc_results, hc_best_solution, hc_best_history = run_hill_climbing(num_runs=NUM_RUNS)
    
    # Run Simulated Annealing if requested
    if run_sa:
        sa_results_standard, sa_best_solution_standard, sa_best_history_standard = run_simulated_annealing(
            num_runs=NUM_RUNS,
            config_name="Standard"
        )
        
        # Compare SA configurations if we have more than one
        if len(all_results["SA"]["configs"]) > 1:
            sa_histories_dict = {config: all_results["SA"]["histories"][i] 
                                for i, config in enumerate(all_results["SA"]["configs"])}
            sa_comparison_plot = compare_convergence_plots(sa_histories_dict, "Simulated Annealing")
    
    # Run Genetic Algorithm if requested
    if run_ga:
        ga_all_start_time = time.time()
        
        for config in ga_configs:
            result, solution, history = run_genetic_algorithm(
                num_runs=NUM_RUNS,
                population_size=config["population_size"],
                generations=config["generations"],
                mutation_rate=config["mutation_rate"],
                elite_size=config["elite_size"],
                mutation_operator_func=config["mutation_operator_func"],
                crossover_operator_func=config["crossover_operator_func"],
                selection_operator_func=config["selection_operator_func"],
                tournament_k=config["tournament_k"],
                boltzmann_temp=config["boltzmann_temp"],
                config_name=config["name"]
            )
        
        ga_all_end_time = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All Genetic Algorithm configurations took {ga_all_end_time - ga_all_start_time:.2f} seconds.")
        
        # Compare GA configurations if we have more than one
        if len(all_results["GA"]["configs"]) > 1:
            ga_histories_dict = {config: all_results["GA"]["histories"][i] 
                                for i, config in enumerate(all_results["GA"]["configs"])}
            ga_comparison_plot = compare_convergence_plots(ga_histories_dict, "Genetic Algorithm", "Generation")
    
    # Cross-algorithm comparison if we have results from multiple algorithms
    if ((run_hc and all_results["HC"]["histories"]) or 
        (run_sa and all_results["SA"]["histories"]) or 
        (run_ga and all_results["GA"]["histories"])):
        
        # Collect all results for final comparison
        all_results_summary = []
        
        # Add HC results
        if run_hc:
            for i, config in enumerate(all_results["HC"]["configs"]):
                all_results_summary.append({
                    "Algorithm": f"Hill Climbing - {config}",
                    "Mean Fitness": np.nanmean(all_results["HC"]["fitness_values"][i]) if all_results["HC"]["fitness_values"][i] else float("nan"),
                    "Std Dev Fitness": np.nanstd(all_results["HC"]["fitness_values"][i]) if all_results["HC"]["fitness_values"][i] else float("nan"),
                    "Mean Exec Time (s)": np.nanmean(all_results["HC"]["exec_times"][i]) if all_results["HC"]["exec_times"][i] else float("nan"),
                    "Overall Best Fitness": min([x for x in all_results["HC"]["fitness_values"][i] if not np.isnan(x)], default=float("nan")),
                    "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
                })
        
        # Add SA results
        if run_sa:
            for i, config in enumerate(all_results["SA"]["configs"]):
                all_results_summary.append({
                    "Algorithm": f"Simulated Annealing - {config}",
                    "Mean Fitness": np.nanmean(all_results["SA"]["fitness_values"][i]) if all_results["SA"]["fitness_values"][i] else float("nan"),
                    "Std Dev Fitness": np.nanstd(all_results["SA"]["fitness_values"][i]) if all_results["SA"]["fitness_values"][i] else float("nan"),
                    "Mean Exec Time (s)": np.nanmean(all_results["SA"]["exec_times"][i]) if all_results["SA"]["exec_times"][i] else float("nan"),
                    "Overall Best Fitness": min([x for x in all_results["SA"]["fitness_values"][i] if not np.isnan(x)], default=float("nan")),
                    "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
                })
        
        # Add GA results
        if run_ga:
            for i, config in enumerate(all_results["GA"]["configs"]):
                all_results_summary.append({
                    "Algorithm": f"Genetic Algorithm - {config}",
                    "Mean Fitness": np.nanmean(all_results["GA"]["fitness_values"][i]) if all_results["GA"]["fitness_values"][i] else float("nan"),
                    "Std Dev Fitness": np.nanstd(all_results["GA"]["fitness_values"][i]) if all_results["GA"]["fitness_values"][i] else float("nan"),
                    "Mean Exec Time (s)": np.nanmean(all_results["GA"]["exec_times"][i]) if all_results["GA"]["exec_times"][i] else float("nan"),
                    "Overall Best Fitness": min([x for x in all_results["GA"]["fitness_values"][i] if not np.isnan(x)], default=float("nan")),
                    "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
                })
        
        # Convert to DataFrame and save
        summary_df = pd.DataFrame(all_results_summary)
        summary_filename = os.path.join(DATA_DIR_RESULTS, "all_algorithms_summary.csv")
        summary_df.to_csv(summary_filename, index=False, sep=";")
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Overall summary saved to {summary_filename}")
        
        # Generate comparative plot for all algorithms
        if ((run_hc and all_results["HC"]["histories"]) and 
            (run_sa and all_results["SA"]["histories"]) and 
            (run_ga and all_results["GA"]["histories"])):
            all_algorithms_plot = compare_all_algorithms()
            
            # Perform statistical tests if we have enough runs
            if NUM_RUNS > 1:
                statistical_results = perform_statistical_tests()
                print(f"Statistical comparison saved to {statistical_results}")
    
    script_total_end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Single-Processor Script execution finished. Total time: {script_total_end_time - script_total_start_time:.2f} seconds.")

# %% [markdown]
# ## Execute All Algorithms
# 
# Run this cell to execute all algorithms in sequence. You can also run individual algorithm cells above for more control.

# %%
# Execute the main function if the script is run directly (or cell is executed in notebook)
if __name__ == "__main__" or __name__ == "builtins":  # For notebook execution
    # Set to True/False to control which algorithms to run
    main(run_hc=True, run_sa=True, run_ga=True, num_runs=NUM_RUNS)
