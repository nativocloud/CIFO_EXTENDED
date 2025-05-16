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
import random
from collections import defaultdict

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

# Define random seeds for reproducibility
USE_FIXED_SEEDS = True  # Set to True for paired statistical tests
RANDOM_SEEDS = [42 + i for i in range(100)]  # Pre-generate 100 seeds

# Dictionary to store all results for comparison
all_results = {
    "HC": {
        "configs": [], 
        "fitness_values": [], 
        "exec_times": [], 
        "histories": [],
        "best_fitness": [],
        "avg_fitness": [],
        "std_dev": [],
        "convergence_speed": [],
        "success_rate": [],
        "fitness_over_time": []
    },
    "SA": {
        "configs": [], 
        "fitness_values": [], 
        "exec_times": [], 
        "histories": [],
        "best_fitness": [],
        "avg_fitness": [],
        "std_dev": [],
        "convergence_speed": [],
        "success_rate": [],
        "fitness_over_time": []
    },
    "GA": {
        "configs": [], 
        "fitness_values": [], 
        "exec_times": [], 
        "histories": [],
        "best_fitness": [],
        "avg_fitness": [],
        "std_dev": [],
        "convergence_speed": [],
        "success_rate": [],
        "fitness_over_time": []
    }
}

# %% [markdown]
# ## Utility Functions

# %%
def calculate_convergence_speed(history, threshold=0.95):
    """
    Calculate convergence speed as the number of iterations needed to reach
    threshold% of the final improvement.
    
    Args:
        history: List of fitness values over iterations
        threshold: Percentage of final improvement to consider as converged (0.0-1.0)
        
    Returns:
        iterations: Number of iterations to reach threshold% improvement
        normalized_speed: Speed normalized to 0-1 range (1 = fastest)
    """
    if not history or len(history) < 2:
        return float('nan'), float('nan')
    
    initial_fitness = history[0]
    final_fitness = history[-1]
    total_improvement = initial_fitness - final_fitness
    
    if total_improvement <= 0:
        return float('nan'), float('nan')  # No improvement
    
    target_improvement = total_improvement * threshold
    target_fitness = initial_fitness - target_improvement
    
    for i, fitness in enumerate(history):
        if fitness <= target_fitness:
            return i, 1.0 - (i / len(history))
    
    return len(history), 0.0  # Did not converge within given iterations

def calculate_success_rate(fitness_values, threshold=None):
    """
    Calculate success rate as percentage of runs that found valid solutions.
    If threshold is provided, only solutions with fitness better than threshold
    are considered successful.
    
    Args:
        fitness_values: List of fitness values from multiple runs
        threshold: Optional fitness threshold to consider a run successful
        
    Returns:
        success_rate: Percentage of successful runs (0-100)
    """
    if not fitness_values:
        return 0.0
    
    # Filter out NaN values (invalid solutions)
    valid_values = [v for v in fitness_values if not np.isnan(v)]
    
    # If threshold is provided, count solutions better than threshold
    if threshold is not None:
        successful_runs = sum(1 for v in valid_values if v <= threshold)
    else:
        successful_runs = len(valid_values)
    
    return (successful_runs / len(fitness_values)) * 100.0

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

def visualize_statistical_comparison(metrics_data=None):
    """
    Create visualizations comparing algorithm performance across different metrics.
    
    Args:
        metrics_data: Dictionary with metrics to visualize. If None, uses all_results
    
    Returns:
        List of saved visualization filepaths
    """
    if metrics_data is None:
        metrics_data = all_results
    
    print("\n--- Creating Statistical Comparison Visualizations ---")
    
    # Metrics to visualize
    metrics = [
        {"name": "Best Fitness", "key": "best_fitness", "lower_is_better": True, "ylabel": "Best Fitness Value"},
        {"name": "Average Fitness", "key": "avg_fitness", "lower_is_better": True, "ylabel": "Average Fitness"},
        {"name": "Standard Deviation", "key": "std_dev", "lower_is_better": True, "ylabel": "Standard Deviation"},
        {"name": "Convergence Speed", "key": "convergence_speed", "lower_is_better": False, "ylabel": "Convergence Speed"},
        {"name": "Execution Time", "key": "exec_times", "lower_is_better": True, "ylabel": "Execution Time (s)"},
        {"name": "Success Rate", "key": "success_rate", "lower_is_better": False, "ylabel": "Success Rate (%)"}
    ]
    
    saved_filepaths = []
    
    # Create boxplots for each metric
    for metric in metrics:
        metric_name = metric["name"]
        metric_key = metric["key"]
        ylabel = metric["ylabel"]
        
        # Prepare data for visualization
        algorithm_names = []
        metric_values = []
        
        # Collect values for each algorithm configuration
        for algo in ["HC", "SA", "GA"]:
            for i, config in enumerate(metrics_data[algo]["configs"]):
                if metric_key in metrics_data[algo] and i < len(metrics_data[algo][metric_key]):
                    values = metrics_data[algo][metric_key][i]
                    if values and len(values) > 0:  # Check if we have values
                        algorithm_names.append(f"{algo} - {config}")
                        metric_values.append(values)
        
        if not algorithm_names or not metric_values:
            print(f"No data available for {metric_name} visualization")
            continue
        
        # Create boxplot
        plt.figure(figsize=(12, 8))
        
        # Create boxplot with individual points
        boxplot = plt.boxplot(metric_values, labels=algorithm_names, patch_artist=True)
        
        # Add individual data points with jitter
        for i, data in enumerate(metric_values):
            # Add jitter to x position
            x = np.random.normal(i+1, 0.04, size=len(data))
            plt.scatter(x, data, alpha=0.5, s=30)
        
        # Customize boxplot colors
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightsteelblue', 'lightpink']
        for i, box in enumerate(boxplot['boxes']):
            box.set(facecolor=colors[i % len(colors)])
        
        plt.title(f"{metric_name} Comparison Across Algorithms")
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(IMAGES_DIR, f"{metric_key}_comparison.png")
        plt.savefig(filepath)
        print(f"Saved {metric_name} comparison visualization to {filepath}")
        plt.close()
        
        saved_filepaths.append(filepath)
    
    # Create fitness over time curve comparison
    if "fitness_over_time" in metrics_data["HC"] or "fitness_over_time" in metrics_data["SA"] or "fitness_over_time" in metrics_data["GA"]:
        plt.figure(figsize=(14, 10))
        
        # Collect fitness over time curves
        for algo in ["HC", "SA", "GA"]:
            for i, config in enumerate(metrics_data[algo]["configs"]):
                if "fitness_over_time" in metrics_data[algo] and i < len(metrics_data[algo]["fitness_over_time"]):
                    curves = metrics_data[algo]["fitness_over_time"][i]
                    if curves and len(curves) > 0:
                        # Average the curves if we have multiple runs
                        avg_curve = np.mean(curves, axis=0)
                        # Normalize x-axis to percentage of iterations
                        x = np.linspace(0, 100, len(avg_curve))
                        plt.plot(x, avg_curve, label=f"{algo} - {config}")
        
        plt.title("Fitness Over Time Comparison")
        plt.xlabel("Percentage of Iterations")
        plt.ylabel("Fitness Value")
        plt.grid(True)
        plt.legend()
        
        # Save figure
        filepath = os.path.join(IMAGES_DIR, "fitness_over_time_comparison.png")
        plt.savefig(filepath)
        print(f"Saved Fitness Over Time comparison visualization to {filepath}")
        plt.close()
        
        saved_filepaths.append(filepath)
    
    # Create summary table
    summary_data = []
    for algo in ["HC", "SA", "GA"]:
        for i, config in enumerate(metrics_data[algo]["configs"]):
            row = {"Algorithm": f"{algo} - {config}"}
            
            for metric in metrics:
                metric_key = metric["key"]
                if metric_key in metrics_data[algo] and i < len(metrics_data[algo][metric_key]):
                    values = metrics_data[algo][metric_key][i]
                    if values and len(values) > 0:
                        row[metric["name"]] = np.mean(values)
                    else:
                        row[metric["name"]] = float('nan')
                else:
                    row[metric["name"]] = float('nan')
            
            summary_data.append(row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary table
        filepath = os.path.join(DATA_DIR_RESULTS, "metrics_summary.csv")
        summary_df.to_csv(filepath, index=False)
        print(f"Saved metrics summary table to {filepath}")
        
        saved_filepaths.append(filepath)
    
    return saved_filepaths

def perform_statistical_tests(metrics_data=None, paired_test=True):
    """
    Perform non-parametric statistical tests to compare algorithm performance.
    
    Following the methodology from Nature Scientific Reports:
    https://www.nature.com/articles/s41598-024-56706-x
    
    Args:
        metrics_data: Dictionary with metrics to analyze. If None, uses all_results
        paired_test: Whether to use paired tests (True) or unpaired tests (False)
    
    Returns:
        results_dict: Dictionary with test results for each metric
    """
    if metrics_data is None:
        metrics_data = all_results
    
    print(f"\n--- Statistical Analysis ({'Paired' if paired_test else 'Unpaired'} Tests) ---")
    print("Null Hypothesis (H₀): The performance metrics are the same across all algorithms")
    print("Alternative Hypothesis (H₁): At least one algorithm performs differently")
    
    # Metrics to analyze
    metrics = [
        {"name": "Best Fitness", "key": "best_fitness", "lower_is_better": True},
        {"name": "Average Fitness", "key": "avg_fitness", "lower_is_better": True},
        {"name": "Standard Deviation", "key": "std_dev", "lower_is_better": True},
        {"name": "Convergence Speed", "key": "convergence_speed", "lower_is_better": False},
        {"name": "Execution Time", "key": "exec_times", "lower_is_better": True},
        {"name": "Success Rate", "key": "success_rate", "lower_is_better": False}
    ]
    
    results_dict = {}
    
    for metric in metrics:
        metric_name = metric["name"]
        metric_key = metric["key"]
        lower_is_better = metric["lower_is_better"]
        
        print(f"\n=== {metric_name} Analysis ===")
        
        # Prepare data for statistical tests
        algorithm_names = []
        metric_values = []
        
        # Collect values for each algorithm configuration
        for algo in ["HC", "SA", "GA"]:
            for i, config in enumerate(metrics_data[algo]["configs"]):
                if metric_key in metrics_data[algo] and i < len(metrics_data[algo][metric_key]):
                    values = metrics_data[algo][metric_key][i]
                    if values and len(values) > 0:  # Check if we have values
                        algorithm_names.append(f"{algo} - {config}")
                        metric_values.append(values)
        
        # Count number of algorithms with valid results
        valid_algorithms = len([values for values in metric_values if len(values) > 0])
        
        if valid_algorithms < 2:
            print(f"Not enough algorithms with valid {metric_name} data for statistical comparison.")
            results_dict[metric_name] = {"status": "insufficient_data"}
            continue
        
        # Determine if we have paired or unpaired data based on parameter and data structure
        is_paired = paired_test and all(len(values) == len(metric_values[0]) for values in metric_values) and NUM_RUNS > 1
        
        print(f"Data type: {'Paired' if is_paired else 'Unpaired'}")
        print(f"Number of algorithms: {valid_algorithms}")
        print(f"Number of runs per algorithm: {NUM_RUNS}")
        
        try:
            if valid_algorithms == 2:
                # Two algorithms comparison
                if is_paired and NUM_RUNS > 1:
                    # Paired data: Wilcoxon signed-rank test
                    print("\nPerforming Wilcoxon signed-rank test (paired data)...")
                    stat, p_value = stats.wilcoxon(metric_values[0], metric_values[1])
                    test_name = "Wilcoxon signed-rank test"
                else:
                    # Unpaired data: Mann-Whitney U test
                    print("\nPerforming Mann-Whitney U test (unpaired data)...")
                    stat, p_value = stats.mannwhitneyu(metric_values[0], metric_values[1], 
                                                      alternative='two-sided')
                    test_name = "Mann-Whitney U test"
                
                print(f"{test_name}: statistic={stat:.4f}, p={p_value:.4f}")
                
                if p_value < 0.05:
                    print(f"Significant difference detected (p < 0.05) between {algorithm_names[0]} and {algorithm_names[1]}")
                    
                    # Determine which algorithm is better based on metric
                    if lower_is_better:
                        better_idx = 0 if np.mean(metric_values[0]) < np.mean(metric_values[1]) else 1
                    else:
                        better_idx = 0 if np.mean(metric_values[0]) > np.mean(metric_values[1]) else 1
                        
                    better_algo = algorithm_names[better_idx]
                    print(f"The better performing algorithm is: {better_algo}")
                    
                    results_dict[metric_name] = {
                        "test": test_name,
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "significant": True,
                        "better_algorithm": better_algo,
                        "algorithms": algorithm_names
                    }
                else:
                    print(f"No significant difference (p >= 0.05) between {algorithm_names[0]} and {algorithm_names[1]}")
                    results_dict[metric_name] = {
                        "test": test_name,
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "significant": False,
                        "algorithms": algorithm_names
                    }
                    
            elif valid_algorithms > 2:
                # Multiple algorithms comparison
                if is_paired and NUM_RUNS > 1:
                    # Paired data: Friedman test followed by Nemenyi
                    print("\nPerforming Friedman test followed by Nemenyi post-hoc test (paired data)...")
                    
                    # Prepare data for Friedman test
                    data_array = np.array([values for values in metric_values])
                    
                    # Friedman test
                    try:
                        from scipy.stats import friedmanchisquare
                        chi2, p_value = friedmanchisquare(*data_array)
                        print(f"Friedman test: chi2={chi2:.4f}, p={p_value:.4f}")
                        
                        results_dict[metric_name] = {
                            "test": "Friedman test",
                            "statistic": float(chi2),
                            "p_value": float(p_value),
                            "significant": p_value < 0.05,
                            "algorithms": algorithm_names,
                            "post_hoc": {}
                        }
                        
                        if p_value < 0.05:
                            print("Significant difference detected (p < 0.05) among algorithms")
                            
                            # Nemenyi post-hoc test
                            try:
                                from scikit_posthocs import posthoc_nemenyi_friedman
                                posthoc = posthoc_nemenyi_friedman(data_array.T)
                                print("\nNemenyi post-hoc test p-values:")
                                
                                # Create DataFrame with algorithm names
                                posthoc_df = pd.DataFrame(posthoc, 
                                                         index=algorithm_names,
                                                         columns=algorithm_names)
                                print(posthoc_df)
                                
                                # Save post-hoc results
                                posthoc_filepath = os.path.join(DATA_DIR_RESULTS, f"nemenyi_posthoc_{metric_key}.csv")
                                posthoc_df.to_csv(posthoc_filepath)
                                print(f"Post-hoc test results saved to {posthoc_filepath}")
                                
                                # Identify significant pairs
                                significant_pairs = []
                                for i in range(len(algorithm_names)):
                                    for j in range(i+1, len(algorithm_names)):
                                        if posthoc[i, j] < 0.05:
                                            if lower_is_better:
                                                better = i if np.mean(metric_values[i]) < np.mean(metric_values[j]) else j
                                            else:
                                                better = i if np.mean(metric_values[i]) > np.mean(metric_values[j]) else j
                                                
                                            worse = j if better == i else i
                                            significant_pairs.append((algorithm_names[better], algorithm_names[worse]))
                                
                                results_dict[metric_name]["post_hoc"] = {
                                    "test": "Nemenyi test",
                                    "p_values": posthoc.tolist(),
                                    "significant_pairs": significant_pairs
                                }
                                
                                if significant_pairs:
                                    print("\nSignificantly different pairs (p < 0.05):")
                                    for better, worse in significant_pairs:
                                        print(f"  {better} performs better than {worse}")
                                else:
                                    print("\nNo algorithm pairs show significant differences in post-hoc tests")
                                    
                            except ImportError:
                                print("scikit-posthocs package not available for Nemenyi test")
                                print("Install with: pip install scikit-posthocs")
                        else:
                            print("No significant difference (p >= 0.05) among algorithms")
                            
                    except ValueError as e:
                        print(f"Error in Friedman test: {e}")
                        print("Falling back to Kruskal-Wallis test...")
                        # Fall back to Kruskal-Wallis
                        is_paired = False
                
                if not is_paired:
                    # Unpaired data: Kruskal-Wallis test followed by Dunn's test
                    print("\nPerforming Kruskal-Wallis test followed by Dunn's post-hoc test (unpaired data)...")
                    
                    # Kruskal-Wallis test
                    stat, p_value = stats.kruskal(*[values for values in metric_values])
                    print(f"Kruskal-Wallis test: H={stat:.4f}, p={p_value:.4f}")
                    
                    results_dict[metric_name] = {
                        "test": "Kruskal-Wallis test",
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,
                        "algorithms": algorithm_names,
                        "post_hoc": {}
                    }
                    
                    if p_value < 0.05:
                        print("Significant difference detected (p < 0.05) among algorithms")
                        
                        # Dunn's post-hoc test
                        try:
                            from scikit_posthocs import posthoc_dunn
                            posthoc = posthoc_dunn([values for values in metric_values], p_adjust='bonferroni')
                            print("\nDunn's post-hoc test p-values (Bonferroni-adjusted):")
                            
                            # Create DataFrame with algorithm names
                            posthoc_df = pd.DataFrame(posthoc, 
                                                     index=algorithm_names,
                                                     columns=algorithm_names)
                            print(posthoc_df)
                            
                            # Save post-hoc results
                            posthoc_filepath = os.path.join(DATA_DIR_RESULTS, f"dunn_posthoc_{metric_key}.csv")
                            posthoc_df.to_csv(posthoc_filepath)
                            print(f"Post-hoc test results saved to {posthoc_filepath}")
                            
                            # Identify significant pairs
                            significant_pairs = []
                            for i in range(len(algorithm_names)):
                                for j in range(i+1, len(algorithm_names)):
                                    if posthoc.iloc[i, j] < 0.05:
                                        if lower_is_better:
                                            better = i if np.mean(metric_values[i]) < np.mean(metric_values[j]) else j
                                        else:
                                            better = i if np.mean(metric_values[i]) > np.mean(metric_values[j]) else j
                                            
                                        worse = j if better == i else i
                                        significant_pairs.append((algorithm_names[better], algorithm_names[worse]))
                            
                            results_dict[metric_name]["post_hoc"] = {
                                "test": "Dunn's test",
                                "p_values": posthoc.values.tolist(),
                                "significant_pairs": significant_pairs
                            }
                            
                            if significant_pairs:
                                print("\nSignificantly different pairs (p < 0.05):")
                                for better, worse in significant_pairs:
                                    print(f"  {better} performs better than {worse}")
                            else:
                                print("\nNo algorithm pairs show significant differences in post-hoc tests")
                                
                        except ImportError:
                            print("scikit-posthocs package not available for Dunn's test")
                            print("Install with: pip install scikit-posthocs")
                    else:
                        print("No significant difference (p >= 0.05) among algorithms")
        except Exception as e:
            print(f"\nError during statistical testing for {metric_name}: {e}")
            print("For meaningful statistical analysis, increase NUM_RUNS (recommended: 30+)")
            results_dict[metric_name] = {"status": "error", "message": str(e)}
    
    print("\nNote: For robust statistical analysis, it's recommended to:")
    print("1. Run each algorithm multiple times (NUM_RUNS >= 30)")
    print("2. Use consistent random seeds across algorithms for paired tests")
    print("3. Interpret results in context of problem domain and computational budget")
    
    # Save results to JSON
    import json
    results_filepath = os.path.join(DATA_DIR_RESULTS, f"statistical_comparison_{'paired' if paired_test else 'unpaired'}.json")
    with open(results_filepath, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nStatistical test results saved to {results_filepath}")
    
    return results_dict

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

# %% [markdown]
# ## Statistical Analysis and Visualization

# %%
def run_statistical_analysis():
    """
    Run both paired and unpaired statistical tests and create visualizations.
    """
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Running Statistical Analysis ---")
    
    # Run paired tests
    print("\n=== PAIRED STATISTICAL TESTS ===")
    paired_results = perform_statistical_tests(paired_test=True)
    
    # Run unpaired tests
    print("\n=== UNPAIRED STATISTICAL TESTS ===")
    unpaired_results = perform_statistical_tests(paired_test=False)
    
    # Create visualizations
    print("\n=== CREATING VISUALIZATIONS ===")
    visualization_files = visualize_statistical_comparison()
    
    # Create comparison table between paired and unpaired results
    print("\n=== COMPARING PAIRED VS UNPAIRED RESULTS ===")
    
    comparison_data = []
    
    # Get all unique metrics from both result sets
    all_metrics = set(paired_results.keys()) | set(unpaired_results.keys())
    
    for metric in all_metrics:
        if metric in paired_results and metric in unpaired_results:
            paired_test = paired_results[metric].get("test", "N/A")
            paired_p = paired_results[metric].get("p_value", float('nan'))
            paired_sig = paired_results[metric].get("significant", False)
            
            unpaired_test = unpaired_results[metric].get("test", "N/A")
            unpaired_p = unpaired_results[metric].get("p_value", float('nan'))
            unpaired_sig = unpaired_results[metric].get("significant", False)
            
            # Check if results agree or disagree
            agreement = paired_sig == unpaired_sig
            
            comparison_data.append({
                "Metric": metric,
                "Paired Test": paired_test,
                "Paired p-value": paired_p,
                "Paired Significant": paired_sig,
                "Unpaired Test": unpaired_test,
                "Unpaired p-value": unpaired_p,
                "Unpaired Significant": unpaired_sig,
                "Agreement": agreement
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        comparison_filepath = os.path.join(DATA_DIR_RESULTS, "paired_vs_unpaired_comparison.csv")
        comparison_df.to_csv(comparison_filepath, index=False)
        print(f"Saved paired vs unpaired comparison to {comparison_filepath}")
        
        # Print summary
        print("\nSummary of paired vs unpaired test agreement:")
        agreement_count = sum(1 for row in comparison_data if row["Agreement"])
        print(f"Agreement rate: {agreement_count}/{len(comparison_data)} metrics ({agreement_count/len(comparison_data)*100:.1f}%)")
        
        # List disagreements
        disagreements = [row["Metric"] for row in comparison_data if not row["Agreement"]]
        if disagreements:
            print(f"Metrics with disagreement between paired and unpaired tests: {', '.join(disagreements)}")
        else:
            print("All metrics show agreement between paired and unpaired tests.")
    
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Statistical analysis completed.")
    
    return {
        "paired_results": paired_results,
        "unpaired_results": unpaired_results,
        "visualization_files": visualization_files
    }
