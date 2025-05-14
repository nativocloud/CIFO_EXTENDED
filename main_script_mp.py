"""
This script is a modified version for implementing multiprocessing to speed up algorithm runs.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing # Added for multiprocessing
# from functools import partial # Not strictly needed if args are tuples for map

from solution import LeagueSolution, LeagueHillClimbingSolution, LeagueSASolution
from evolution import genetic_algorithm, hill_climbing, simulated_annealing
from operators import (
    mutate_swap_constrained,
    mutate_targeted_player_exchange,
    mutate_shuffle_within_team_constrained,
    crossover_one_point_prefer_valid,
    crossover_uniform_prefer_valid,
    selection_ranking,
    selection_tournament_variable_k,
    selection_boltzmann
)

# Load player data (remains global for now, passed to workers)
players_df = pd.read_csv("players.csv", sep=";")
players_data_global = players_df.to_dict(orient="records")

# Define problem parameters (can be global or passed)
NUM_TEAMS_GLOBAL = 5
TEAM_SIZE_GLOBAL = 7
MAX_BUDGET_GLOBAL = 750
NUM_RUNS_GLOBAL = 30 # Default, will be used by the main part of the script

# --- Worker function for Hill Climbing ---
def hc_worker(args_tuple):
    run_id, p_data, n_teams, t_size, m_budget, max_iter, verbose_flag = args_tuple
    start_time_hc_run = time.time()
    initial_hc_solution_run = LeagueHillClimbingSolution(p_data, num_teams=n_teams, team_size=t_size, max_budget=m_budget)
    retry_attempts_hc = 0
    max_retry_hc = 5
    while not initial_hc_solution_run.is_valid(p_data) and retry_attempts_hc < max_retry_hc:
        initial_hc_solution_run = LeagueHillClimbingSolution(p_data, num_teams=n_teams, team_size=t_size, max_budget=m_budget)
        retry_attempts_hc += 1
    if not initial_hc_solution_run.is_valid(p_data):
        return {"run_id": run_id, "fitness": float('nan'), "exec_time": time.time() - start_time_hc_run, "solution_assignment": None, "history": []}
    hc_solution_obj_run, hc_fitness_val_run, hc_history_convergence_run = hill_climbing(
        initial_solution=initial_hc_solution_run, 
        players_data=p_data, 
        max_iterations=max_iter, 
        verbose=verbose_flag
    )
    end_time_hc_run = time.time()
    hc_exec_time_run = end_time_hc_run - start_time_hc_run
    if hc_solution_obj_run:
        return {"run_id": run_id, "fitness": hc_fitness_val_run, "exec_time": hc_exec_time_run, "solution_assignment": hc_solution_obj_run.assignment, "history": hc_history_convergence_run}
    else:
        return {"run_id": run_id, "fitness": float('nan'), "exec_time": hc_exec_time_run, "solution_assignment": None, "history": []}

# --- Worker function for Simulated Annealing ---
def sa_worker(args_tuple):
    run_id, p_data, n_teams, t_size, m_budget, sa_params_dict, verbose_flag = args_tuple
    start_time_sa_run = time.time()
    initial_sa_solution = LeagueSASolution(p_data, num_teams=n_teams, team_size=t_size, max_budget=m_budget)
    retry_attempts_sa = 0
    max_retry_sa = 5
    while not initial_sa_solution.is_valid(p_data) and retry_attempts_sa < max_retry_sa:
        initial_sa_solution = LeagueSASolution(p_data, num_teams=n_teams, team_size=t_size, max_budget=m_budget)
        retry_attempts_sa += 1
    if not initial_sa_solution.is_valid(p_data):
        return {"run_id": run_id, "fitness": float('nan'), "exec_time": time.time() - start_time_sa_run, "solution_assignment": None, "history": []}
    sa_solution_run, sa_fitness_run, sa_history_run = simulated_annealing(
        initial_solution=initial_sa_solution,
        players_data=p_data,
        initial_temp=sa_params_dict["initial_temp"],
        final_temp=sa_params_dict["final_temp"],
        alpha=sa_params_dict["alpha"],
        iterations_per_temp=sa_params_dict["iterations_per_temp"],
        verbose=verbose_flag
    )
    end_time_sa_run = time.time()
    sa_exec_time_run = end_time_sa_run - start_time_sa_run
    if sa_solution_run:
        return {"run_id": run_id, "fitness": sa_fitness_run, "exec_time": sa_exec_time_run, "solution_assignment": sa_solution_run.assignment, "history": sa_history_run}
    else:
        return {"run_id": run_id, "fitness": float('nan'), "exec_time": sa_exec_time_run, "solution_assignment": None, "history": []}

# --- Worker function for Genetic Algorithm ---
def ga_worker(args_tuple):
    run_id, p_data, n_teams, t_size, m_budget, ga_config_dict, ga_params_dict, verbose_flag = args_tuple
    start_time_ga_run = time.time()
    best_solution_ga_run, history_ga_run = genetic_algorithm(
        players_data=p_data,
        num_teams=n_teams,
        team_size=t_size,
        max_budget=m_budget,
        population_size=ga_params_dict["population_size"],
        generations=ga_params_dict["generations"],
        mutation_rate=ga_params_dict["mutation_rate"],
        mutation_operator_func=ga_config_dict["mutation_operator_func"],
        crossover_operator_func=ga_config_dict["crossover_operator_func"],
        selection_operator_func=ga_config_dict["selection_operator_func"],
        tournament_k=ga_config_dict.get("tournament_k"),
        boltzmann_temp=ga_config_dict.get("boltzmann_temp"),
        # elitism_size is handled internally by genetic_algorithm based on its signature in evolution.py
        # crossover_rate is also handled internally by genetic_algorithm
        verbose=verbose_flag
    )
    end_time_ga_run = time.time()
    ga_exec_time_run = end_time_ga_run - start_time_ga_run

    if best_solution_ga_run:
        best_fitness_ga_run = best_solution_ga_run.fitness(p_data) # Calculate fitness here
        return {"run_id": run_id, "fitness": best_fitness_ga_run, "exec_time": ga_exec_time_run, "solution_assignment": best_solution_ga_run.assignment, "history": history_ga_run, "config_name": ga_config_dict["name"]}
    else:
        # If best_solution_ga_run is None, it means GA failed to produce any result (e.g. empty initial population)
        return {"run_id": run_id, "fitness": float("nan"), "exec_time": ga_exec_time_run, "solution_assignment": None, "history": [], "config_name": ga_config_dict["name"]}


if __name__ == '__main__':
    script_start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Script execution started.")

    try:
        num_processes = multiprocessing.cpu_count()
    except NotImplementedError:
        num_processes = 4 # Default if cpu_count() is not available
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using {num_processes} processes for parallel execution.")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading player data...")
    # Player data is already loaded globally, this is just a log point
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Player data loaded successfully. Total players: {len(players_data_global)}")
    if players_data_global:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] First player data: {players_data_global[0]}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All algorithms (HC, SA, GA) will be run {NUM_RUNS_GLOBAL} times each using multiprocessing.")

    all_results_summary = []

    # ---- 1. Hill Climbing ----
    hc_section_start_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting Hill Climbing Algorithm ({NUM_RUNS_GLOBAL} runs) ---")
    hc_args_list = []
    for i in range(NUM_RUNS_GLOBAL):
        hc_args_list.append((i, players_data_global, NUM_TEAMS_GLOBAL, TEAM_SIZE_GLOBAL, MAX_BUDGET_GLOBAL, 1000, False))
    
    hc_results_parallel = []
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launching Hill Climbing worker pool...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        hc_results_parallel = pool.map(hc_worker, hc_args_list)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hill Climbing worker pool finished.")

    hc_all_fitness_values = [res["fitness"] for res in hc_results_parallel]
    hc_all_exec_times = [res["exec_time"] for res in hc_results_parallel]
    best_hc_fitness_overall = float("inf")
    best_hc_solution_assignment_overall = None
    best_hc_history_overall = []

    for res in hc_results_parallel:
        if res["solution_assignment"] is not None and not np.isnan(res["fitness"]) and res["fitness"] < best_hc_fitness_overall:
            best_hc_fitness_overall = res["fitness"]
            best_hc_solution_assignment_overall = res["solution_assignment"]
            best_hc_history_overall = res["history"]

    hc_mean_fitness = np.nanmean(hc_all_fitness_values) if hc_all_fitness_values else float("nan")
    hc_std_fitness = np.nanstd(hc_all_fitness_values) if hc_all_fitness_values else float("nan")
    hc_mean_exec_time_per_run = np.nanmean(hc_all_exec_times) if hc_all_exec_times else float("nan")
    hc_section_end_time = time.time()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hill Climbing ({NUM_RUNS_GLOBAL} runs) processing finished.")
    print(f"  Mean Best Fitness: {hc_mean_fitness:.4f}")
    print(f"  Std Dev Best Fitness: {hc_std_fitness:.4f}")
    print(f"  Mean Execution Time per run: {hc_mean_exec_time_per_run:.2f}s")
    if best_hc_solution_assignment_overall is not None:
        print(f"  Overall Best HC Fitness: {best_hc_fitness_overall:.4f}")
        all_results_summary.append({
            "Algorithm": "Hill Climbing", 
            "Mean Fitness": hc_mean_fitness, 
            "Std Dev Fitness": hc_std_fitness, 
            "Mean Exec Time (s)": hc_mean_exec_time_per_run,
            "Overall Best Fitness": best_hc_fitness_overall,
            "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
        })
        plt.figure(figsize=(10, 6))
        plt.plot(best_hc_history_overall, marker="o", linestyle="-")
        plt.title(f"Hill Climbing Convergence (Best of {NUM_RUNS_GLOBAL} Runs - Parallel)")
        plt.xlabel("Improvement Step")
        plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
        plt.grid(True)
        plt.savefig("hc_convergence_parallel.png")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved Hill Climbing convergence plot to hc_convergence_parallel.png")
        plt.close()
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hill Climbing did not find any valid solution across all runs that produced a best overall.")
        all_results_summary.append({
            "Algorithm": "Hill Climbing", 
            "Mean Fitness": hc_mean_fitness, 
            "Std Dev Fitness": hc_std_fitness, 
            "Mean Exec Time (s)": hc_mean_exec_time_per_run,
            "Overall Best Fitness": float('nan'),
            "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
        })
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hill Climbing section took {hc_section_end_time - hc_section_start_time:.2f} seconds.")

    # ---- 2. Simulated Annealing ----
    sa_section_start_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting Simulated Annealing Algorithm ({NUM_RUNS_GLOBAL} runs) ---")
    sa_params_dict = {
        "initial_temp": 1000,
        "final_temp": 0.1,
        "alpha": 0.99,
        "iterations_per_temp": 50
    }
    sa_args_list = []
    for i in range(NUM_RUNS_GLOBAL):
        sa_args_list.append((i, players_data_global, NUM_TEAMS_GLOBAL, TEAM_SIZE_GLOBAL, MAX_BUDGET_GLOBAL, sa_params_dict, False))

    sa_results_parallel = []
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launching Simulated Annealing worker pool...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        sa_results_parallel = pool.map(sa_worker, sa_args_list)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Simulated Annealing worker pool finished.")

    sa_all_fitness_values = [res["fitness"] for res in sa_results_parallel]
    sa_all_exec_times = [res["exec_time"] for res in sa_results_parallel]
    best_sa_fitness_overall = float("inf")
    best_sa_solution_assignment_overall = None
    best_sa_history_overall = []

    for res in sa_results_parallel:
        if res["solution_assignment"] is not None and not np.isnan(res["fitness"]) and res["fitness"] < best_sa_fitness_overall:
            best_sa_fitness_overall = res["fitness"]
            best_sa_solution_assignment_overall = res["solution_assignment"]
            best_sa_history_overall = res["history"]
    
    sa_mean_fitness = np.nanmean(sa_all_fitness_values) if sa_all_fitness_values else float("nan")
    sa_std_fitness = np.nanstd(sa_all_fitness_values) if sa_all_fitness_values else float("nan")
    sa_mean_exec_time_per_run = np.nanmean(sa_all_exec_times) if sa_all_exec_times else float("nan")
    sa_section_end_time = time.time()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Simulated Annealing ({NUM_RUNS_GLOBAL} runs) processing finished.")
    print(f"  Mean Best Fitness: {sa_mean_fitness:.4f}")
    print(f"  Std Dev Best Fitness: {sa_std_fitness:.4f}")
    print(f"  Mean Execution Time per run: {sa_mean_exec_time_per_run:.2f}s")
    if best_sa_solution_assignment_overall is not None:
        print(f"  Overall Best SA Fitness: {best_sa_fitness_overall:.4f}")
        all_results_summary.append({
            "Algorithm": "Simulated Annealing", 
            "Mean Fitness": sa_mean_fitness, 
            "Std Dev Fitness": sa_std_fitness, 
            "Mean Exec Time (s)": sa_mean_exec_time_per_run,
            "Overall Best Fitness": best_sa_fitness_overall,
            "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
        })
        plt.figure(figsize=(10, 6))
        plt.plot(best_sa_history_overall, linestyle="-")
        plt.title(f"Simulated Annealing Convergence (Best of {NUM_RUNS_GLOBAL} Runs - Parallel)")
        plt.xlabel("Iteration Step")
        plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
        plt.grid(True)
        plt.savefig("sa_convergence_parallel.png")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved Simulated Annealing convergence plot to sa_convergence_parallel.png")
        plt.close()
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Simulated Annealing did not find any valid solution across all runs that produced a best overall.")
        all_results_summary.append({
            "Algorithm": "Simulated Annealing", 
            "Mean Fitness": sa_mean_fitness, 
            "Std Dev Fitness": sa_std_fitness, 
            "Mean Exec Time (s)": sa_mean_exec_time_per_run,
            "Overall Best Fitness": float('nan'),
            "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
        })
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Simulated Annealing section took {sa_section_end_time - sa_section_start_time:.2f} seconds.")

    # ---- 3. Genetic Algorithms ----
    ga_section_start_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting Genetic Algorithms ({NUM_RUNS_GLOBAL} runs per config) ---")
    ga_params_dict = {
        "population_size": 50,
        "generations": 100,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "elitism_size": 2
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
            "tournament_k": 5, 
            "boltzmann_temp": None 
        }
    ]

    ga_overall_results_by_config = {}

    for config in ga_configs_new:
        config_start_time = time.time()
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]   Running GA Configuration: {config['name']}")
        ga_args_list = []
        for i in range(NUM_RUNS_GLOBAL):
            ga_args_list.append((i, players_data_global, NUM_TEAMS_GLOBAL, TEAM_SIZE_GLOBAL, MAX_BUDGET_GLOBAL, config, ga_params_dict, False))
        
        ga_results_config_parallel = []
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]   Launching GA worker pool for {config['name']}...")
        with multiprocessing.Pool(processes=num_processes) as pool:
            ga_results_config_parallel = pool.map(ga_worker, ga_args_list)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]   GA worker pool for {config['name']} finished.")

        ga_config_all_fitness = [res["fitness"] for res in ga_results_config_parallel]
        ga_config_all_exec_times = [res["exec_time"] for res in ga_results_config_parallel]
        best_ga_fitness_config_overall = float("inf")
        best_ga_solution_assignment_config_overall = None
        best_ga_history_config_overall = []

        for res in ga_results_config_parallel:
            if res["solution_assignment"] is not None and not np.isnan(res["fitness"]) and res["fitness"] < best_ga_fitness_config_overall:
                best_ga_fitness_config_overall = res["fitness"]
                best_ga_solution_assignment_config_overall = res["solution_assignment"]
                best_ga_history_config_overall = res["history"]

        mean_fitness_config = np.nanmean(ga_config_all_fitness) if ga_config_all_fitness else float("nan")
        std_fitness_config = np.nanstd(ga_config_all_fitness) if ga_config_all_fitness else float("nan")
        mean_exec_time_config_per_run = np.nanmean(ga_config_all_exec_times) if ga_config_all_exec_times else float("nan")
        config_end_time = time.time()

        ga_overall_results_by_config[config["name"]] = {
            "mean_fitness": mean_fitness_config,
            "std_fitness": std_fitness_config,
            "mean_exec_time": mean_exec_time_config_per_run,
            "overall_best_fitness": best_ga_fitness_config_overall if best_ga_solution_assignment_config_overall is not None else float('nan'),
            "best_solution_assignment": best_ga_solution_assignment_config_overall,
            "best_history": best_ga_history_config_overall,
            "all_fitness_values": ga_config_all_fitness,
            "all_exec_times": ga_config_all_exec_times
        }
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]   GA Config {config['name']} ({NUM_RUNS_GLOBAL} runs) processing finished.")
        print(f"    Mean Best Fitness: {mean_fitness_config:.4f}")
        print(f"    Std Dev Best Fitness: {std_fitness_config:.4f}")
        print(f"    Mean Execution Time per run: {mean_exec_time_config_per_run:.2f}s")
        if best_ga_solution_assignment_config_overall is not None:
            print(f"    Overall Best GA Fitness for Config: {best_ga_fitness_config_overall:.4f}")
            all_results_summary.append({
                "Algorithm": config["name"], 
                "Mean Fitness": mean_fitness_config, 
                "Std Dev Fitness": std_fitness_config, 
                "Mean Exec Time (s)": mean_exec_time_config_per_run,
                "Overall Best Fitness": best_ga_fitness_config_overall,
                "Mutation Op": config["mutation_operator_func"].__name__,
                "Crossover Op": config["crossover_operator_func"].__name__,
                "Selection Op": config["selection_operator_func"].__name__
            })
            plt.figure(figsize=(10, 6))
            plt.plot(best_ga_history_config_overall)
            plt.title(f"GA Convergence: {config['name']} (Best of {NUM_RUNS_GLOBAL} Runs - Parallel)")
            plt.xlabel("Generation")
            plt.ylabel("Best Fitness in Population")
            plt.grid(True)
            plot_filename = f"ga_convergence_parallel_{config['name'].replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png"
            plt.savefig(plot_filename)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]   Saved GA convergence plot to {plot_filename}")
            plt.close()
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]   GA Config {config['name']} did not find any valid solution across all runs that produced a best overall.")
            all_results_summary.append({
                "Algorithm": config["name"], 
                "Mean Fitness": mean_fitness_config, 
                "Std Dev Fitness": std_fitness_config, 
                "Mean Exec Time (s)": mean_exec_time_config_per_run,
                "Overall Best Fitness": float('nan'),
                "Mutation Op": config["mutation_operator_func"].__name__,
                "Crossover Op": config["crossover_operator_func"].__name__,
                "Selection Op": config["selection_operator_func"].__name__
            })
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]   GA Configuration {config['name']} took {config_end_time - config_start_time:.2f} seconds.")
    ga_section_end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Genetic Algorithms section took {ga_section_end_time - ga_section_start_time:.2f} seconds.")

    # ---- 4. Comparative Analysis ----
    analysis_start_time = time.time()
    print(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting Comparative Analysis ---")
    results_df = pd.DataFrame(all_results_summary)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Results Summary Table:\n{results_df.to_string()}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating comparative fitness plot...")
    plt.figure(figsize=(14, 8)) # Adjusted size for better label visibility
    alg_names = results_df["Algorithm"]
    mean_fitness_values = results_df["Mean Fitness"]
    std_dev_fitness = results_df["Std Dev Fitness"]
    # Filter out NaN values for plotting to avoid warnings/errors
    valid_indices = ~np.isnan(mean_fitness_values)
    plt.bar(alg_names[valid_indices], mean_fitness_values[valid_indices], yerr=std_dev_fitness[valid_indices], capsize=5, color=["blue", "green", "red", "purple", "orange", "cyan", "magenta", "yellow"][:sum(valid_indices)])
    plt.xlabel("Algorithm Configuration")
    plt.ylabel("Mean Best Fitness (Lower is Better)")
    plt.title("Comparison of Algorithm Performance (Mean Fitness with Std Dev) - Parallel")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("comparative_fitness_parallel.png")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved comparative fitness plot to comparative_fitness_parallel.png")
    plt.close()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating comparative execution times plot...")
    plt.figure(figsize=(14, 8)) # Adjusted size
    mean_exec_times = results_df["Mean Exec Time (s)"]
    valid_indices_time = ~np.isnan(mean_exec_times)
    plt.bar(alg_names[valid_indices_time], mean_exec_times[valid_indices_time], color=["blue", "green", "red", "purple", "orange", "cyan", "magenta", "yellow"][:sum(valid_indices_time)])
    plt.xlabel("Algorithm Configuration")
    plt.ylabel("Mean Execution Time per run (s)")
    plt.title("Comparison of Algorithm Execution Times - Parallel")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("comparative_times_parallel.png")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved comparative execution times plot to comparative_times_parallel.png")
    plt.close()
    analysis_end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Comparative Analysis section took {analysis_end_time - analysis_start_time:.2f} seconds.")

    script_end_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Script finished. Total execution time: {script_end_time - script_start_time:.2f} seconds.")

