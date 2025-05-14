"""
This script is a modified version for implementing multiprocessing to speed up algorithm runs.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing # Added for multiprocessing
import os # Added for path joining

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
        elite_size=ga_params_dict["elitism_size"], 
        mutation_operator_func=ga_config_dict["mutation_operator_func"],
        crossover_operator_func=ga_config_dict["crossover_operator_func"],
        selection_operator_func=ga_config_dict["selection_operator_func"],
        tournament_k=ga_config_dict.get("tournament_k"),
        boltzmann_temp=ga_config_dict.get("boltzmann_temp"),
        verbose=verbose_flag
    )
    end_time_ga_run = time.time()
    ga_exec_time_run = end_time_ga_run - start_time_ga_run

    if best_solution_ga_run:
        best_fitness_ga_run = best_solution_ga_run.fitness(p_data) 
        return {"run_id": run_id, "fitness": best_fitness_ga_run, "exec_time": ga_exec_time_run, "solution_assignment": best_solution_ga_run.assignment, "history": history_ga_run, "config_name": ga_config_dict["name"]}
    else:
        return {"run_id": run_id, "fitness": float('nan'), "exec_time": ga_exec_time_run, "solution_assignment": None, "history": [], "config_name": ga_config_dict["name"]}


if __name__ == '__main__':
    script_start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Multi-Processor Script execution started.")

    IMAGES_MP_DIR = "/home/ubuntu/CIFO_EXTENDED_Project/images_mp" # Updated directory
    if not os.path.exists(IMAGES_MP_DIR):
        os.makedirs(IMAGES_MP_DIR)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Created directory: {IMAGES_MP_DIR}")

    try:
        num_processes = multiprocessing.cpu_count()
    except NotImplementedError:
        num_processes = 4 
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using {num_processes} processes for parallel execution.")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading player data...")
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
            "Algorithm": "Hill Climbing (MP)", 
            "Mean Fitness": hc_mean_fitness, 
            "Std Dev Fitness": hc_std_fitness, 
            "Mean Exec Time (s)": hc_mean_exec_time_per_run,
            "Overall Best Fitness": best_hc_fitness_overall,
            "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
        })
        plt.figure(figsize=(10, 6))
        plt.plot(best_hc_history_overall, marker="o", linestyle="-")
        plt.title(f"Hill Climbing Convergence (Best of {NUM_RUNS_GLOBAL} Runs - MP)")
        plt.xlabel("Improvement Step")
        plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
        plt.grid(True)
        plt.savefig(os.path.join(IMAGES_MP_DIR, "hc_convergence_mp.png")) # Updated path
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved Hill Climbing convergence plot to {IMAGES_MP_DIR}/hc_convergence_mp.png")
        plt.close()
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hill Climbing did not find any valid solution across all runs that produced a best overall.")
        all_results_summary.append({
            "Algorithm": "Hill Climbing (MP)", 
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
            "Algorithm": "Simulated Annealing (MP)", 
            "Mean Fitness": sa_mean_fitness, 
            "Std Dev Fitness": sa_std_fitness, 
            "Mean Exec Time (s)": sa_mean_exec_time_per_run,
            "Overall Best Fitness": best_sa_fitness_overall,
            "Mutation Op": "N/A", "Crossover Op": "N/A", "Selection Op": "N/A"
        })
        plt.figure(figsize=(10, 6))
        plt.plot(best_sa_history_overall, linestyle="-")
        plt.title(f"Simulated Annealing Convergence (Best of {NUM_RUNS_GLOBAL} Runs - MP)")
        plt.xlabel("Iteration Step")
        plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
        plt.grid(True)
        plt.savefig(os.path.join(IMAGES_MP_DIR, "sa_convergence_mp.png")) # Updated path
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved Simulated Annealing convergence plot to {IMAGES_MP_DIR}/sa_convergence_mp.png")
        plt.close()
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Simulated Annealing did not find any valid solution across all runs that produced a best overall.")
        all_results_summary.append({
            "Algorithm": "Simulated Annealing (MP)", 
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
        "mutation_rate": 0.2, 
        "crossover_rate": 0.8, 
        "elitism_size": 2 
    }
    ga_configs_new = [
        {
            "name": "GA_Cfg1_AdSwapC_Ad1PtPV_SelTournK3",
            "mutation_operator_func": mutate_swap_constrained,
            "crossover_operator_func": crossover_one_point_prefer_valid,
            "selection_operator_func": selection_tournament_variable_k,
            "tournament_k": 3,
            "boltzmann_temp": None 
        },
        {
            "name": "GA_Cfg2_AdTargEx_AdUnifPV_SelRank",
            "mutation_operator_func": mutate_targeted_player_exchange,
            "crossover_operator_func": crossover_uniform_prefer_valid,
            "selection_operator_func": selection_ranking,
            "tournament_k": None, 
            "boltzmann_temp": None 
        },
        {
            "name": "GA_Cfg3_AdShufWTC_Ad1PtPV_SelBoltz",
            "mutation_operator_func": mutate_shuffle_within_team_constrained,
            "crossover_operator_func": crossover_one_point_prefer_valid,
            "selection_operator_func": selection_boltzmann,
            "tournament_k": None, 
            "boltzmann_temp": 100 
        },
        {
            "name": "GA_Cfg4_AdTargEx_AdUnifPV_SelTournK5",
            "mutation_operator_func": mutate_targeted_player_exchange,
            "crossover_operator_func": crossover_uniform_prefer_valid,
            "selection_operator_func": selection_tournament_variable_k,
            "tournament_k": 5,
            "boltzmann_temp": None 
        },
        {
            "name": "GA_Cfg5_BaseSwap_Base1Pt_SelRank",
            "mutation_operator_func": mutate_swap,
            "crossover_operator_func": crossover_one_point,
            "selection_operator_func": selection_ranking,
            "tournament_k": None,
            "boltzmann_temp": None
        },
        {
            "name": "GA_Cfg6_BaseTeamShift_AdUnifPV_SelTournK3",
            "mutation_operator_func": mutate_team_shift,
            "crossover_operator_func": crossover_uniform_prefer_valid,
            "selection_operator_func": selection_tournament_variable_k,
            "tournament_k": 3,
            "boltzmann_temp": None
        },
        {
            "name": "GA_Cfg7_AdSwapC_BaseUnif_SelBoltz",
            "mutation_operator_func": mutate_swap_constrained,
            "crossover_operator_func": crossover_uniform,
            "selection_operator_func": selection_boltzmann,
            "tournament_k": None,
            "boltzmann_temp": 100
        },
        {
            "name": "GA_Cfg8_BaseShuffle_BaseUnif_SelTournK4",
            "mutation_operator_func": mutate_shuffle_team,
            "crossover_operator_func": crossover_uniform,
            "selection_operator_func": selection_tournament_variable_k,
            "tournament_k": 4,
            "boltzmann_temp": None
        },
        {
            "name": "GA_Cfg9_AdTargEx_Base1Pt_SelRank",
            "mutation_operator_func": mutate_targeted_player_exchange,
            "crossover_operator_func": crossover_one_point,
            "selection_operator_func": selection_ranking,
            "tournament_k": None,
            "boltzmann_temp": None
        }
    ]

    ga_args_list_master = []
    for config_idx, ga_config_dict in enumerate(ga_configs_new):
        for i in range(NUM_RUNS_GLOBAL):
            ga_args_list_master.append(( (config_idx * NUM_RUNS_GLOBAL) + i, players_data_global, NUM_TEAMS_GLOBAL, TEAM_SIZE_GLOBAL, MAX_BUDGET_GLOBAL, ga_config_dict, ga_params_dict, False) )

    ga_results_parallel_all_configs = []
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launching Genetic Algorithm worker pool for {len(ga_configs_new)} configurations...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        ga_results_parallel_all_configs = pool.map(ga_worker, ga_args_list_master)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Genetic Algorithm worker pool finished.")

    for config_idx, ga_config_dict in enumerate(ga_configs_new):
        config_name = ga_config_dict["name"]
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing results for GA Configuration: {config_name}")
        
        current_config_results = [res for res in ga_results_parallel_all_configs if res["config_name"] == config_name]
        
        ga_all_fitness_values_config = [res["fitness"] for res in current_config_results]
        ga_all_exec_times_config = [res["exec_time"] for res in current_config_results]
        
        best_ga_fitness_overall_config = float("inf")
        best_ga_solution_assignment_overall_config = None
        best_ga_history_overall_config = []

        for res in current_config_results:
            if res["solution_assignment"] is not None and not np.isnan(res["fitness"]) and res["fitness"] < best_ga_fitness_overall_config:
                best_ga_fitness_overall_config = res["fitness"]
                best_ga_solution_assignment_overall_config = res["solution_assignment"]
                best_ga_history_overall_config = res["history"]

        ga_mean_fitness_config = np.nanmean(ga_all_fitness_values_config) if ga_all_fitness_values_config else float("nan")
        ga_std_fitness_config = np.nanstd(ga_all_fitness_values_config) if ga_all_fitness_values_config else float("nan")
        ga_mean_exec_time_config = np.nanmean(ga_all_exec_times_config) if ga_all_exec_times_config else float("nan")

        print(f"  Mean Best Fitness (Config: {config_name}): {ga_mean_fitness_config:.4f}")
        print(f"  Std Dev Best Fitness (Config: {config_name}): {ga_std_fitness_config:.4f}")
        print(f"  Mean Execution Time per run (Config: {config_name}): {ga_mean_exec_time_config:.2f}s")
        
        if best_ga_solution_assignment_overall_config is not None:
            print(f"  Overall Best GA Fitness (Config: {config_name}): {best_ga_fitness_overall_config:.4f}")
            all_results_summary.append({
                "Algorithm": f"{config_name} (MP)", 
                "Mean Fitness": ga_mean_fitness_config, 
                "Std Dev Fitness": ga_std_fitness_config, 
                "Mean Exec Time (s)": ga_mean_exec_time_config,
                "Overall Best Fitness": best_ga_fitness_overall_config,
                "Mutation Op": ga_config_dict["mutation_operator_func"].__name__,
                "Crossover Op": ga_config_dict["crossover_operator_func"].__name__,
                "Selection Op": ga_config_dict["selection_operator_func"].__name__
            })
            plt.figure(figsize=(10, 6))
            plt.plot(best_ga_history_overall_config, marker='.', linestyle='-')
            plt.title(f"GA Convergence ({config_name} - Best of {NUM_RUNS_GLOBAL} Runs - MP)")
            plt.xlabel("Generation")
            plt.ylabel("Fitness (Std Dev of Avg Team Skills)")
            plt.grid(True)
            safe_config_name = config_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
            plt.savefig(os.path.join(IMAGES_MP_DIR, f"ga_convergence_mp_{safe_config_name}.png")) # Updated path
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved GA convergence plot to {IMAGES_MP_DIR}/ga_convergence_mp_{safe_config_name}.png")
            plt.close()
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GA Config {config_name} did not find any valid solution across all runs that produced a best overall.")
            all_results_summary.append({
                "Algorithm": f"{config_name} (MP)", 
                "Mean Fitness": ga_mean_fitness_config, 
                "Std Dev Fitness": ga_std_fitness_config, 
                "Mean Exec Time (s)": ga_mean_exec_time_config,
                "Overall Best Fitness": float('nan'),
                "Mutation Op": ga_config_dict["mutation_operator_func"].__name__,
                "Crossover Op": ga_config_dict["crossover_operator_func"].__name__,
                "Selection Op": ga_config_dict["selection_operator_func"].__name__
            })
    ga_section_end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Genetic Algorithms section took {ga_section_end_time - ga_section_start_time:.2f} seconds.")

    # ---- Comparative Analysis ----
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Generating Comparative Analysis Plots (MP) ---")
    results_df = pd.DataFrame(all_results_summary)
    print("\nOverall Results Summary Table (Multi-Processor):")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df)
    results_df.to_csv(os.path.join(IMAGES_MP_DIR, "comparative_results_mp.csv"), index=False) # Updated path
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved comparative results CSV to {IMAGES_MP_DIR}/comparative_results_mp.csv")

    plt.figure(figsize=(15, 8))
    plot_df_fitness = results_df.dropna(subset=['Overall Best Fitness'])
    plt.bar(plot_df_fitness["Algorithm"], plot_df_fitness["Overall Best Fitness"], color="skyblue")
    plt.xlabel("Algorithm Configuration")
    plt.ylabel("Overall Best Fitness (Std Dev)")
    plt.title("Comparative Overall Best Fitness (Multi-Processor)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_MP_DIR, "comparative_fitness_mp.png")) # Updated path
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved comparative fitness plot to {IMAGES_MP_DIR}/comparative_fitness_mp.png")

    plt.figure(figsize=(15, 8))
    plot_df_time = results_df.dropna(subset=['Mean Exec Time (s)'])
    plt.bar(plot_df_time["Algorithm"], plot_df_time["Mean Exec Time (s)"], color="lightcoral")
    plt.xlabel("Algorithm Configuration")
    plt.ylabel("Mean Execution Time per Run (s)")
    plt.title("Comparative Mean Execution Time (Multi-Processor)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_MP_DIR, "comparative_times_mp.png")) # Updated path
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved comparative execution time plot to {IMAGES_MP_DIR}/comparative_times_mp.png")

    script_end_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Multi-Processor Script execution finished. Total time: {script_end_time - script_start_time:.2f} seconds.")

