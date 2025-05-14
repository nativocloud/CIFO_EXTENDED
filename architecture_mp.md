# Multi-Processor Version Architecture Document (`architecture_mp.md`)

## 1. Introduction

This document describes the software architecture of the multi-processor (MP) version of the Sports League Assignment optimization project. It focuses on how multiprocessing is integrated to parallelize the execution of independent algorithm runs, thereby reducing the overall experimentation time. This version builds upon the core logic defined in `solution.py`, `operators.py`, and `evolution.py`.

## 2. Overall Architecture and Parallelization Strategy

The multi-processor version (`main_script_mp.py`) retains the same core algorithmic components as the single-processor version but introduces a parallel execution layer using Python's `multiprocessing` module. The primary strategy is to parallelize the `NUM_RUNS` independent executions for each algorithm (Hill Climbing, Simulated Annealing) and each Genetic Algorithm configuration.

**Key Architectural Changes from Single-Processor Version:**

*   **Worker Functions**: Dedicated worker functions (`hc_worker`, `sa_worker`, `ga_worker`) are defined. Each worker function encapsulates the logic for a single run of a specific algorithm or GA configuration.
*   **`multiprocessing.Pool`**: A process pool is utilized to distribute the `NUM_RUNS` tasks (each corresponding to one call of a worker function) across available CPU cores.
*   **Argument Packaging**: Arguments for each worker function call are packaged into tuples.
*   **Results Collection**: Results from each worker process (fitness, execution time, best solution assignment, history) are collected from the pool and then aggregated to compute overall statistics.
*   **Global Data**: `players_data_global` and core parameters like `NUM_TEAMS_GLOBAL` are defined globally and passed as arguments to the worker functions. Since `players_data_global` is read-only within the workers, this approach is generally safe for multiprocessing.

## 3. Key Components in `main_script_mp.py`

### 3.1. Worker Functions

These functions are designed to be executed in separate processes.

1.  **`hc_worker(args_tuple)`**
    *   **Responsibilities**: Executes a single run of the Hill Climbing algorithm.
        *   Unpacks arguments: `run_id`, `players_data`, `num_teams`, `team_size`, `max_budget`, `max_iterations`, `verbose_flag`.
        *   Initializes a `LeagueHillClimbingSolution` instance, with retries for validity.
        *   Calls the `hill_climbing` function from `evolution.py`.
        *   Returns a dictionary containing `run_id`, `fitness`, `exec_time`, `solution_assignment` (the player assignment list, not the full object, for easier pickling), and `history`.

2.  **`sa_worker(args_tuple)`**
    *   **Responsibilities**: Executes a single run of the Simulated Annealing algorithm.
        *   Unpacks arguments: `run_id`, `players_data`, `num_teams`, `team_size`, `max_budget`, `sa_params_dict`, `verbose_flag`.
        *   Initializes a `LeagueSASolution` instance, with retries for validity.
        *   Calls the `simulated_annealing` function from `evolution.py`.
        *   Returns a dictionary similar to `hc_worker`.

3.  **`ga_worker(args_tuple)`**
    *   **Responsibilities**: Executes a single run of a specific Genetic Algorithm configuration.
        *   Unpacks arguments: `run_id`, `players_data`, `num_teams`, `team_size`, `max_budget`, `ga_config_dict` (containing operator functions and specific GA params like tournament_k), `ga_params_dict` (general GA params like population_size), `verbose_flag`.
        *   Calls the `genetic_algorithm` function from `evolution.py`.
        *   Returns a dictionary similar to other workers, also including `config_name`.

### 3.2. Main Execution Block (`if __name__ == '__main__':`)

*   **Initialization**: 
    *   Sets up the output directory for graphs (`mp_graphs`).
    *   Determines the number of CPU cores to use for the process pool (`num_processes`).
    *   Loads global `players_data_global`.
*   **Algorithm Execution (Parallelized Sections)**: For each algorithm type (HC, SA, GA configurations):
    1.  **Argument Preparation**: Creates a list of argument tuples (`hc_args_list`, `sa_args_list`, `ga_args_list_for_config`), where each tuple contains all necessary parameters for one worker function call (one run).
    2.  **Process Pool Creation and Execution**: 
        *   A `multiprocessing.Pool(processes=num_processes)` is created.
        *   The `pool.map(worker_function, args_list)` method is used to distribute the tasks. This call blocks until all worker processes have completed their assigned runs.
    3.  **Results Aggregation**: 
        *   The list of dictionaries returned by `pool.map` is processed.
        *   Fitness values, execution times are extracted.
        *   The overall best solution (based on its assignment and fitness) and its history are identified from all parallel runs.
    4.  **Statistics and Plotting**: Calculates mean fitness, standard deviation, and mean execution time per run. Generates and saves convergence plots for the best overall run and updates the `all_results_summary`.
*   **Comparative Plots**: After all algorithms have run, generates comparative plots for fitness and execution times across all algorithms and configurations, similar to the single-processor version but using data from the parallel runs.
*   **Logging**: Provides console logs for script start/end, section start/end, worker pool launch/finish, and summary results.

## 4. Data Flow in Parallel Execution

1.  `players_data_global` and other global parameters are defined in the main process.
2.  For each set of `NUM_RUNS_GLOBAL` for an algorithm, a list of argument tuples is created. Each tuple includes a copy of or reference to `players_data_global` and other necessary parameters.
3.  The `multiprocessing.Pool.map()` function handles the distribution of these argument tuples to the worker functions running in separate processes. Python's `multiprocessing` module pickles (serializes) the arguments to send them to child processes.
4.  Each worker process unpickles its arguments, executes its assigned algorithm run, and performs computations using its local copy/reference of the data.
5.  Worker functions return a dictionary of results. These dictionaries are pickled by the child process and sent back to the main process.
6.  The main process unpickles the result dictionaries and aggregates them.

**Note on Solution Objects**: Worker functions return the `solution_assignment` (a list of integers) instead of the full `LeagueSolution` (or derived) object. This is a common practice in multiprocessing to simplify serialization (pickling) and reduce inter-process communication overhead, as complex objects can sometimes cause pickling issues. The main process can reconstruct a solution object if needed, but for plotting and statistics, the assignment and fitness are often sufficient.

## 5. Interaction with Core Modules

*   **`solution.py`**: Worker functions still create instances of `LeagueSolution`, `LeagueHillClimbingSolution`, and `LeagueSASolution` internally to manage and evaluate solutions for each run.
*   **`operators.py`**: The `ga_worker` passes the relevant operator functions (from `operators.py`) to the `genetic_algorithm` function in `evolution.py` as per the GA configuration being run.
*   **`evolution.py`**: The core algorithm logic within `genetic_algorithm`, `hill_climbing`, and `simulated_annealing` functions remains largely unchanged. They are called by the worker functions with the appropriate parameters for a single run.

## 6. Conclusion

The multi-processor architecture effectively parallelizes the independent runs of the optimization algorithms, leveraging multiple CPU cores to reduce the total execution time for comprehensive experiments. The design isolates the parallel execution logic within `main_script_mp.py` by using worker functions and a process pool, while reusing the existing core algorithmic and solution representation modules. This approach maintains modularity and allows for significant speedups for computationally intensive experimentation phases involving multiple stochastic runs.

