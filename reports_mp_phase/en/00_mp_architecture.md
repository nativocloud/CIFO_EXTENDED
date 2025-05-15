# Multiprocessing (MP) Phase Architecture Document

**Version:** 1.0
**Date:** May 15, 2025

## 1. Introduction

This document describes the software architecture implemented for the multiprocessing (MP) phase of the CIFO EXTENDED project. The main objective of this phase was to extend the single-processor (SP) phase architecture to allow parallel execution of multiple instances of metaheuristic algorithms, aiming to obtain statistically more robust results and accelerate the experimentation process.

## 2. MP Architecture Overview

The MP phase architecture builds upon the modules developed in the SP phase (`solution.py`, `evolution.py`, `operators.py`), introducing a new main script (`main_script_mp.py` and its variants like `main_script_mp_30_runs.py`, `main_script_mp_param_var.py`, `main_script_mp_final_param_var.py`) that orchestrates parallel executions using Python's `multiprocessing` module.

The main additional or modified components in the MP architecture are:

*   **Multiprocessing Main Scripts (e.g., `main_script_mp.py`):** Responsible for configuring and managing the parallel execution of multiple runs of one or more algorithms. They use a `Pool` of processes to distribute the work.
*   **Worker Execution Function:** A wrapper function that encapsulates the logic of a single algorithm run, making it suitable for execution by a process in a `Pool`.
*   **Results Aggregation:** Mechanisms to collect and aggregate results (fitness, execution time, etc.) from all parallel runs for subsequent analysis.

## 3. MP Component Details

### 3.1. Multiprocessing Main Scripts

These scripts are central to the MP phase.

*   **Example (`main_script_mp.py`):**
    *   **Algorithm and Run Configuration:** Defines which algorithms and with what parameters will be executed, and how many times each should be repeated (e.g., 5 runs, 30 runs).
    *   **Process Pool Creation:** Uses `multiprocessing.Pool()` to create a set of worker processes, usually corresponding to the number of available CPU cores or a configurable value.
    *   **Task Preparation:** For each required algorithm run, parameters are packaged (e.g., into a tuple or dictionary) to be passed to the worker function.
    *   **Task Distribution:** Uses methods like `pool.map()` or `pool.apply_async()` to submit tasks to the `Pool` processes.
    *   **Results Collection:** Waits for all tasks to complete and collects the results returned by each worker process.
    *   **Aggregated Results Analysis and Presentation:** Calculates statistical metrics (mean, standard deviation, overall best) from individual results and presents them or saves them to a summary file (e.g., CSV).

### 3.2. Worker Execution Function

This function is executed by each process in the `Pool`.

*   **Typical Structure:**
    ```python
    def run_algorithm_worker(params):
        # Unpack parameters (algorithm name, configuration, player data, etc.)
        algorithm_name, config, player_data, num_teams, ... = params

        # Generate initial solution (if necessary)
        initial_solution = LeagueSolution(player_data, ...)

        # Execute the algorithm (e.g., hill_climbing, genetic_algorithm)
        start_time = time.time()
        if algorithm_name == 'HC':
            best_solution_data = evolution.hill_climbing(initial_solution, ...)
        elif algorithm_name == 'GA':
            best_solution_data = evolution.genetic_algorithm(player_data, ...)
        # ... other algorithms
        end_time = time.time()

        execution_time = end_time - start_time
        fitness = best_solution_data['best_fitness'] # or similar

        # Return relevant results (fitness, time, perhaps the best solution)
        return (fitness, execution_time, config_name) # Example
    ```
*   **Isolation:** Each call to this function is executed in a separate process, ensuring that runs are independent and do not share memory problematically (unless explicitly managed with `multiprocessing` mechanisms).

### 3.3. SP Phase Modules (`solution.py`, `evolution.py`, `operators.py`)

These modules are largely reused without significant modifications to their internal logic. The main change is how they are invoked (via the worker function in separate processes).

*   **Considerations:**
    *   **Global State:** Care must be taken that these modules do not rely on global state that could be modified concurrently and unsafely between processes. In the project's practice, explicitly passing all necessary data (like `player_data`) to functions helps mitigate this.
    *   **Serialization:** Objects passed between processes (like `initial_solution` or parameters) must be serializable (picklable) by Python.

## 4. Data Flow in MP Phase

1.  The MP main script (e.g., `main_script_mp.py`) loads player data and defines algorithm configurations and the number of repetitions.
2.  It creates a list of parameter tuples, where each tuple contains everything needed for a single algorithm run.
3.  A `multiprocessing.Pool` is created.
4.  Parameter tuples are passed to the worker function using `pool.map()` (or similar).
5.  Each process in the `Pool` executes the worker function with a set of parameters:
    a.  The worker function invokes the appropriate algorithm from `evolution.py`.
    b.  The algorithm uses `solution.py` and `operators.py` as in the SP phase.
    c.  The worker function returns the result of its run (e.g., fitness, time).
6.  The main script collects the list of results from all runs.
7.  Results are aggregated (calculating means, standard deviations, etc.) and saved or presented.

## 5. Design Decisions and Rationale for MP

*   **Use of `multiprocessing.Pool`:** Provides a convenient, high-level way to manage a set of worker processes and distribute tasks, abstracting many low-level details of process management.
*   **Worker Wrapper Function:** Isolates the logic of a single run, making the code cleaner and facilitating parallelization.
*   **Explicit Data Passing:** Minimizes issues with shared state, as each worker process receives its own copy of necessary data (or references to immutable data).
*   **SP Code Reuse:** The MP architecture was built on the solid foundation of the SP phase, reusing algorithm and solution modules, which accelerated development.
*   **CSV Summary Files:** Using CSV files to save aggregated results facilitates later analysis and generation of charts/reports.

## 6. Challenges and Considerations in MP Architecture

*   **Multiprocessing Overhead:** Process creation and inter-process communication (data serialization/deserialization) introduce some overhead. For very short tasks, this overhead can diminish the gains from parallelization.
*   **Memory Consumption:** Each process may have its own copy of some data, which can increase total memory consumption, especially for large datasets or large populations in GAs.
*   **Debugging:** Debugging multiprocess applications can be more complex than sequential applications.
*   **Optimal Number of Processes:** Determining the ideal number of processes in the `Pool` (usually related to the number of CPU cores) may require some experimentation to optimize performance.

## 7. Evolution of MP Architecture in the Project

Throughout the project, the MP architecture evolved through different main scripts to accommodate various experimentation phases:

*   `main_script_mp.py`: For the initial 5 runs.
*   `main_script_mp_30_runs.py`: For the 30 runs of promising algorithms.
*   `main_script_mp_param_var.py`: For testing GA parameter variations (with 5 runs per variation).
*   `main_script_mp_final_param_var.py`: For the final analysis with 30 runs of the most promising parameter variations.

The core multiprocessing logic (use of `Pool`, worker function) remained consistent, with variations mainly in how algorithm configurations and experimentation loops were defined.

This architecture allowed for a much broader and statistically significant exploration of the solution space and algorithm behavior compared to the single-processor phase.

