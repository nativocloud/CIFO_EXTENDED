# Single-Processor (SP) Phase Architecture Document

**Version:** 1.0
**Date:** May 15, 2025

## 1. Introduction

This document describes the software architecture implemented for the single-processor (SP) phase of the CIFO EXTENDED project. The objective of this phase was to establish a solid foundation for the implementation and testing of metaheuristic algorithms (Hill Climbing, Simulated Annealing, and Genetic Algorithms) for the sports team formation optimization problem.

## 2. Architecture Overview

The SP phase architecture was designed modularly, aiming to separate the different system responsibilities into distinct components to facilitate development, maintenance, and testability. The interaction between modules is orchestrated by a main script.

The main components of the architecture are:

*   **Solution Module (`solution.py`):** Responsible for representing a candidate solution, calculating its quality (fitness), and validating problem constraints.
*   **Evolution Module (`evolution.py`):** Contains the core logic of the implemented metaheuristic algorithms (Hill Climbing, Simulated Annealing, Genetic Algorithm).
*   **Operators Module (`operators.py`):** Provides the specific genetic operators (selection, crossover, mutation) used by the Genetic Algorithm.
*   **Main Script (`main_script_sp.py`):** Application entry point, responsible for loading input data, configuring algorithm parameters, instantiating and executing algorithms, and presenting/saving results.
*   **Input Data:** CSV file containing player data (skill, position, salary, etc.).

## 3. Component Details

### 3.1. Solution Module (`solution.py`)

This module encapsulates all logic related to an individual solution for the team formation problem.

*   **`LeagueSolution` Class:**
    *   **Internal Representation:** A solution is represented as an assignment of players to teams. Initially, a list where the index represents the player ID and the value the team ID. Later optimizations might involve using NumPy arrays for efficiency.
    *   **Main Functionalities:**
        *   `__init__(self, player_data, num_teams, team_size, budget_per_team, initial_assignment=None)`: Constructor that initializes a solution, potentially generating a random valid assignment or using a provided one.
        *   `calculate_fitness(self)`: Calculates the objective function, typically minimizing the variance of the average skill among teams.
        *   `is_valid(self)`: Checks if the current solution satisfies all problem constraints (team budget, number of players per team, positional constraints).
        *   `get_random_neighbor(self)`: Generates a neighboring solution through a small modification to the current solution (e.g., moving a player to a different team).
        *   Other auxiliary methods for solution manipulation and evaluation.

### 3.2. Evolution Module (`evolution.py`)

This module contains the implementations of the optimization algorithms.

*   **`hill_climbing(initial_solution, max_iterations, player_data, num_teams, team_size, budget_per_team)` function:**
    *   Implements the Hill Climbing algorithm. Starts with an initial solution and iteratively tries to find better neighbors until a local maximum is reached or the maximum number of iterations is achieved.
*   **`simulated_annealing(initial_solution, max_iterations, initial_temperature, cooling_rate, player_data, num_teams, team_size, budget_per_team)` function:**
    *   Implements the Simulated Annealing algorithm. Allows moves to worse solutions with a probability that decreases over time (temperature), helping to escape local optima.
*   **`genetic_algorithm(player_data, num_teams, team_size, budget_per_team, population_size, num_generations, tournament_size, crossover_rate, mutation_rate, elite_size, selection_operator, crossover_operator, mutation_operator)` function:**
    *   Implements the Genetic Algorithm. Maintains a population of solutions that evolves through selection, crossover, and mutation operators over several generations.

### 3.3. Operators Module (`operators.py`)

This module provides functions for genetic operators.

*   **Selection Operators:**
    *   `selection_tournament(population, fitnesses, tournament_size)`
    *   `selection_roulette_wheel(population, fitnesses)`
    *   `selection_ranking(population, fitnesses)`
    *   `selection_boltzmann(population, fitnesses, temperature)`
*   **Crossover Operators:**
    *   `crossover_one_point(parent1, parent2, player_data, num_teams, team_size, budget_per_team)`
    *   `crossover_uniform(parent1, parent2, player_data, num_teams, team_size, budget_per_team)`
    *   `_prefer_valid` versions that attempt to generate valid offspring directly.
*   **Mutation Operators:**
    *   `mutation_random_player_team_change(solution, player_data, num_teams, team_size, budget_per_team)`
    *   `mutation_swap_players_between_teams(solution, player_data, num_teams, team_size, budget_per_team)`

### 3.4. Main Script (`main_script_sp.py`)

This script is the orchestrator of the SP phase.

*   **Data Loading:** Reads player data from the CSV file (e.g., `players.csv`).
*   **Configuration:** Defines parameters for each algorithm (e.g., number of iterations, population size, mutation/crossover rates, etc.).
*   **Algorithm Execution:**
    *   Instantiates `LeagueSolution` to generate initial solutions.
    *   Calls algorithm functions from the `evolution.py` module, passing configured parameters and operators.
*   **Results Collection and Presentation:**
    *   Saves the best solution found, its fitness, and execution time.
    *   Prints results to the console or saves them to log/summary files.
*   **Profiling (Optional):** May include code for profiling (e.g., `cProfile`) to identify performance bottlenecks.

## 4. Data Flow

1.  `main_script_sp.py` loads player data.
2.  For each algorithm to be executed:
    a.  An initial solution is created (or a population of solutions for GAs) using `LeagueSolution`.
    b.  `main_script_sp.py` invokes the corresponding algorithm function in `evolution.py`.
    c.  Algorithms in `evolution.py` use `LeagueSolution` to evaluate fitness, validate solutions, and generate neighbors/offspring.
    d.  The GA uses operators from `operators.py` for selection, crossover, and mutation.
    e.  The best solution and performance metrics are returned to `main_script_sp.py`.
3.  `main_script_sp.py` presents or saves the results.

## 5. Design Decisions and Rationale

*   **Modularity:** Separation into `solution.py`, `evolution.py`, and `operators.py` allows each component to be developed and tested more independently. It also facilitates adding new algorithms or operators.
*   **Solution Abstraction:** The `LeagueSolution` class abstracts the details of solution representation and its operations, making the algorithm code cleaner.
*   **Configurability:** `main_script_sp.py` centralizes algorithm configuration, allowing easy experimentation with different parameters.
*   **Focus on Initial Clarity:** The first version of the architecture prioritized clarity and functional correctness over premature optimization. Optimizations (like vectorization and `deepcopy` reduction) were applied later, after profiling.

## 6. SP Architecture Limitations

*   **Sequential Processing:** By definition, this architecture executes algorithms and their repetitions sequentially, which can be time-consuming for extensive analyses.
*   **Scalability:** Managing multiple executions and aggregating results for statistical analysis is done manually or with simple scripts, which does not scale well for a large number of experiments.

This architecture served as the foundation for subsequent project phases, where multiprocessing was introduced to address these limitations.

