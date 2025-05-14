# Single-Processor Version Architecture Document (`architecture_sp.md`)

## 1. Introduction

This document outlines the software architecture of the single-processor version of the Sports League Assignment optimization project. It details the key components, their responsibilities, interactions, and the overall design philosophy. The primary goal of this version is to implement and evaluate Hill Climbing (HC), Simulated Annealing (SA), and Genetic Algorithm (GA) approaches for solving the constrained optimization problem.

## 2. Overall Architecture

The single-processor version is structured around several key Python modules:

-   **`solution.py`**: Defines the core classes for representing a solution to the league assignment problem. This includes the base solution representation and specialized versions for local search algorithms.
-   **`operators.py`**: Contains functions for genetic algorithm operators, including various selection, crossover, and mutation strategies.
-   **`evolution.py`**: Implements the main logic for the evolutionary (Genetic Algorithm) and local search (Hill Climbing, Simulated Annealing) algorithms. It utilizes the classes from `solution.py` and functions from `operators.py`.
-   **`main_script_sp.py`**: Serves as the main execution script. It handles data loading, orchestrates the execution of the different algorithms for a specified number of runs, collects results, generates plots, and manages logging.

The general flow is initiated by `main_script_sp.py`, which calls functions in `evolution.py` to run the algorithms. These algorithm functions, in turn, create and manipulate instances of solution classes defined in `solution.py` and, in the case of GAs, use operator functions from `operators.py`.

## 3. Key Modules and Components

### 3.1. `solution.py` - Solution Representation

This module is central to defining what a solution looks like and how its validity and quality are assessed.

**Classes:**

1.  **`LeagueSolution` (Base Class)**
    *   **Responsibilities**: 
        *   Represents a potential assignment of players to teams.
        *   Initializes a solution, attempting to generate a valid assignment using a constructive heuristic (`_random_valid_assignment_constructive`) if no specific assignment is provided.
        *   Validates a given assignment against all problem constraints (team size, positional requirements, budget) via the `is_valid()` method.
        *   Calculates the fitness of a valid solution (standard deviation of average team skills) via the `fitness()` method. Invalid solutions are assigned `float('inf')` fitness.
        *   Provides a `copy()` method for creating deep copies of solution instances.
    *   **Key Attributes**: `assignment` (list representing player-to-team mapping), `num_teams`, `team_size`, `max_budget`, `players_data_ref` (reference to the full player dataset).

2.  **`LeagueHillClimbingSolution(LeagueSolution)` (Derived Class)**
    *   **Inheritance**: Inherits from `LeagueSolution`.
    *   **Responsibilities**: 
        *   Extends `LeagueSolution` specifically for the Hill Climbing algorithm.
        *   Provides a `get_neighbors()` method, which generates a list of neighboring solutions by performing pairwise swaps of player assignments. Only valid neighbors are returned.
    *   **Overrides**: `copy()` method to ensure instances of `LeagueHillClimbingSolution` are created.

3.  **`LeagueSASolution(LeagueSolution)` (Derived Class)**
    *   **Inheritance**: Inherits from `LeagueSolution`.
    *   **Responsibilities**: 
        *   Extends `LeagueSolution` specifically for the Simulated Annealing algorithm.
        *   Provides a `get_random_neighbor()` method, which generates a single random neighboring solution by swapping two player assignments. It attempts to find a valid neighbor within a set number of tries; otherwise, it returns a copy of the current solution.
    *   **Overrides**: `copy()` method to ensure instances of `LeagueSASolution` are created.

**Class Diagram (Textual Representation):**

```
[LeagueSolution]
  - assignment
  - num_teams
  - team_size
  - max_budget
  - players_data_ref
  + __init__(...)
  + _random_valid_assignment_constructive(...)
  + is_valid(...)
  + fitness(...)
  + copy()

    ^
    |
    +--- [LeagueHillClimbingSolution]
    |      + get_neighbors(...)
    |      + copy() // Override
    |
    +--- [LeagueSASolution]
           + get_random_neighbor(...)
           + copy() // Override
```

### 3.2. `operators.py` - Genetic Algorithm Operators

This module provides a collection of functions that implement the core genetic operators used by the Genetic Algorithm.

**Function Categories:**

*   **Selection Operators**: (e.g., `selection_ranking`, `selection_tournament_variable_k`, `selection_boltzmann`)
    *   Responsible for choosing parent solutions from the current population for reproduction.
*   **Crossover Operators**: (e.g., `crossover_one_point_prefer_valid`, `crossover_uniform_prefer_valid`)
    *   Responsible for combining genetic material from two parent solutions to create offspring. These versions prioritize creating valid offspring.
*   **Mutation Operators**: (e.g., `mutate_swap_constrained`, `mutate_targeted_player_exchange`, `mutate_shuffle_within_team_constrained`)
    *   Responsible for introducing small, random changes into an individual solution to maintain diversity and explore new areas of the search space. These versions are designed to respect problem constraints or attempt to repair invalid solutions.

These functions typically take one or more `LeagueSolution` objects as input and return one or more `LeagueSolution` objects.

### 3.3. `evolution.py` - Algorithm Implementation

This module contains the procedural logic for the implemented optimization algorithms.

**Key Functions:**

1.  **`generate_population(players_data_full, size, ...)`**
    *   **Responsibilities**: Creates an initial population of `LeagueSolution` objects. It repeatedly calls the `LeagueSolution` constructor (which uses the constructive heuristic) and validates each new individual, aiming to create a population of the specified `size` consisting of valid solutions.

2.  **`genetic_algorithm(players_data, population_size, generations, ...)`**
    *   **Responsibilities**: Implements the main loop of the Genetic Algorithm.
        *   Initializes a population using `generate_population()`.
        *   Iterates for a specified number of `generations`.
        *   In each generation: applies elitism, performs selection using a chosen `selection_operator_func`, applies crossover using `crossover_operator_func`, and applies mutation using `mutation_operator_func`.
        *   Ensures that new individuals are validated.
        *   Tracks the best solution found and its fitness history.
    *   **Dependencies**: `LeagueSolution`, `generate_population`, and various functions from `operators.py`.

3.  **`hill_climbing(initial_solution, players_data, max_iterations, ...)`**
    *   **Responsibilities**: Implements the Hill Climbing algorithm.
        *   Starts with an `initial_solution` (an instance of `LeagueHillClimbingSolution`).
        *   Iteratively explores neighbors using `initial_solution.get_neighbors()`.
        *   Moves to the best neighbor if it offers improved fitness.
        *   Terminates if no better neighbor is found or `max_iterations` is reached.
    *   **Dependencies**: `LeagueHillClimbingSolution`.

4.  **`simulated_annealing(initial_solution, players_data, initial_temp, ...)`**
    *   **Responsibilities**: Implements the Simulated Annealing algorithm.
        *   Starts with an `initial_solution` (an instance of `LeagueSASolution`).
        *   Iteratively explores random neighbors using `initial_solution.get_random_neighbor()`.
        *   Accepts better solutions unconditionally and worse solutions with a probability dependent on the current temperature and the fitness difference.
        *   Gradually reduces the temperature according to a cooling schedule (`alpha`).
        *   Terminates when the temperature reaches `final_temp`.
    *   **Dependencies**: `LeagueSASolution`.

### 3.4. `main_script_sp.py` - Orchestration and Experimentation

This script is the entry point for running experiments with the single-processor version.

**Responsibilities:**

*   **Setup**: Imports necessary modules and loads player data from `players.csv`.
*   **Parameter Definition**: Defines global parameters like `NUM_TEAMS`, `TEAM_SIZE`, `MAX_BUDGET`, and `NUM_RUNS` (number of independent runs for each stochastic algorithm).
*   **Algorithm Execution**: 
    *   Iterates `NUM_RUNS` times for Hill Climbing, Simulated Annealing, and each configured Genetic Algorithm.
    *   For each run: 
        *   Initializes the appropriate solution object (e.g., `LeagueHillClimbingSolution`, `LeagueSASolution`, or `LeagueSolution` for GA population).
        *   Includes a retry mechanism for generating valid initial solutions.
        *   Calls the respective algorithm function from `evolution.py`.
        *   Records fitness values, execution times, and convergence history.
*   **Results Aggregation**: Calculates and prints summary statistics (mean fitness, standard deviation of fitness, mean execution time) for each algorithm/configuration across all runs.
*   **Plotting**: Generates and saves convergence plots for the best run of each algorithm and comparative plots for fitness and execution times.
*   **Logging**: Prints detailed logs to the console, including timestamps, progress updates, and final results for each section and run.

## 4. Data Flow

1.  Player data is loaded from `players.csv` into a list of dictionaries by `main_script_sp.py`.
2.  This `players_data` is passed to `LeagueSolution` (and its derived classes) constructors and to algorithm functions in `evolution.py`.
3.  Solution objects (`LeagueSolution`, etc.) store a reference to this `players_data` for validation and fitness calculation.
4.  Algorithms in `evolution.py` generate/manipulate solution assignments and use the `is_valid()` and `fitness()` methods of the solution objects.
5.  Results (best solutions, fitness values, histories, execution times) are returned from `evolution.py` functions to `main_script_sp.py`.
6.  `main_script_sp.py` aggregates these results and generates plots and summary statistics.

## 5. Limitations in Diagram Generation

Generating complex, visually rich UML class diagrams or sequence diagrams automatically is beyond the current capabilities of this system. The architecture and class relationships are therefore described textually. The textual class diagram provided for `solution.py` offers a simplified structural overview.

## 6. Conclusion

The single-processor architecture is designed modularly, separating concerns of solution representation, algorithmic logic, genetic operators, and experimental orchestration. This structure facilitates testing, modification, and extension of individual components. The use of inheritance in `solution.py` allows for specialized behavior for different local search algorithms while maintaining a common interface for solution validation and fitness evaluation.

