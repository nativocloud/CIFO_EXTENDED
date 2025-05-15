# Single-Processor Phase Report - Metaheuristic Algorithm Optimization for Team Selection

**CIFO Team**

**Date:** May 15, 2025

## 1. Introduction

This document details the first phase of the project developed by the CIFO team, focusing on the implementation and analysis of metaheuristic algorithms for the sports team selection optimization problem in a single-processor context. The main objective of this phase was to establish a solid foundation by implementing different algorithms and operators, and to identify their performance and characteristics before exploring more advanced optimizations and parallelization.

The problem consists of forming a set of teams from an available pool of players, respecting a set of constraints (budget, number of players per team, positional requirements) and aiming to optimize a certain criterion (in this case, minimizing the variance of the average skill among the formed teams).

This report will describe the initial code architecture, the solution representation, the chosen data structures, the implemented algorithms (Hill Climbing, Simulated Annealing, and Genetic Algorithms with various operators), the rationale behind the design choices, and the initial performance results.

## 2. Single-Processor Code Architecture (Initial Version)

The code architecture for the single-processor phase was designed modularly to facilitate the implementation, testing, and replacement of different components of the metaheuristic algorithms. The main Python files involved in this architecture are:

*   `solution.py`: Defines the representation of a solution to the problem (an assignment of players to teams) and the methods to evaluate its validity and quality (fitness). It also includes methods for generating neighboring solutions or modifying existing solutions.
*   `evolution.py`: Contains the implementations of the main metaheuristic algorithms: Hill Climbing, Simulated Annealing, and the generic Genetic Algorithm.
*   `operators.py`: Implements the different genetic operators used by the Genetic Algorithm, such as selection, crossover, and mutation operators.
*   `main_script_sp.py`: The main script that orchestrates the execution of the algorithms, loads player data, configures algorithm parameters, and collects/presents the results.

### 2.1. Solution Representation and Data Structures

The rationale behind the choice of solution representation and data structures was simplicity and efficiency for the required operations.

*   **Solution Representation (`LeagueSolution` in `solution.py`):** A solution is represented as a list (or, later, a NumPy array) where the index corresponds to a player's ID and the value at that index corresponds to the ID of the team to which that player has been assigned. For example, `assignment[player_id] = team_id`.
    *   **Rationale:** This representation is straightforward, easy to manipulate, and allows quick access to any player's team. It is also compact.

*   **Player Data:** Player data (ID, name, position, salary, skill) is initially loaded from a CSV file into a convenient data structure (such as a list of dictionaries or a Pandas DataFrame) in the main script. For internal operations within the solution and algorithm classes, this data is often passed or converted to formats more optimized for calculation (like NumPy arrays for salaries, skills, and numerical positions, as was done in the vectorization phase).
    *   **Rationale:** CSV is a common and easy-to-use format for data input. Internally, conversion to NumPy arrays (especially after vectorization) aims to speed up numerical calculations.

*   **Team Structure:** Implicitly, teams are collections of players. The solution representation allows for the easy reconstruction of each team's composition by filtering players based on the `team_id` assigned to them.

### 2.2. Implemented Algorithms (Initial, Non-Optimized Version)

Three main types of metaheuristic algorithms were selected for this initial phase, due to their popularity and different approaches to exploring the solution space:

1.  **Hill Climbing (HC):**
    *   **Description:** An iterative local search algorithm that starts with an arbitrary solution and tries to find a better solution by moving to a neighbor with better fitness. It terminates when it reaches a local optimum, where no neighbor has better fitness.
    *   **Selection Rationale:** It is one of the simplest optimization algorithms to implement and understand. It serves as a good baseline for comparison with more complex algorithms. It is fast but prone to getting stuck in local optima.

2.  **Simulated Annealing (SA):**
    *   **Description:** A probabilistic optimization technique inspired by the annealing process in metallurgy. It allows moves to worse solutions with a certain probability, which decreases as the algorithm progresses (the "temperature" drops). This helps to escape local optima.
    *   **Selection Rationale:** It offers better global exploration capability than Hill Climbing, with the ability to avoid local optima. Its effectiveness depends on the correct parameterization of the cooling schedule.

3.  **Genetic Algorithms (GAs):**
    *   **Description:** Search algorithms inspired by biological evolution. They maintain a population of candidate solutions that evolve over generations through genetic operators like selection, crossover (recombination), and mutation.
    *   **Selection Rationale:** They are powerful for complex optimization problems and can effectively explore large solution spaces. Their population-based nature allows for maintaining diversity and exploring multiple regions of the search space simultaneously.

### 2.3. Selection of Operators for Testing (Genetic Algorithms)

For the Genetic Algorithms, a variety of operators were implemented to allow testing of different evolutionary strategies:

*   **Selection Operators:** Determine which individuals from the current population are chosen to reproduce.
    *   `selection_tournament`: Tournament selection. Several individuals are randomly chosen from the population, and the best among them is selected.
        *   **Rationale:** Simple to implement, efficient, and allows control over selective pressure through tournament size.
    *   `selection_roulette_wheel`: Roulette wheel selection. An individual's probability of being selected is proportional to its fitness.
        *   **Rationale:** A classic method that gives all individuals a chance but favors the fittest.
    *   `selection_ranking`: Rank-based selection. Individuals are sorted by their fitness, and selection probability is based on their rank, not the absolute fitness value.
        *   **Rationale:** Avoids premature convergence problems that can occur with roulette wheel if there are individuals with fitness much higher than the rest.
    *   `selection_boltzmann` (implemented, but may require temperature adjustment to be effective):
        *   **Rationale:** Inspired by Boltzmann exploration, adjusts selective pressure over time.

*   **Crossover Operators:** Combine the genetic material of two parents to create one or more offspring.
    *   `crossover_one_point`: One-point crossover. A cut point is randomly chosen, and segments of the parents are swapped to form offspring.
    *   `crossover_uniform`: Uniform crossover. For each gene (position in the assignment array), it is randomly decided from which parent the offspring will inherit that gene.
    *   `_prefer_valid` versions: Variants of these operators were also explored that attempt to generate valid offspring more directly or have mechanisms to repair invalid offspring (although the main validation occurs after offspring generation).
        *   **Rationale for Variety:** Different crossover operators explore the solution space in different ways. One-point crossover tends to preserve blocks of genes, while uniform crossover promotes greater mixing.

*   **Mutation Operators:** Introduce small random changes in individuals to maintain genetic diversity and allow exploration of new areas of the solution space.
    *   `mutation_random_player_team_change`: Randomly changes the team of a randomly selected player.
    *   `mutation_swap_players_between_teams`: Swaps two players between two different teams.
    *   `mutation_swap_players_different_teams_specific_roles` (more complex, may have been simplified or focused on more generic swaps initially): Attempts to swap players of specific roles between teams.
        *   **Rationale for Variety:** Different types of mutation allow for different types of "jumps" in the solution space. Simple mutations are less disruptive, while more complex mutations can allow escaping local optima more easily but may also be harder to ensure they quickly result in valid solutions.

The CIFO team's initial selection of these algorithms and operators aimed to cover a spectrum of metaheuristic approaches, from the simplest (HC) to the most complex and population-based (GAs with multiple operators), to evaluate their suitability and performance on the specific team formation problem.

## 3. Results of Initial Optimizations (Single Processor)

After the initial implementation of the algorithms, the CIFO team proceeded to a profiling and optimization phase of the single-processor code. The goal was to identify and mitigate major performance bottlenecks before considering parallelization. The two main optimization areas focused on were reducing the use of `deepcopy` and vectorizing critical functions.

### 3.1. Initial Profiling and Bottleneck Identification

The execution of the `main_script_sp_timing_estimate.py` script (configured for 1 run per algorithm and 1 generation for Genetic Algorithms) revealed that the total execution time was approximately **85.66 seconds**. Profiling analysis with `cProfile` highlighted the following as the biggest time consumers:

*   **Simulated Annealing (SA):** Was the slowest individual algorithm, consuming about **73.17 seconds** (approximately 85% of the total script time).
*   **`copy.deepcopy`:** This operation was identified as the main cause of slowness, especially within Simulated Annealing. It was called over 23 million times, totaling **53.49 seconds** of execution time.
*   **Solution Functions:** The functions `solution.py:178(get_random_neighbor)` (used by SA), `solution.py:122(fitness)`, and `solution.py:85(is_valid)` also consumed significant portions of time, mainly due to the high number of calls in SA and, in the case of `get_random_neighbor`, its internal use of `deepcopy`.

### 3.2. Optimization of `deepcopy` Usage in Simulated Annealing

Given the massive impact of `deepcopy` on SA's performance, the first optimization implemented by the CIFO team aimed to reduce its usage.

*   **Implemented Change:** In the `simulated_annealing` function (in `evolution.py`), calls to `deepcopy(neighbor_solution)` to update the `current_solution` (both when a better solution was found and when a worse one was probabilistically accepted) were replaced with direct assignments (`current_solution = neighbor_solution`).
*   **Rationale:** This change was considered safe because the `get_random_neighbor` function (in `solution.py`) already created and returned a *new solution object* (neighbor) that was an independent instance. The call to `deepcopy` to update the global `best_solution` was maintained to ensure its integrity.
*   **Impact:** This optimization resulted in a very significant performance improvement:
    *   The total script time was reduced from 85.66 seconds to **41.42 seconds** (an improvement of approximately 51.7%).
    *   The specific time for the `simulated_annealing` function decreased from 73.17 seconds to **28.91 seconds** (an improvement of approximately 60.5%).
    *   The total number of calls to `deepcopy` in the script decreased drastically.
    *   The quality of the solution (fitness) achieved by the algorithms remained the same.

### 3.3. Vectorization of `is_valid()` and `fitness()` Functions

The second optimization attempt focused on vectorizing the `is_valid()` and `fitness()` functions in `solution.py`, using NumPy to replace Python loops with more efficient array operations.

*   **Implemented Changes:**
    1.  **Data Preprocessing:** In the `LeagueSolution` class constructor, player data (salaries, skills, positions) were converted and stored as NumPy arrays.
    2.  **Rewriting `is_valid()`:** Checks for team size, budget, and positional requirements were rewritten to use NumPy operations like `np.bincount`, boolean masks, and `np.array_equal`.
    3.  **Rewriting `fitness()`:** The calculation of average skill per team was optimized using `np.bincount` with the `weights` argument, and the standard deviation was calculated with `np.std`.
*   **Rationale:** The intention was to speed up these frequently called functions by leveraging NumPy's efficiency for numerical operations on arrays.
*   **Impact:** After implementing vectorization and the necessary corrections in `evolution.py`, `operators.py`, and `main_script_sp_timing_estimate.py` for compatibility, the full script execution (with `deepcopy` optimizations in SA and vectorization) resulted in a total time of approximately **40.96 seconds**.
    *   Comparing with the time after `deepcopy` optimization (41.42s), vectorization, in the current context of calls (one solution at a time for these functions in most cases), did not bring an additional significant improvement in overall execution time. The `is_valid` and `fitness` functions themselves became faster in isolation, but their contribution to the total time had already been reduced by the decreased number of calls or the predominance of other factors in SA.
    *   However, the vectorized implementation is considered more robust and potentially more scalable if the number of players or teams were to increase significantly, or if the algorithms were adapted to evaluate batches of solutions.

These results from the single-processor phase provided valuable insights into the algorithms' behavior and the effectiveness of optimization strategies, paving the way for the exploration of multiprocessing.

## 4. Conclusions of the Single-Processor Phase and Next Steps

The single-processor implementation and optimization phase, conducted by the CIFO team, was crucial for establishing a solid baseline for the team selection problem. The initial implementation of Hill Climbing, Simulated Annealing, and Genetic Algorithms with various operators allowed for a thorough understanding of their behavior and inherent computational challenges.

Profiling proved to be an indispensable tool, identifying the excessive use of `deepcopy` in Simulated Annealing as the main performance bottleneck. Targeted optimization of this issue resulted in a significant reduction in execution time (approximately 51.7% in total script time), without compromising the quality of the solutions found. The subsequent vectorization of the `is_valid()` and `fitness()` functions, although not providing substantial additional time gains in this specific calling context, modernized the codebase and prepared it for potentially more demanding scenarios.
