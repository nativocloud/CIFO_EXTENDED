# In-Depth Code Review: Single-Processor Version (`code_review_sp.md`)

## 1. Introduction

This document provides a detailed line-by-line review of the core Python scripts for the single-processor version of the Sports League Assignment project. The aim is to explain the logic, structure, and functionality of each key component to facilitate understanding, maintenance, and potential future development. The review covers `solution.py`, `evolution.py`, and `main_script_sp.py`.

## 2. Review of `solution.py`

This file defines the classes responsible for representing and managing solutions to the Sports League Assignment problem.

### 2.1. Imports

```python
import random
import numpy as np
from copy import deepcopy
```

*   **`random`**: Used for random operations, primarily in the constructive heuristic for initial solution generation and in selecting players or operations randomly.
*   **`numpy` (as `np`)**: Used for numerical operations, specifically `np.mean` and `np.std` for calculating average team skills and their standard deviation (the fitness function).
*   **`deepcopy` from `copy`**: Used to create independent copies of solution assignments or entire solution objects, crucial for preventing unintended modifications to parent solutions or individuals in a population during evolutionary processes or neighborhood generation.

### 2.2. Class `LeagueSolution`

This is the base class for representing a league assignment.

#### 2.2.1. `__init__(self, players_data_full, assignment=None, num_teams=5, team_size=7, max_budget=750)`

```python
class LeagueSolution:
    def __init__(self, players_data_full, assignment=None, num_teams=5, team_size=7, max_budget=750):
        self.num_teams = num_teams
        self.team_size = team_size
        self.max_budget = max_budget
        self.players_data_ref = players_data_full # Store a reference

        if assignment:
            self.assignment = assignment
        else:
            self.assignment = self._random_valid_assignment_constructive(players_data_full)
```

*   **Purpose**: Initializes a `LeagueSolution` object.
*   **Parameters**:
    *   `players_data_full`: A list of dictionaries, where each dictionary represents a player and their attributes. This is stored as a reference (`self.players_data_ref`) for use in validation and fitness calculations.
    *   `assignment` (optional): A pre-existing assignment list (player index to team ID). If `None`, a new assignment is generated.
    *   `num_teams`, `team_size`, `max_budget`: Core problem constraints, initialized with default values.
*   **Logic**: 
    *   Stores problem parameters (`num_teams`, `team_size`, `max_budget`) and a reference to the complete player data.
    *   If an `assignment` is provided, it's used directly.
    *   Otherwise, it calls `_random_valid_assignment_constructive()` to attempt to generate an initial valid assignment. The success of this heuristic is not guaranteed and is checked externally (e.g., during population generation).

#### 2.2.2. `_random_valid_assignment_constructive(self, players_data_full)`

```python
    def _random_valid_assignment_constructive(self, players_data_full):
        # ... (implementation details) ...
        available_players = [(i, p) for i, p in enumerate(players_data_full)]
        random.shuffle(available_players)
        # ... (loop through teams and roles to assign players) ...
        return final_assignment
```

*   **Purpose**: Attempts to create a valid player assignment heuristically.
*   **Logic**:
    1.  Creates a list of `(original_index, player_data)` tuples from `players_data_full` and shuffles it to introduce randomness.
    2.  Initializes `final_assignment` (list of team IDs for each player) with -1 (unassigned).
    3.  Defines `target_roles` (e.g., `{"GK": 1, "DEF": 2, ...}`).
    4.  Iterates through each `team_id` from 0 to `num_teams - 1`:
        *   Initializes `current_team_cost` and `current_team_roles` count.
        *   Iterates through each `role` and the `count_needed` for that role:
            *   Searches through `available_players` for a player matching the `role`, fitting the `max_budget` for the current team, and ensuring the team doesn't exceed `team_size`.
            *   If a suitable player is found: assigns the player to the current `team_id` in `final_assignment`, updates team cost and role counts, and removes the player from `available_players`.
            *   If no player is found for a required slot, that slot remains unfilled by this heuristic (the overall solution might become invalid, caught by `is_valid()`).
    5.  Returns the `final_assignment` list. This assignment is not guaranteed to be valid if the heuristic fails to fill all slots correctly due to player pool characteristics or tight constraints.

#### 2.2.3. `is_valid(self, players_data_ref_for_check)`

```python
    def is_valid(self, players_data_ref_for_check):
        # ... (initial checks for assignment existence and length) ...
        teams = [[] for _ in range(self.num_teams)]
        # ... (populate teams based on self.assignment) ...
        for team_idx, team_players in enumerate(teams):
            if len(team_players) != self.team_size:
                return False
            # ... (check roles and budget for each team) ...
        return True
```

*   **Purpose**: Checks if the current solution's assignment meets all problem constraints.
*   **Parameter**: `players_data_ref_for_check` (typically `self.players_data_ref`).
*   **Logic**:
    1.  Basic checks: Returns `False` if `self.assignment` is `None` or its length doesn't match the total number of players.
    2.  Constructs `teams`: A list of lists, where each inner list contains player data for a team, based on `self.assignment`.
    3.  Player Assignment Count: Implicitly checks if each player is assigned to exactly one team by the structure of `self.assignment` (a list where index is player ID). The length check at the start is crucial here.
    4.  Iterates through each `team_players` in `teams`:
        *   **Team Size**: Returns `False` if `len(team_players)` is not equal to `self.team_size`.
        *   **Positional Roles**: Counts players in each role (`GK`, `DEF`, `MID`, `FWD`) within the team. Returns `False` if these counts don't match the required structure (1 GK, 2 DEF, 2 MID, 2 FWD).
        *   **Budget**: Calculates the total `budget` for the team. Returns `False` if `budget` exceeds `self.max_budget`.
    5.  If all checks pass for all teams, returns `True`.

#### 2.2.4. `fitness(self, players_data_ref_for_fitness)`

```python
    def fitness(self, players_data_ref_for_fitness):
        if not self.is_valid(players_data_ref_for_fitness):
            return float("inf")
        # ... (calculate average skill for each team) ...
        avg_skills = [np.mean(skills_in_team) for skills_in_team in team_skills_values]
        return np.std(avg_skills)
```

*   **Purpose**: Calculates the fitness of the solution. Lower fitness (standard deviation) is better.
*   **Parameter**: `players_data_ref_for_fitness` (typically `self.players_data_ref`).
*   **Logic**:
    1.  Validity Check: If `self.is_valid()` returns `False`, the solution is infeasible, so its fitness is `float("inf")` (a very large number, effectively penalizing it).
    2.  Team Skills: Gathers the skill values of players in each team.
    3.  Average Skills: Calculates the average skill for each team using `np.mean()`.
    4.  Standard Deviation: Calculates the standard deviation of these average team skills using `np.std()`. This value is the fitness score.

#### 2.2.5. `copy(self)`

```python
    def copy(self):
        return LeagueSolution(self.players_data_ref, assignment=deepcopy(self.assignment), \
                              num_teams=self.num_teams, team_size=self.team_size, \
                              max_budget=self.max_budget)
```

*   **Purpose**: Creates and returns a deep copy of the current `LeagueSolution` instance.
*   **Logic**: Initializes a new `LeagueSolution` object, passing the reference to `players_data_ref` (as it's read-only data and shared) and a `deepcopy` of the `self.assignment` list. Other parameters (`num_teams`, etc.) are also passed.

### 2.3. Class `LeagueHillClimbingSolution(LeagueSolution)`

This class inherits from `LeagueSolution` and is specialized for the Hill Climbing algorithm.

#### 2.3.1. `__init__(...)`

```python
class LeagueHillClimbingSolution(LeagueSolution):
    def __init__(self, players_data_full, assignment=None, num_teams=5, team_size=7, max_budget=750):
        super().__init__(players_data_full, assignment, num_teams, team_size, max_budget)
```

*   **Purpose**: Initializes a `LeagueHillClimbingSolution` object.
*   **Logic**: Simply calls the `__init__` method of the parent class (`LeagueSolution`) using `super()`.

#### 2.3.2. `get_neighbors(self, players_data_ref_for_neighbors)`

```python
    def get_neighbors(self, players_data_ref_for_neighbors):
        neighbors = []
        num_players = len(self.assignment)
        for i in range(num_players):
            for j in range(i + 1, num_players):
                new_assign = deepcopy(self.assignment)
                new_assign[i], new_assign[j] = new_assign[j], new_assign[i] # Swap player assignments
                neighbor = LeagueHillClimbingSolution(players_data_ref_for_neighbors, assignment=new_assign, ...)
                if neighbor.is_valid(players_data_ref_for_neighbors):
                    neighbors.append(neighbor)
        return neighbors
```

*   **Purpose**: Generates all valid neighboring solutions by swapping the team assignments of every pair of players.
*   **Parameter**: `players_data_ref_for_neighbors` (typically `self.players_data_ref`).
*   **Logic**:
    1.  Initializes an empty list `neighbors`.
    2.  Iterates through all unique pairs of player indices (`i`, `j`).
    3.  For each pair:
        *   Creates a `deepcopy` of the current `self.assignment`.
        *   Swaps the team assignments of player `i` and player `j` in this new assignment.
        *   Creates a new `LeagueHillClimbingSolution` object (`neighbor`) with this modified assignment.
        *   If `neighbor.is_valid()` is `True`, adds the `neighbor` to the `neighbors` list.
    4.  Returns the list of `neighbors`.

#### 2.3.3. `copy(self)`

```python
    def copy(self):
        return LeagueHillClimbingSolution(self.players_data_ref, assignment=deepcopy(self.assignment), ...)
```

*   **Purpose**: Overrides the base class `copy` method to ensure that copying a `LeagueHillClimbingSolution` results in a new instance of `LeagueHillClimbingSolution`.

### 2.4. Class `LeagueSASolution(LeagueSolution)`

This class inherits from `LeagueSolution` and is specialized for the Simulated Annealing algorithm.

#### 2.4.1. `__init__(...)`

```python
class LeagueSASolution(LeagueSolution):
    def __init__(self, players_data_full, assignment=None, num_teams=5, team_size=7, max_budget=750):
        super().__init__(players_data_full, assignment, num_teams, team_size, max_budget)
```

*   **Purpose**: Initializes a `LeagueSASolution` object.
*   **Logic**: Calls the `__init__` method of the parent class (`LeagueSolution`).

#### 2.4.2. `get_random_neighbor(self, players_data_ref_for_neighbor)`

```python
    def get_random_neighbor(self, players_data_ref_for_neighbor):
        max_attempts = 100 
        attempts = 0
        num_players = len(self.assignment)
        while attempts < max_attempts:
            idx1, idx2 = random.sample(range(num_players), 2)
            candidate_assignment = deepcopy(self.assignment)
            candidate_assignment[idx1], candidate_assignment[idx2] = candidate_assignment[idx2], candidate_assignment[idx1]
            potential_neighbor = LeagueSASolution(players_data_ref_for_neighbor, assignment=candidate_assignment, ...)
            if potential_neighbor.is_valid(players_data_ref_for_neighbor):
                return potential_neighbor
            attempts += 1
        return self.copy()
```

*   **Purpose**: Generates a single random, valid neighboring solution by swapping the assignments of two randomly chosen players.
*   **Parameter**: `players_data_ref_for_neighbor` (typically `self.players_data_ref`).
*   **Logic**:
    1.  Sets `max_attempts` to find a valid neighbor (e.g., 100).
    2.  In a loop up to `max_attempts`:
        *   Randomly samples two distinct player indices (`idx1`, `idx2`).
        *   Creates a `deepcopy` of `self.assignment` and swaps the assignments for these two players.
        *   Creates a new `LeagueSASolution` (`potential_neighbor`) with this new assignment.
        *   If `potential_neighbor.is_valid()`, returns it immediately.
    3.  If no valid neighbor is found after `max_attempts`, returns a `copy()` of the current solution (effectively staying in the same state).

#### 2.4.3. `copy(self)`

```python
    def copy(self):
        return LeagueSASolution(self.players_data_ref, assignment=deepcopy(self.assignment), ...)
```

*   **Purpose**: Overrides the base class `copy` method to ensure that copying a `LeagueSASolution` results in a new instance of `LeagueSASolution`.

## 3. Review of `evolution.py`

This file implements the core logic for the Genetic Algorithm, Hill Climbing, and Simulated Annealing.

### 3.1. Imports

```python
import random
import numpy as np
from copy import deepcopy
from solution import LeagueSolution
from operators import (
    mutate_swap_constrained, # Example mutation
    # ... other mutation, crossover, and selection operators
)
```

*   Imports `random`, `numpy`, `deepcopy` similar to `solution.py`.
*   Imports `LeagueSolution` from `solution.py` (used for GA population generation).
*   Imports various operator functions (mutation, crossover, selection) from `operators.py`.

### 3.2. Function `generate_population(...)`

```python
def generate_population(players_data_full, size, num_teams=5, team_size=7, max_budget=750):
    population = []
    # ... (loop with attempts to create valid individuals)
    while len(population) < size and attempts_total < max_total_attempts:
        candidate = LeagueSolution(players_data_full, assignment=None, ...)
        if candidate.is_valid(players_data_full):
            population.append(candidate)
        # ...
    return population
```

*   **Purpose**: Creates an initial population of `LeagueSolution` objects for the Genetic Algorithm.
*   **Logic**:
    1.  Initializes an empty `population` list.
    2.  Loops until the `population` reaches the desired `size` or a `max_total_attempts` limit is hit.
    3.  In each iteration, creates a `candidate` `LeagueSolution`. The `LeagueSolution` constructor itself calls `_random_valid_assignment_constructive`.
    4.  Checks if the `candidate.is_valid()`. If so, adds it to the `population`.
    5.  Prints a warning if the full `size` could not be achieved.
    6.  Returns the generated `population`.

### 3.3. Function `genetic_algorithm(...)`

```python
def genetic_algorithm(
    players_data, 
    population_size=50, generations=30, mutation_rate=0.2, elite_size=5,
    mutation_operator_func=mutate_swap_constrained, # Default operators
    # ... other parameters like crossover_operator_func, selection_operator_func, etc.
):
    population = generate_population(players_data, population_size, ...)
    if not population: # Handle empty initial population
        return None, []
    history = []
    best_solution = min(population, key=lambda s: s.fitness(players_data))
    history.append(best_solution.fitness(players_data))

    for gen in range(generations):
        new_population = []
        population.sort(key=lambda x: x.fitness(players_data)) # Sort by fitness
        new_population.extend(population[:elite_size]) # Elitism

        while len(new_population) < population_size:
            # Selection
            parent1 = selection_operator_func(population, players_data, ...)
            parent2 = selection_operator_func(population, players_data, ...)
            # Crossover
            child = crossover_operator_func(parent1, parent2, players_data)
            # Mutation
            if random.random() < mutation_rate:
                child = mutation_operator_func(child, players_data)
            
            if child.is_valid(players_data):
                new_population.append(child)
            # ... (optional: fill with new valid individuals if child is invalid)
        
        if not new_population: break # Stop if population collapses
        population = new_population
        # Update best_solution and history
        # ...
    return best_solution, history
```

*   **Purpose**: Implements the main Genetic Algorithm loop.
*   **Parameters**: Takes `players_data`, GA parameters (population size, generations, mutation rate, elite size), and functions for mutation, crossover, and selection operators, along with their specific parameters (e.g., `tournament_k`).
*   **Logic**:
    1.  **Initialization**: Generates an initial `population` using `generate_population()`. If the population is empty, returns `None`.
    2.  **History Tracking**: Initializes `history` (list to store best fitness per generation) and finds the initial `best_solution` in the population.
    3.  **Generational Loop**: Iterates for the specified number of `generations`:
        *   **Elitism**: Copies the top `elite_size` individuals from the current sorted `population` to `new_population`.
        *   **Reproduction Loop**: Continues until `new_population` reaches `population_size`:
            *   **Selection**: Selects two `parent1` and `parent2` from the current `population` using the provided `selection_operator_func` (e.g., tournament, ranking, Boltzmann).
            *   **Crossover**: Creates a `child` solution by applying `crossover_operator_func` to `parent1` and `parent2`.
            *   **Mutation**: Applies `mutation_operator_func` to the `child` with a probability of `mutation_rate`.
            *   **Validation & Addition**: If the `child.is_valid()`, it's added to `new_population`. An optional step to generate a new valid filler if the child is invalid is included to maintain population size.
        *   **Population Update**: If `new_population` is empty (population collapse), the GA stops. Otherwise, `population` is replaced by `new_population`.
        *   **Best Solution Update**: Finds the best individual in the current `population`. If it's better than the overall `best_solution`, `best_solution` is updated (using `deepcopy`).
        *   Appends the current overall `best_solution.fitness()` to `history`.
    4.  Returns the overall `best_solution` found and the `history` of best fitness values.

### 3.4. Function `hill_climbing(...)`

```python
def hill_climbing(initial_solution, players_data, max_iterations=1000, verbose=False):
    current_solution = initial_solution 
    current_fitness = current_solution.fitness(players_data)
    history = [current_fitness]

    for iteration in range(max_iterations):
        neighbors = current_solution.get_neighbors(players_data)
        if not neighbors: break # No valid neighbors

        best_neighbor = min(neighbors, key=lambda x: x.fitness(players_data))
        best_neighbor_fitness = best_neighbor.fitness(players_data)

        if best_neighbor_fitness < current_fitness:
            current_solution = best_neighbor 
            current_fitness = best_neighbor_fitness
            history.append(current_fitness)
        else:
            break # No improvement
            
    return current_solution, current_fitness, history
```

*   **Purpose**: Implements the Hill Climbing local search algorithm.
*   **Parameters**: `initial_solution` (a `LeagueHillClimbingSolution` object), `players_data`, `max_iterations`.
*   **Logic**:
    1.  **Initialization**: Sets `current_solution` to `initial_solution` and calculates its `current_fitness`. Initializes `history`.
    2.  **Iterative Improvement**: Loops for `max_iterations`:
        *   Gets all valid `neighbors` of the `current_solution` using `current_solution.get_neighbors()`.
        *   If no `neighbors` are found, stops.
        *   Finds the `best_neighbor` among them (lowest fitness).
        *   If `best_neighbor_fitness` is less than `current_fitness`:
            *   Updates `current_solution` to `best_neighbor` and `current_fitness`.
            *   Appends the new `current_fitness` to `history`.
        *   Else (no improvement found), stops.
    3.  Returns the final `current_solution`, its `current_fitness`, and the `history` of fitness values.

### 3.5. Function `simulated_annealing(...)`

```python
def simulated_annealing(
    initial_solution, players_data, initial_temp=1000, final_temp=1,
    alpha=0.99, iterations_per_temp=100, verbose=False
):
    current_solution = initial_solution
    current_fitness = current_solution.fitness(players_data)
    best_solution = deepcopy(current_solution) 
    best_fitness = current_fitness
    history = [current_fitness] 
    temp = initial_temp

    while temp > final_temp:
        for _ in range(iterations_per_temp):
            neighbor_solution = current_solution.get_random_neighbor(players_data)
            neighbor_fitness = neighbor_solution.fitness(players_data)
            delta_e = neighbor_fitness - current_fitness
            
            if delta_e < 0: # Better solution
                current_solution = deepcopy(neighbor_solution) 
                current_fitness = neighbor_fitness
                if current_fitness < best_fitness: # Update overall best
                    best_solution = deepcopy(current_solution)
                    best_fitness = current_fitness
            else: # Worse solution, accept with probability
                if temp > 1e-8: # Avoid division by zero if temp is too small
                    acceptance_probability = np.exp(-delta_e / temp)
                    if random.random() < acceptance_probability:
                        current_solution = deepcopy(neighbor_solution) 
                        current_fitness = neighbor_fitness
            history.append(current_fitness) # Track current fitness at each step
        temp *= alpha # Cool down

    return best_solution, best_fitness, history
```

*   **Purpose**: Implements the Simulated Annealing algorithm.
*   **Parameters**: `initial_solution` (a `LeagueSASolution` object), `players_data`, and SA parameters (`initial_temp`, `final_temp`, `alpha` cooling rate, `iterations_per_temp`).
*   **Logic**:
    1.  **Initialization**: Sets `current_solution` to `initial_solution`, calculates `current_fitness`. Initializes `best_solution` (as a `deepcopy`), `best_fitness`, `history`, and `temp` to `initial_temp`.
    2.  **Cooling Loop**: Continues as long as `temp` is greater than `final_temp`:
        *   **Inner Loop**: Iterates `iterations_per_temp` times:
            *   Generates a `neighbor_solution` using `current_solution.get_random_neighbor()`.
            *   Calculates `neighbor_fitness` and `delta_e` (change in fitness).
            *   **Acceptance Criteria**:
                *   If `delta_e < 0` (neighbor is better), accept it: update `current_solution` and `current_fitness`. If this new `current_fitness` is also better than `best_fitness`, update `best_solution` and `best_fitness`.
                *   Else (neighbor is worse), calculate `acceptance_probability` using `np.exp(-delta_e / temp)`. If `random.random()` is less than this probability, accept the worse solution (update `current_solution` and `current_fitness`).
            *   Appends the `current_fitness` (which might be from the accepted neighbor or the previous solution) to `history`.
        *   **Cool Down**: Multiplies `temp` by `alpha` (e.g., 0.99) to decrease it.
    3.  Returns the overall `best_solution` found, its `best_fitness`, and the `history` of current fitness values over iterations.

## 4. Review of `main_script_sp.py`

This script orchestrates the experiments for the single-processor version.

### 4.1. Imports and Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from solution import LeagueSolution, LeagueHillClimbingSolution, LeagueSASolution
from evolution import genetic_algorithm, hill_climbing, simulated_annealing 
from operators import (
    # ... various operator functions ...
)

SP_GRAPHS_DIR = "/home/ubuntu/CIFO_EXTENDED_Project/sp_graphs"
if not os.path.exists(SP_GRAPHS_DIR):
    os.makedirs(SP_GRAPHS_DIR)

players_df = pd.read_csv("players.csv", sep=";")
players_data = players_df.to_dict(orient="records")

NUM_TEAMS = 5
TEAM_SIZE = 7
MAX_BUDGET = 750
NUM_RUNS = 30
```

*   **Standard Imports**: `pandas` for CSV reading, `numpy` for numerical ops, `matplotlib.pyplot` for plotting, `time` for timing, `os` for path operations.
*   **Project-Specific Imports**: Solution classes from `solution.py`, algorithm functions from `evolution.py`, and operator functions from `operators.py`.
*   **Graph Directory**: Defines `SP_GRAPHS_DIR` and creates it if it doesn't exist.
*   **Data Loading**: Loads player data from `players.csv` into a pandas DataFrame and then converts it to `players_data` (list of dictionaries).
*   **Global Parameters**: Defines `NUM_TEAMS`, `TEAM_SIZE`, `MAX_BUDGET`, and `NUM_RUNS` (number of independent executions for each algorithm).

### 4.2. `main()` Function

This function contains the main experimental logic.

```python
def main():
    script_total_start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Single-Processor Script execution started.")
    # ... (data loading logs) ...
    all_results_summary = []
```

*   Starts with overall script timing and logging.
*   Initializes `all_results_summary` to store aggregated results from different algorithms.

#### 4.2.1. Hill Climbing Section

```python
    hc_section_start_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting Hill Climbing Algorithm ({NUM_RUNS} runs) ---")
    hc_all_fitness_values = []
    hc_all_exec_times = []
    # ... (variables for best overall HC solution) ...

    for i in range(NUM_RUNS):
        # ... (log run number and start time) ...
        initial_hc_solution_run = LeagueHillClimbingSolution(players_data, ...)
        # Retry loop for valid initial solution
        while not initial_hc_solution_run.is_valid(players_data) and retry_attempts_hc < max_retry_hc:
            # ... (retry logic) ...
        if not initial_hc_solution_run.is_valid(players_data):
            # ... (skip run if no valid initial solution) ...
            continue

        hc_solution_obj_run, hc_fitness_val_run, hc_history_convergence_run = hill_climbing(...)
        # ... (store fitness, time, update best overall HC solution and history) ...

    # Calculate mean/std for HC fitness and time
    # ...
    # Log HC summary results
    # ...
    # Plot HC convergence for the best run
    if best_hc_solution_overall:
        plt.figure(...)
        plt.plot(best_hc_history_overall, ...)
        plt.savefig(os.path.join(SP_GRAPHS_DIR, "hc_convergence_sp.png"))
        # ...
    # Add HC results to all_results_summary
    # ...
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hill Climbing section took ... seconds.")
```

*   **Loop**: Runs the Hill Climbing algorithm `NUM_RUNS` times.
*   **Initialization**: For each run, creates an `initial_hc_solution_run` of type `LeagueHillClimbingSolution`.
*   **Retry Mechanism**: Includes a loop to retry generating a valid initial solution up to `max_retry_hc` times if the first attempt fails.
*   **Execution**: Calls `hill_climbing()` from `evolution.py`.
*   **Data Collection**: Stores fitness values and execution times for each run. Tracks the `best_hc_solution_overall` and its convergence `best_hc_history_overall`.
*   **Aggregation & Logging**: After all runs, calculates and prints mean fitness, standard deviation of fitness, and mean execution time. Logs the overall best HC fitness.
*   **Plotting**: Generates and saves a convergence plot for the best HC run to the `sp_graphs` directory.
*   **Summary**: Appends HC results to `all_results_summary`.

#### 4.2.2. Simulated Annealing Section

This section follows a very similar structure to the Hill Climbing section:
*   Loops `NUM_RUNS` times.
*   Initializes `LeagueSASolution` for each run, with a retry mechanism for validity.
*   Calls `simulated_annealing()` from `evolution.py` with specified SA parameters (`initial_temp`, `final_temp`, etc.).
*   Collects, aggregates, logs, and plots results for SA, saving `sa_convergence_sp.png`.
*   Appends SA results to `all_results_summary`.

#### 4.2.3. Genetic Algorithm Section

```python
    ga_section_start_time = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Starting Genetic Algorithms ({NUM_RUNS} runs per config) ---")
    ga_params_dict = { # General GA parameters
        "population_size": 50, "generations": 100, ...
    }
    ga_configs_new = [
        { "name": "GA_Config_1 (...)", "mutation_operator_func": ..., ... },
        # ... (other GA configurations with different operators/parameters) ...
    ]

    for config_idx, ga_config_dict in enumerate(ga_configs_new):
        # ... (log config name) ...
        ga_all_fitness_values_config = []
        # ... (variables for best overall GA solution for this config) ...

        for i in range(NUM_RUNS):
            # ... (log run number) ...
            best_solution_ga_run, history_ga_run = genetic_algorithm(
                players_data=players_data,
                # ... (pass parameters from ga_params_dict and ga_config_dict) ...
            )
            # ... (store fitness, time, update best overall GA for this config) ...

        # Calculate mean/std for this GA config
        # ...
        # Log summary for this GA config
        # ...
        # Plot convergence for the best run of this GA config
        if best_ga_solution_overall_config:
            plt.figure(...)
            plt.plot(best_ga_history_overall_config, ...)
            plt.savefig(os.path.join(SP_GRAPHS_DIR, f"ga_convergence_{ga_config_dict['name'].replace(' ', '_').replace('(', '').replace(')', '')}_sp.png"))
            # ...
        # Add results for this GA config to all_results_summary
        # ...
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Genetic Algorithms section took ... seconds.")
```

*   **Configurations**: Defines a `ga_params_dict` for common GA parameters and a list `ga_configs_new`, where each element is a dictionary specifying a unique GA configuration (name, mutation operator, crossover operator, selection operator, and specific operator parameters like `tournament_k`).
*   **Outer Loop**: Iterates through each `ga_config_dict` in `ga_configs_new`.
*   **Inner Loop**: For each configuration, runs the Genetic Algorithm `NUM_RUNS` times.
*   **Execution**: Calls `genetic_algorithm()` from `evolution.py`, passing the appropriate parameters from `ga_params_dict` and the current `ga_config_dict`.
*   **Data Collection, Aggregation, Logging, Plotting**: Similar to HC and SA, but performed per GA configuration. Convergence plots are saved with names reflecting the GA configuration (e.g., `ga_convergence_GA_Config_1_..._sp.png`).
*   **Summary**: Appends results for each GA configuration to `all_results_summary`.

#### 4.2.4. Comparative Analysis and Plots

```python
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Generating Comparative Analysis Plots ---")
    results_df = pd.DataFrame(all_results_summary)
    print("\nOverall Results Summary Table (Single-Processor):")
    print(results_df.to_string())

    # Plotting comparative fitness
    plt.figure(figsize=(12, 7))
    # ... (bar plot for Mean Fitness of all algorithms/configs) ...
    plt.savefig(os.path.join(SP_GRAPHS_DIR, "comparative_fitness_sp.png"))
    plt.close()

    # Plotting comparative execution times
    plt.figure(figsize=(12, 7))
    # ... (bar plot for Mean Execution Time of all algorithms/configs) ...
    plt.savefig(os.path.join(SP_GRAPHS_DIR, "comparative_times_sp.png"))
    plt.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Comparative plots saved to {SP_GRAPHS_DIR}")
```

*   **DataFrame**: Converts `all_results_summary` (list of dictionaries) into a pandas DataFrame for easy display and plotting.
*   **Summary Table**: Prints the `results_df` to the console.
*   **Comparative Plots**: 
    *   Generates a bar chart comparing the 
