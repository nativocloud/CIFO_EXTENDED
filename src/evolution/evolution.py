import random
import numpy as np
from copy import deepcopy

from solution import LeagueSolution # LeagueHillClimbingSolution and LeagueSASolution are used in main_script.py directly
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


def generate_population(players_data_full, size, num_teams=5, team_size=7, max_budget=750):
    """Generates an initial population of LeagueSolution objects."""
    population = []
    attempts_total = 0
    max_total_attempts = size * 100 # Try up to 100 times per individual needed

    while len(population) < size and attempts_total < max_total_attempts:
        # Create a candidate solution using the new constructor that takes players_data_full
        # The constructor itself now tries to create a valid assignment.
        candidate = LeagueSolution(players_data_full, assignment=None, 
                                   num_teams=num_teams, team_size=team_size, 
                                   max_budget=max_budget)
        
        # Check if the constructively generated assignment is valid
        if candidate.is_valid(): # MODIFIED: Vectorized call
            population.append(candidate)
        else:
            # Optionally, log if the constructive heuristic failed for this attempt
            # print(f"Debug: Constructive heuristic failed to produce a valid individual in generate_population attempt {attempts_total + 1}")
            pass
        attempts_total += 1

    if len(population) < size:
        print(f"Warning: Could only generate {len(population)}/{size} valid individuals for initial population after {attempts_total} attempts.")
    return population


def genetic_algorithm(
    players_data, 
    population_size=50,
    generations=30,
    mutation_rate=0.2,
    elite_size=5,
    mutation_operator_func=mutate_swap_constrained,
    crossover_operator_func=crossover_one_point_prefer_valid,
    selection_operator_func=selection_tournament_variable_k,
    tournament_k=3,
    boltzmann_temp=100,
    num_teams=5, 
    team_size=7,
    max_budget=750,
    verbose=False
):
    # generate_population now correctly uses players_data (as players_data_full)
    population = generate_population(players_data, population_size, num_teams, team_size, max_budget)
    if not population:
        print("Error: Initial population is empty. GA cannot proceed.")
        return None, []
        
    history = []
    # Ensure fitness is called with players_data
    best_solution = min(population, key=lambda s: s.fitness())
    best_fitness_initial = best_solution.fitness()
    history.append(best_fitness_initial)

    if verbose:
        print(f"Generation 0: Best Fitness = {best_fitness_initial}")

    for gen in range(generations):
        new_population = []
        # Ensure fitness is called with players_data for sorting
        population.sort(key=lambda x: x.fitness())
        new_population.extend(population[:elite_size])

        while len(new_population) < population_size:
            # Selection operators should also receive players_data if they need to evaluate fitness internally
            # Assuming current selection operators in operators.py are adapted or don_t directly call fitness
            # For now, passing players_data to them as they might need it for fitness evaluation
            if selection_operator_func == selection_tournament_variable_k:
                parent1 = selection_operator_func(population, players_data, k=tournament_k)
                parent2 = selection_operator_func(population, players_data, k=tournament_k)
            elif selection_operator_func == selection_boltzmann:
                parent1 = selection_operator_func(population, players_data, temperature=boltzmann_temp, k=1) # Assuming k=1 for single selection
                parent2 = selection_operator_func(population, players_data, temperature=boltzmann_temp, k=1)
            else: # e.g., selection_ranking
                parent1 = selection_operator_func(population, players_data)
                parent2 = selection_operator_func(population, players_data)
            
            # Crossover operators should also receive players_data
            child = crossover_operator_func(parent1, parent2, players_data)

            if random.random() < mutation_rate:
                # Mutation operators should also receive players_data
                child = mutation_operator_func(child, players_data)
            
            # Validate the child with players_data
            if child.is_valid(): # MODIFIED: Vectorized call
                new_population.append(child)
            elif len(new_population) < population_size: # If child is invalid, try to fill with a new valid one
                # This part might be slow if generation is hard. Consider if this is the best strategy.
                # For now, keeping it to ensure population size if possible.
                valid_fillers = generate_population(players_data, 1, num_teams, team_size, max_budget)
                if valid_fillers:
                    new_population.append(valid_fillers[0])
                # else: # Could not generate a filler, new_population might be smaller

        if not new_population: # If, after all attempts, new_population is empty
            if verbose:
                print(f"Warning: Generation {gen+1} resulted in an empty new population. Stopping GA.")
            break 
            
        population = new_population
        # Ensure fitness is called with players_data
        current_best_in_pop = min(population, key=lambda s: s.fitness())
        
        # Update overall best_solution if current_best_in_pop is better
        if current_best_in_pop.fitness() < best_solution.fitness():
            best_solution = deepcopy(current_best_in_pop) # Use deepcopy for solution objects
        
        history.append(best_solution.fitness())
        if verbose:
            print(f"Generation {gen+1}: Best Fitness = {best_solution.fitness()}")

    return best_solution, history


def hill_climbing(initial_solution, players_data, max_iterations=1000, verbose=False):
    current_solution = initial_solution 
    current_fitness = current_solution.fitness() # MODIFIED: Vectorized call
    history = [current_fitness]
    if verbose:
        print(f"Initial HC solution fitness: {current_fitness}")

    for iteration in range(max_iterations):
        neighbors = current_solution.get_neighbors() # MODIFIED: Vectorized call
        if not neighbors:
            if verbose:
                print(f"Iteration {iteration}: No valid neighbors found. Stopping.")
            break

        best_neighbor = min(neighbors, key=lambda x: x.fitness())
        best_neighbor_fitness = best_neighbor.fitness()

        if best_neighbor_fitness < current_fitness:
            current_solution = best_neighbor 
            current_fitness = best_neighbor_fitness
            history.append(current_fitness)
            if verbose:
                print(f"Iteration {iteration}: New best fitness = {current_fitness}")
        else:
            if verbose:
                print(f"Iteration {iteration}: No better neighbor found. Stopping.")
            break 
            
    if verbose:
        print(f"Hill Climbing finished. Best fitness: {current_fitness}")
    return current_solution, current_fitness, history
def simulated_annealing(
    initial_solution, 
    players_data,     
    initial_temp=1000,
    final_temp=1,
    alpha=0.99,
    iterations_per_temp=100,
    verbose=False
):
    current_solution = initial_solution
    current_fitness = current_solution.fitness() # MODIFIED: Vectorized call 
    best_fitness = current_fitness
    
    history = [current_fitness] 
    temp = initial_temp
    iteration_count = 0

    if verbose:
        print(f"Initial SA solution fitness: {current_fitness}, Temp: {temp}")

    while temp > final_temp:
        for _ in range(iterations_per_temp):
            iteration_count += 1
            neighbor_solution = current_solution.get_random_neighbor() # MODIFIED: Vectorized call
            neighbor_fitness = neighbor_solution.fitness() # MODIFIED: Vectorized call   
            delta_e = neighbor_fitness - current_fitness
            
            if delta_e < 0: 
                current_solution = neighbor_solution # Optimized: Assuming get_random_neighbor returns a new, independent object
                current_fitness = neighbor_fitness
                if current_fitness < best_fitness: 
                    best_solution = deepcopy(current_solution)
                    best_fitness = current_fitness
            else: 
                if temp > 1e-8: 
                    acceptance_probability = np.exp(-delta_e / temp)
                    if random.random() < acceptance_probability:
                        current_solution = neighbor_solution # Optimized: Assuming get_random_neighbor returns a new, independent object
                        current_fitness = neighbor_fitness
            
            history.append(current_fitness) 
            if verbose and iteration_count % (iterations_per_temp * 10) == 0:
                print(f"Iter: {iteration_count}, Temp: {temp:.2f}, Current Fitness: {current_fitness:.4f}, Best Fitness: {best_fitness:.4f}")
        
        temp *= alpha 

    if verbose:
        print(f"Simulated Annealing finished. Best fitness found: {best_fitness}")
    return best_solution, best_fitness, history

