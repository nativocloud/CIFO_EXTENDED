import random
import numpy as np
from copy import deepcopy

from solution import LeagueSolution, LeagueHillClimbingSolution, LeagueSASolution
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


def generate_population(players_data, size, num_teams=5, team_size=7, max_budget=750):
    population = []
    attempts = 0
    max_attempts_per_individual = 100
    while len(population) < size and attempts < size * max_attempts_per_individual:
        # Note: LeagueSolution still needs num_teams, team_size, max_budget for its own __init__
        # if it were to create a random assignment internally without these.
        # However, if an assignment is passed, these might not be strictly needed in its __init__
        # For now, we keep them as they are used if assignment is None.
        candidate = LeagueSolution(num_teams=num_teams, team_size=team_size, max_budget=max_budget)
        if candidate.is_valid(players_data):
            population.append(candidate)
        attempts += 1
    if len(population) < size:
        print(f"Warning: Could only generate {len(population)}/{size} valid individuals for initial population.")
    return population


def genetic_algorithm(
    players_data, # Changed from players to players_data for consistency
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
    # generate_population now takes players_data
    population = generate_population(players_data, population_size, num_teams, team_size, max_budget)
    if not population:
        print("Error: Initial population is empty. GA cannot proceed.")
        return None, []
        
    history = []
    best_solution = min(population, key=lambda s: s.fitness(players_data))
    best_fitness_initial = best_solution.fitness(players_data)
    history.append(best_fitness_initial)

    if verbose:
        print(f"Generation 0: Best Fitness = {best_fitness_initial}")

    for gen in range(generations):
        new_population = []
        population.sort(key=lambda x: x.fitness(players_data))
        new_population.extend(population[:elite_size])

        while len(new_population) < population_size:
            if selection_operator_func == selection_tournament_variable_k:
                parent1 = selection_operator_func(population, players_data, k=tournament_k)
                parent2 = selection_operator_func(population, players_data, k=tournament_k)
            elif selection_operator_func == selection_boltzmann:
                parent1 = selection_operator_func(population, players_data, temperature=boltzmann_temp, k=1)
                parent2 = selection_operator_func(population, players_data, temperature=boltzmann_temp, k=1)
            else:
                parent1 = selection_operator_func(population, players_data)
                parent2 = selection_operator_func(population, players_data)
            
            if crossover_operator_func in [crossover_one_point_prefer_valid, crossover_uniform_prefer_valid]:
                child = crossover_operator_func(parent1, parent2, players_data)
            else:
                child = crossover_operator_func(parent1, parent2)

            if random.random() < mutation_rate:
                if mutation_operator_func in [mutate_swap_constrained, mutate_targeted_player_exchange, mutate_shuffle_within_team_constrained]:
                    child = mutation_operator_func(child, players_data)
                else:
                    child = mutation_operator_func(child)
            
            if child.is_valid(players_data):
                new_population.append(child)
            elif len(new_population) < population_size:
                valid_fillers = generate_population(players_data, 1, num_teams, team_size, max_budget)
                if valid_fillers:
                    new_population.append(valid_fillers[0])

        if not new_population:
            if verbose:
                print(f"Warning: Generation {gen+1} resulted in an empty new population. Stopping GA.")
            break 
            
        population = new_population
        current_best_in_pop = min(population, key=lambda s: s.fitness(players_data))
        
        # Update overall best_solution if current_best_in_pop is better
        if current_best_in_pop.fitness(players_data) < best_solution.fitness(players_data):
            best_solution = deepcopy(current_best_in_pop)
        
        history.append(best_solution.fitness(players_data))
        if verbose:
            print(f"Generation {gen+1}: Best Fitness = {best_solution.fitness(players_data)}")

    return best_solution, history


def hill_climbing(initial_solution, players_data, max_iterations=1000, verbose=False):
    """
    Generic Hill Climbing algorithm.
    Assumes initial_solution is a valid solution object that implements:
    - fitness(players_data)
    - get_neighbors(players_data) -> returns list of neighbor solution objects
    - is_valid(players_data) (implicitly, neighbors should be valid or filtered)
    """
    current_solution = initial_solution # current_solution is already an instance of LeagueHillClimbingSolution
    # It is assumed that initial_solution is already validated before calling this function.
    
    current_fitness = current_solution.fitness(players_data)
    history = [current_fitness]
    if verbose:
        print(f"Initial HC solution fitness: {current_fitness}")

    for iteration in range(max_iterations):
        # get_neighbors should return valid neighbors or they should be filtered if not inherently valid
        neighbors = current_solution.get_neighbors(players_data)
        if not neighbors: # No valid neighbors found by the solution's method
            if verbose:
                print(f"Iteration {iteration}: No valid neighbors found. Stopping.")
            break

        # Find the best neighbor (minimization)
        best_neighbor = min(neighbors, key=lambda x: x.fitness(players_data))
        best_neighbor_fitness = best_neighbor.fitness(players_data)

        if best_neighbor_fitness < current_fitness:
            current_solution = best_neighbor # The neighbor object itself
            current_fitness = best_neighbor_fitness
            history.append(current_fitness)
            if verbose:
                print(f"Iteration {iteration}: New best fitness = {current_fitness}")
        else:
            if verbose:
                print(f"Iteration {iteration}: No better neighbor found. Stopping.")
            break # Local optimum reached
            
    if verbose:
        print(f"Hill Climbing finished. Best fitness: {current_fitness}")
    # Return the best solution object found and its fitness, and history
    return current_solution, current_fitness, history


def simulated_annealing(
    initial_solution, # Expects an instance of LeagueSASolution (or similar with required methods)
    players_data,     # The players data needed for fitness and neighbor generation
    initial_temp=1000,
    final_temp=1,
    alpha=0.99,
    iterations_per_temp=100,
    verbose=False
):
    """
    Generic Simulated Annealing algorithm.
    Assumes initial_solution is a valid solution object that implements:
    - fitness(players_data)
    - get_random_neighbor(players_data) -> returns a random neighbor solution object
    - is_valid(players_data) (implicitly, random neighbor should be valid or handled)
    """
    current_solution = initial_solution # current_solution is already an instance of LeagueSASolution
    # It is assumed that initial_solution is already validated before calling this function.

    current_fitness = current_solution.fitness(players_data)
    best_solution = deepcopy(current_solution) # Keep track of the best solution found so far
    best_fitness = current_fitness
    
    history = [current_fitness] # Tracks fitness of the current solution over iterations
    temp = initial_temp
    iteration_count = 0

    if verbose:
        print(f"Initial SA solution fitness: {current_fitness}, Temp: {temp}")

    while temp > final_temp:
        for _ in range(iterations_per_temp):
            iteration_count += 1
            # get_random_neighbor should return a valid neighbor or handle validity
            neighbor_solution = current_solution.get_random_neighbor(players_data)
            neighbor_fitness = neighbor_solution.fitness(players_data)
            
            delta_e = neighbor_fitness - current_fitness
            
            if delta_e < 0: # Neighbor is better
                current_solution = deepcopy(neighbor_solution) # Accept the better neighbor
                current_fitness = neighbor_fitness
                if current_fitness < best_fitness: # Update overall best if this is the new best
                    best_solution = deepcopy(current_solution)
                    best_fitness = current_fitness
            else: # Neighbor is worse or same
                if temp > 1e-8: # Avoid division by zero if temp is very small
                    acceptance_probability = np.exp(-delta_e / temp)
                    if random.random() < acceptance_probability:
                        current_solution = deepcopy(neighbor_solution) # Accept worse solution
                        current_fitness = neighbor_fitness
            
            history.append(current_fitness) # Log current fitness for convergence tracking
            if verbose and iteration_count % (iterations_per_temp * 10) == 0:
                print(f"Iter: {iteration_count}, Temp: {temp:.2f}, Current Fitness: {current_fitness:.4f}, Best Fitness: {best_fitness:.4f}")
        
        temp *= alpha # Cool down

    if verbose:
        print(f"Simulated Annealing finished. Best fitness found: {best_fitness}")
    # Return the best solution object found, its fitness, and the history of current fitness values
    return best_solution, best_fitness, history

