import random
import numpy as np # Make sure numpy is imported for selection_boltzmann
from copy import deepcopy # Make sure deepcopy is imported for SA and other parts

from solution import LeagueSolution, LeagueHillClimbingSolution, LeagueSASolution # Added LeagueSASolution

# Import all operators, including the new ones
from operators import (
    # Original/Base Mutations
    mutate_swap, 
    mutate_team_shift,
    mutate_shuffle_team, 
    # New/Adapted Mutations
    mutate_swap_constrained,
    mutate_targeted_player_exchange,
    mutate_shuffle_within_team_constrained,
    # Original/Base Crossovers
    crossover_one_point,
    crossover_uniform,
    # New/Adapted Crossovers
    crossover_one_point_prefer_valid,
    crossover_uniform_prefer_valid,
    # Original/Base Selections
    # selection_tournament, # This will be replaced by variable_k version
    selection_ranking,
    # New/Adapted Selections
    selection_tournament_variable_k,
    selection_boltzmann
)


def generate_population(players, size, num_teams=5, team_size=7, max_budget=750):
    population = []
    attempts = 0
    max_attempts_per_individual = 100 # Prevent infinite loop if valid generation is too hard
    while len(population) < size and attempts < size * max_attempts_per_individual:
        candidate = LeagueSolution(num_teams=num_teams, team_size=team_size, max_budget=max_budget)
        if candidate.is_valid(players):
            population.append(candidate)
        attempts += 1
    if len(population) < size:
        # Fallback or warning if not enough valid individuals could be generated
        print(f"Warning: Could only generate {len(population)}/{size} valid individuals for initial population.")
    return population

def genetic_algorithm(
    players,
    population_size=50,
    generations=30,
    mutation_rate=0.2,
    elite_size=5,
    mutation_operator_func=mutate_swap_constrained, # Default to a constrained one
    crossover_operator_func=crossover_one_point_prefer_valid, # Default to a prefer_valid one
    selection_operator_func=selection_tournament_variable_k,
    # Parameters for specific operators
    tournament_k=3, # For selection_tournament_variable_k
    boltzmann_temp=100, # For selection_boltzmann
    num_teams=5, # Pass problem parameters
    team_size=7,
    max_budget=750,
    verbose=False
):
    population = generate_population(players, population_size, num_teams, team_size, max_budget)
    if not population: # If population generation failed
        print("Error: Initial population is empty. GA cannot proceed.")
        return None, []
        
    history = []
    # Ensure fitness is calculated for the initial best_solution
    best_solution = min(population, key=lambda s: s.fitness(players))
    best_fitness_initial = best_solution.fitness(players)
    history.append(best_fitness_initial)

    if verbose:
        print(f"Generation 0: Best Fitness = {best_fitness_initial}")

    for gen in range(generations):
        new_population = []
        
        # Sort population by fitness (ascending for minimization)
        population.sort(key=lambda x: x.fitness(players))
        
        # Elitism: Carry over the best individuals
        new_population.extend(population[:elite_size])

        while len(new_population) < population_size:
            # Select parents
            if selection_operator_func == selection_tournament_variable_k:
                parent1 = selection_operator_func(population, players, k=tournament_k)
                parent2 = selection_operator_func(population, players, k=tournament_k)
            elif selection_operator_func == selection_boltzmann:
                parent1 = selection_operator_func(population, players, temperature=boltzmann_temp, k=1)
                parent2 = selection_operator_func(population, players, temperature=boltzmann_temp, k=1)
            else: # For selection_ranking or other simple selectors
                parent1 = selection_operator_func(population, players)
                parent2 = selection_operator_func(population, players)
            
            # Crossover
            # Pass `players` if the crossover operator expects it (e.g., prefer_valid versions)
            if crossover_operator_func in [crossover_one_point_prefer_valid, crossover_uniform_prefer_valid]:
                child = crossover_operator_func(parent1, parent2, players)
            else:
                child = crossover_operator_func(parent1, parent2)

            # Mutation
            if random.random() < mutation_rate:
                # Pass `players` if the mutation operator expects it (e.g., constrained versions)
                if mutation_operator_func in [mutate_swap_constrained, mutate_targeted_player_exchange, mutate_shuffle_within_team_constrained]:
                    child = mutation_operator_func(child, players)
                else:
                    child = mutation_operator_func(child)
            
            # Add to new population if valid
            if child.is_valid(players):
                new_population.append(child)
            elif len(new_population) < population_size: # Try to fill population if child is invalid
                # Add a random valid individual if we are struggling to fill the population
                # This is a simple strategy to maintain population size
                valid_fillers = generate_population(players, 1, num_teams, team_size, max_budget)
                if valid_fillers:
                    new_population.append(valid_fillers[0])

        if not new_population: # Should not happen if elitism > 0 and initial pop was valid
            if verbose:
                print(f"Warning: Generation {gen+1} resulted in an empty new population. Stopping GA.")
            break 
            
        population = new_population
        current_best_solution = min(population, key=lambda s: s.fitness(players))
        current_best_fitness = current_best_solution.fitness(players)
        
        if current_best_fitness < best_solution.fitness(players):
            best_solution = deepcopy(current_best_solution) # Store a deep copy
        
        history.append(best_solution.fitness(players)) # Track fitness of the *overall best* solution found so far
        if verbose:
            print(f"Generation {gen+1}: Best Fitness = {best_solution.fitness(players)}")

    return best_solution, history


def hill_climbing(players, max_iterations=1000, num_teams=5, team_size=7, max_budget=750, verbose=False):
    current = LeagueHillClimbingSolution(num_teams=num_teams, team_size=team_size, max_budget=max_budget)
    while not current.is_valid(players):
        current = LeagueHillClimbingSolution(num_teams=num_teams, team_size=team_size, max_budget=max_budget)

    current_fitness = current.fitness(players)
    history = [current_fitness]
    if verbose:
        print(f"Initial HC solution fitness: {current_fitness}")

    for iteration in range(max_iterations):
        neighbors = current.get_neighbors(players)
        if not neighbors:
            if verbose:
                print(f"Iteration {iteration}: No valid neighbors found. Stopping.")
            break

        neighbor = min(neighbors, key=lambda x: x.fitness(players))
        neighbor_fitness = neighbor.fitness(players)

        if neighbor_fitness < current_fitness:
            current = neighbor # Already a LeagueHillClimbingSolution object
            current_fitness = neighbor_fitness
            history.append(current_fitness)
            if verbose:
                print(f"Iteration {iteration}: New best fitness = {current_fitness}")
        else:
            if verbose:
                print(f"Iteration {iteration}: No better neighbor found. Stopping.")
            break
    if verbose:
        print(f"Hill Climbing finished. Best fitness: {current_fitness}")
    return current, current_fitness, history


def simulated_annealing_for_league(
    players,
    initial_temp=1000,
    final_temp=1,
    alpha=0.99,
    iterations_per_temp=100,
    num_teams=5, 
    team_size=7, 
    max_budget=750,
    verbose=False
):
    current_solution = LeagueSASolution(num_teams=num_teams, team_size=team_size, max_budget=max_budget)
    while not current_solution.is_valid(players):
        current_solution = LeagueSASolution(num_teams=num_teams, team_size=team_size, max_budget=max_budget)
    
    current_fitness = current_solution.fitness(players)
    best_solution = deepcopy(current_solution)
    best_fitness = current_fitness
    
    history = [current_fitness]
    temp = initial_temp
    iteration_count = 0

    if verbose:
        print(f"Initial SA solution fitness: {current_fitness}, Temp: {temp}")

    while temp > final_temp:
        for _ in range(iterations_per_temp):
            iteration_count += 1
            neighbor_solution = current_solution.get_random_neighbor(players)
            neighbor_fitness = neighbor_solution.fitness(players)
            delta_e = neighbor_fitness - current_fitness
            
            if delta_e < 0:
                current_solution = deepcopy(neighbor_solution)
                current_fitness = neighbor_fitness
                if current_fitness < best_fitness:
                    best_solution = deepcopy(current_solution)
                    best_fitness = current_fitness
            else:
                if temp > 1e-8: # Avoid division by zero if temp is very small
                    acceptance_probability = np.exp(-delta_e / temp)
                    if random.random() < acceptance_probability:
                        current_solution = deepcopy(neighbor_solution)
                        current_fitness = neighbor_fitness
            history.append(current_fitness)
            if verbose and iteration_count % (iterations_per_temp * 10) == 0: # Print less frequently
                print(f"Iter: {iteration_count}, Temp: {temp:.2f}, Current Fitness: {current_fitness:.4f}, Best Fitness: {best_fitness:.4f}")
        temp *= alpha

    if verbose:
        print(f"Simulated Annealing finished. Best fitness: {best_fitness}")
    return best_solution, best_fitness, history

