import random
from operators import (
    mutate_swap,
    mutate_team_shift,
    mutate_shuffle_team,
    crossover_one_point,
    crossover_uniform,
    selection_tournament,
    selection_ranking
)
from solution import LeagueSolution, LeagueHillClimbingSolution

# Map operator names to functions for flexible configuration
MUTATIONS = [mutate_swap, mutate_team_shift, mutate_shuffle_team]
CROSSOVERS = [crossover_one_point, crossover_uniform]
SELECTIONS = [selection_tournament, selection_ranking]

def generate_population(players, size):
    population = []
    while len(population) < size:
        candidate = LeagueSolution()
        if candidate.is_valid(players):
            population.append(candidate)
    return population

def genetic_algorithm(
    players,
    population_size=50,
    generations=30,
    mutation_rate=0.2,
    elite_size=5,
    mutation_operator=mutate_swap,
    crossover_operator=crossover_one_point,
    selection_operator=selection_tournament
):
    population = generate_population(players, population_size)
    history = []
    best_solution = min(population, key=lambda s: s.fitness(players))

    for gen in range(generations):
        new_population = []
        population.sort(key=lambda x: x.fitness(players))
        new_population.extend(population[:elite_size])

        while len(new_population) < population_size:
            # No loop principal do GA
            parent1 = selection_operator(population, players)
            parent2 = selection_operator(population, players)
            
            #parent1 = selection_operator(population)
            #parent2 = selection_operator(population)
            child = crossover_operator(parent1, parent2)

            if random.random() < mutation_rate:
                child = mutation_operator(child)

            if child.is_valid(players):
                new_population.append(child)

        population = new_population
        current_best = min(population, key=lambda s: s.fitness(players))
        if current_best.fitness(players) < best_solution.fitness(players):
            best_solution = current_best
        history.append(best_solution.fitness(players))

    return best_solution, history


def hill_climbing(players, max_iterations=1000, verbose=False):
    current = LeagueHillClimbingSolution()
    while not current.is_valid(players):
        current = LeagueHillClimbingSolution()

    current_fitness = current.fitness(players)
    history = [current_fitness]

    for iteration in range(max_iterations):
        neighbors = current.get_neighbors(players)
        if not neighbors:
            break

        neighbor = min(neighbors, key=lambda x: x.fitness(players))
        neighbor_fitness = neighbor.fitness(players)

        if neighbor_fitness < current_fitness:
            current = neighbor
            current_fitness = neighbor_fitness
            history.append(current_fitness)
            if verbose:
                print(f"Iteration {iteration}: fitness = {current_fitness}")
        else:
            break

    return current, current_fitness, history



def simulated_annealing_for_league(
    players,
    initial_temp=1000,
    final_temp=1,
    alpha=0.99,
    iterations_per_temp=100,
    verbose=False
):
    """Simulated Annealing for the Sports League Optimization problem."""
    
    # Initialize solution - ensure it's valid from the start
    current_solution = LeagueSASolution()
    while not current_solution.is_valid(players):
        current_solution = LeagueSASolution()
    
    current_fitness = current_solution.fitness(players)
    best_solution = deepcopy(current_solution)
    best_fitness = current_fitness
    
    history = [current_fitness]
    temp = initial_temp
    
    iteration_count = 0

    if verbose:
        print(f"Initial SA solution fitness: {current_fitness}")

    while temp > final_temp:
        for _ in range(iterations_per_temp):
            iteration_count += 1
            # Generate a random neighbor that is guaranteed to be valid by LeagueSASolution.get_random_neighbor
            neighbor_solution = current_solution.get_random_neighbor(players)
            neighbor_fitness = neighbor_solution.fitness(players)
            
            # Calculate energy difference (delta E)
            # For minimization, if neighbor_fitness < current_fitness, delta_e is negative (good)
            delta_e = neighbor_fitness - current_fitness
            
            if delta_e < 0: # Neighbor is better, always accept
                current_solution = deepcopy(neighbor_solution)
                current_fitness = neighbor_fitness
                if current_fitness < best_fitness:
                    best_solution = deepcopy(current_solution)
                    best_fitness = current_fitness
            else: # Neighbor is worse, accept with a probability
                if temp > 1e-6: # Avoid division by zero if temp is too small
                    acceptance_probability = np.exp(-delta_e / temp)
                    if random.random() < acceptance_probability:
                        current_solution = deepcopy(neighbor_solution)
                        current_fitness = neighbor_fitness
            
            history.append(current_fitness) # Record current fitness at each step
            
            if verbose and iteration_count % 50 == 0:
                print(f"Iter: {iteration_count}, Temp: {temp:.2f}, Current Fitness: {current_fitness:.4f}, Best Fitness: {best_fitness:.4f}")
        
        temp *= alpha # Cool down

    if verbose:
        print(f"Simulated Annealing finished. Best fitness: {best_fitness}")
        
    return best_solution, best_fitness, history

