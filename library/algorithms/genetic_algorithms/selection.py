import random
from copy import deepcopy

from solution import Solution


def fitness_proportionate_selection(population: list[Solution], maximization: bool):
    if maximization:
        fitness_values = []
        for ind in population:
            if ind.fitness() < 0:
                # If fitness is negative (invalid solution like in Knapsack)
                # Set fitness to very small positive value
                # Probability of selecting this individual is nearly 0.
                fitness_values.append(0.0000001)
            else:
                fitness_values.append(ind.fitness())
    else:
        # Minimization: Use the inverse of the fitness value
        # Lower fitness should have higher probability of being selected
        fitness_values = [1 / ind.fitness() for ind in population]

    total_fitness = sum(fitness_values)
    # Generate random number between 0 and total
    random_nr = random.uniform(0, total_fitness)
    # For each individual, check if random number is inside the individual's "box"
    box_boundary = 0
    for ind_idx, ind in enumerate(population):
        box_boundary += fitness_values[ind_idx]
        if random_nr <= box_boundary:
            return deepcopy(ind)
