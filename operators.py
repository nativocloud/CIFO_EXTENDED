import random
import numpy as np
from copy import deepcopy
from solution import LeagueSolution

# MUTATION OPERATORS --------

def mutate_swap(solution):
    new_assign = solution.assignment[:]
    i, j = random.sample(range(len(new_assign)), 2)
    new_assign[i], new_assign[j] = new_assign[j], new_assign[i]
    return LeagueSolution(new_assign)

def mutate_team_shift(solution):
    new_assign = solution.assignment[:]
    shift = random.randint(1, 4)
    for i in range(len(new_assign)):
        new_assign[i] = (new_assign[i] + shift) % 5
    return LeagueSolution(new_assign)

def mutate_shuffle_team(solution):
    new_assign = solution.assignment[:]
    chosen_team = random.randint(0, 4)
    indices = [i for i, team in enumerate(new_assign) if team == chosen_team]
    shuffled = indices[:]
    random.shuffle(shuffled)
    for old, new in zip(indices, shuffled):
        new_assign[old] = chosen_team
    return LeagueSolution(new_assign)

# CROSSOVER OPERATORS --------

def crossover_one_point(parent1, parent2):
    cut = random.randint(1, len(parent1.assignment) - 2)
    child_assign = parent1.assignment[:cut] + parent2.assignment[cut:]
    return LeagueSolution(child_assign)

def crossover_uniform(parent1, parent2):
    child_assign = [
        parent1.assignment[i] if random.random() < 0.5 else parent2.assignment[i]
        for i in range(len(parent1.assignment))
    ]
    return LeagueSolution(child_assign)

# SELECTION OPERATORS --------

def selection_tournament(population, players, k=3):
    selected = random.sample(population, k)
    selected.sort(key=lambda sol: sol.fitness(players)) 
    return selected[0]

def selection_ranking(population, players):
    sorted_pop = sorted(population, key=lambda s: s.fitness(players)) 
    ranks = list(range(1, len(sorted_pop)+1))
    total = sum(ranks)
    probs = [r / total for r in ranks[::-1]]  # Best gets highest prob
    return random.choices(sorted_pop, weights=probs, k=1)[0]
