import random
import numpy as np
from copy import deepcopy
from solution import LeagueSolution

# MUTATION OPERATORS --------

# Note: The following unconstrained operators (mutate_swap, mutate_team_shift, mutate_shuffle_team)
# do not use the new LeagueSolution constructor signature correctly (missing players_data).
# They are not currently used by the GA configurations that caused errors, but would need
# refactoring if they were to be used with the updated LeagueSolution.

def mutate_swap(solution, players_data): # Added players_data for consistency, though not used by old constructor call
    new_assign = solution.assignment[:]
    i, j = random.sample(range(len(new_assign)), 2)
    new_assign[i], new_assign[j] = new_assign[j], new_assign[i]
    # This instantiation is still problematic with the new LeagueSolution constructor
    # It needs players_data as the first argument.
    # For now, assuming these are not the ones causing the immediate error.
    # If used, this would need: LeagueSolution(players_data, assignment=new_assign, ...)
    return LeagueSolution(solution.players_data_ref, assignment=new_assign, num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)

def mutate_team_shift(solution, players_data): # Added players_data
    new_assign = solution.assignment[:]
    shift = random.randint(1, 4)
    for i in range(len(new_assign)):
        new_assign[i] = (new_assign[i] + shift) % solution.num_teams 
    return LeagueSolution(solution.players_data_ref, assignment=new_assign, num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)

def mutate_shuffle_team(solution, players_data): # Added players_data
    new_assign = solution.assignment[:]
    chosen_team = random.randint(0, solution.num_teams - 1)
    indices = [i for i, team in enumerate(new_assign) if team == chosen_team]
    shuffled = indices[:]
    random.shuffle(shuffled)
    for old, new in zip(indices, shuffled):
        new_assign[old] = chosen_team 
    return LeagueSolution(solution.players_data_ref, assignment=new_assign, num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)

def mutate_swap_constrained(solution, players_data, max_attempts=10):
    original_assignment = deepcopy(solution.assignment)
    num_players = len(solution.assignment)

    for _attempt in range(max_attempts):
        new_assignment = deepcopy(original_assignment)
        idx1, idx2 = random.sample(range(num_players), 2)
        new_assignment[idx1], new_assignment[idx2] = new_assignment[idx2], new_assignment[idx1]
        mutated_solution = LeagueSolution(players_data, assignment=new_assignment, num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)
        if mutated_solution.is_valid(players_data):
            return mutated_solution
    return LeagueSolution(players_data, assignment=deepcopy(original_assignment), num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)

def mutate_targeted_player_exchange(solution, players_data, max_attempts=20):
    original_assignment = deepcopy(solution.assignment)
    num_players = len(solution.assignment)
    num_teams = solution.num_teams

    for _attempt in range(max_attempts):
        new_assignment = deepcopy(original_assignment)
        if num_teams < 2:
            return LeagueSolution(players_data, assignment=deepcopy(original_assignment), num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)
        team_idx1, team_idx2 = random.sample(range(num_teams), 2)
        players_in_team1_indices = [i for i, t_id in enumerate(new_assignment) if t_id == team_idx1]
        players_in_team2_indices = [i for i, t_id in enumerate(new_assignment) if t_id == team_idx2]
        if not players_in_team1_indices or not players_in_team2_indices:
            continue
        player_idx_from_team1 = random.choice(players_in_team1_indices)
        player_idx_from_team2 = random.choice(players_in_team2_indices)
        new_assignment[player_idx_from_team1] = team_idx2
        new_assignment[player_idx_from_team2] = team_idx1
        mutated_solution = LeagueSolution(players_data, assignment=new_assignment, num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)
        if mutated_solution.is_valid(players_data):
            return mutated_solution
    return LeagueSolution(players_data, assignment=deepcopy(original_assignment), num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)

def mutate_shuffle_within_team_constrained(solution, players_data, max_attempts=20):
    original_assignment = deepcopy(solution.assignment)
    num_players = len(solution.assignment)
    num_teams = solution.num_teams
    if num_teams == 0 or num_players == 0:
        return LeagueSolution(players_data, assignment=deepcopy(original_assignment), num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)
    for _attempt in range(max_attempts):
        new_assignment = deepcopy(original_assignment)
        chosen_team_id = random.randint(0, num_teams - 1)
        player_indices_in_chosen_team = [i for i, t_id in enumerate(new_assignment) if t_id == chosen_team_id]
        player_indices_not_in_chosen_team = [i for i, t_id in enumerate(new_assignment) if t_id != chosen_team_id]
        if not player_indices_in_chosen_team or not player_indices_not_in_chosen_team:
            continue
        p1_idx = random.choice(player_indices_in_chosen_team)
        p2_idx = random.choice(player_indices_not_in_chosen_team)
        p2_original_team_id = new_assignment[p2_idx]
        new_assignment[p1_idx] = p2_original_team_id
        new_assignment[p2_idx] = chosen_team_id
        mutated_solution = LeagueSolution(players_data, assignment=new_assignment, num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)
        if mutated_solution.is_valid(players_data):
            return mutated_solution
    return LeagueSolution(players_data, assignment=deepcopy(original_assignment), num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)

# CROSSOVER OPERATORS --------

# Note: The following unconstrained operators (crossover_one_point, crossover_uniform)
# also do not use the new LeagueSolution constructor signature correctly.
# They are not currently used by the GA configurations that caused errors.

def crossover_one_point(parent1, parent2, players_data): # Added players_data
    cut = random.randint(1, len(parent1.assignment) - 2)
    child_assign = parent1.assignment[:cut] + parent2.assignment[cut:]
    return LeagueSolution(players_data, assignment=child_assign, num_teams=parent1.num_teams, team_size=parent1.team_size, max_budget=parent1.max_budget)

def crossover_uniform(parent1, parent2, players_data): # Added players_data
    child_assign = [
        parent1.assignment[i] if random.random() < 0.5 else parent2.assignment[i]
        for i in range(len(parent1.assignment))
    ]
    return LeagueSolution(players_data, assignment=child_assign, num_teams=parent1.num_teams, team_size=parent1.team_size, max_budget=parent1.max_budget)

def crossover_one_point_prefer_valid(parent1, parent2, players_data, max_attempts=10):
    len_assignment = len(parent1.assignment)
    if len_assignment < 2:
        return LeagueSolution(players_data, assignment=deepcopy(parent1.assignment), num_teams=parent1.num_teams, team_size=parent1.team_size, max_budget=parent1.max_budget)
    last_child_solution = None
    for _attempt in range(max_attempts):
        cut_point = random.randint(1, len_assignment - 1)
        child_assignment1 = parent1.assignment[:cut_point] + parent2.assignment[cut_point:]
        child_solution = LeagueSolution(players_data, assignment=child_assignment1, num_teams=parent1.num_teams, team_size=parent1.team_size, max_budget=parent1.max_budget)
        if child_solution.is_valid(players_data):
            return child_solution
        last_child_solution = child_solution
    if last_child_solution:
        return last_child_solution
    else:
        return LeagueSolution(players_data, assignment=deepcopy(parent1.assignment), num_teams=parent1.num_teams, team_size=parent1.team_size, max_budget=parent1.max_budget)

def crossover_uniform_prefer_valid(parent1, parent2, players_data, max_attempts=10):
    len_assignment = len(parent1.assignment)
    last_child_solution = None
    for _attempt in range(max_attempts):
        child_assignment = []
        for i in range(len_assignment):
            if random.random() < 0.5:
                child_assignment.append(parent1.assignment[i])
            else:
                child_assignment.append(parent2.assignment[i])
        child_solution = LeagueSolution(players_data, assignment=child_assignment, num_teams=parent1.num_teams, team_size=parent1.team_size, max_budget=parent1.max_budget)
        if child_solution.is_valid(players_data):
            return child_solution
        last_child_solution = child_solution
    if last_child_solution:
        return last_child_solution
    else:
        return LeagueSolution(players_data, assignment=deepcopy(parent1.assignment), num_teams=parent1.num_teams, team_size=parent1.team_size, max_budget=parent1.max_budget)

# SELECTION OPERATORS --------

def selection_tournament_variable_k(population, players_data, k):
    if not population:
        raise ValueError("Population cannot be empty for tournament selection.")
    if k > len(population):
        k = len(population)
    if k <= 0:
        raise ValueError("Tournament size k must be positive.")
        
    selected_tournament_individuals = random.sample(population, k)
    winner = min(selected_tournament_individuals, key=lambda sol: sol.fitness(players_data))
    return winner

def selection_ranking(population, players_data):
    if not population:
        raise ValueError("Population cannot be empty for ranking selection.")
        
    sorted_pop = sorted(population, key=lambda s: s.fitness(players_data))
    ranks = list(range(len(sorted_pop), 0, -1))
    total_rank_sum = sum(ranks)
    
    if total_rank_sum == 0:
        return random.choice(sorted_pop)
        
    probabilities = [rank / total_rank_sum for rank in ranks]
    return random.choices(sorted_pop, weights=probabilities, k=1)[0]

def selection_boltzmann(population, players_data, temperature, k=1):
    if not population:
        raise ValueError("Population cannot be empty for Boltzmann selection.")
    if temperature <= 0:
        raise ValueError("Temperature must be positive for Boltzmann selection.")

    fitness_values = np.array([sol.fitness(players_data) for sol in population])
    valid_fitness_values = np.array([f for f in fitness_values if f != float("inf")])
    if len(valid_fitness_values) == 0:
        return random.choices(population, k=k)
        
    probabilities_unnormalized = np.exp(-fitness_values / temperature)
    probabilities_unnormalized[np.isinf(probabilities_unnormalized)] = 0
    probabilities_unnormalized[np.isnan(probabilities_unnormalized)] = 0

    sum_probs = np.sum(probabilities_unnormalized)

    if sum_probs == 0 or np.isnan(sum_probs) or np.isinf(sum_probs):
        if not population:
             raise ValueError("Population is empty, cannot select.")
        return random.choices(population, k=k)
        
    probabilities = probabilities_unnormalized / sum_probs
    probabilities = probabilities / np.sum(probabilities)
    chosen_individuals = random.choices(population, weights=probabilities, k=k)
    return chosen_individuals if k > 1 else chosen_individuals[0]

