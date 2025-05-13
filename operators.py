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
        new_assign[i] = (new_assign[i] + shift) % 5 # Assuming 5 teams
    return LeagueSolution(new_assign)

def mutate_shuffle_team(solution):
    # This is the original version from the project files, kept for reference or if needed.
    # The adapted version is mutate_shuffle_within_team_constrained
    new_assign = solution.assignment[:]
    chosen_team = random.randint(0, 4) # Assuming 5 teams
    indices = [i for i, team in enumerate(new_assign) if team == chosen_team]
    shuffled = indices[:]
    random.shuffle(shuffled)
    # The following loop, as discussed, is a NO-OP for the current representation if it means
    # re-assigning players already in chosen_team back to chosen_team.
    # For it to be meaningful, it would imply shuffling player objects into fixed slots, or similar.
    for old, new in zip(indices, shuffled):
        new_assign[old] = chosen_team 
    return LeagueSolution(new_assign)

def mutate_swap_constrained(solution, players, max_attempts=10):
    """Performs a swap mutation between two players' team assignments.
    Ensures that the resulting assignment is valid. If not, it attempts
    a few more times before returning the original solution if no valid swap is found.
    """
    original_assignment = deepcopy(solution.assignment)
    num_players = len(solution.assignment)

    for attempt in range(max_attempts):
        new_assignment = deepcopy(original_assignment)
        idx1, idx2 = random.sample(range(num_players), 2)
        new_assignment[idx1], new_assignment[idx2] = new_assignment[idx2], new_assignment[idx1]
        mutated_solution = LeagueSolution(new_assignment, solution.num_teams, solution.team_size, solution.max_budget)
        if mutated_solution.is_valid(players):
            return mutated_solution
    return LeagueSolution(deepcopy(original_assignment), solution.num_teams, solution.team_size, solution.max_budget)

def mutate_targeted_player_exchange(solution, players, max_attempts=20):
    """Selects a player P1 from team T1 and P2 from a different team T2.
    Swaps P1 and P2. Checks if both T1 and T2 remain valid.
    If not, retries with different players/teams or reverts.
    """
    original_assignment = deepcopy(solution.assignment)
    num_players = len(solution.assignment)
    num_teams = solution.num_teams

    for attempt in range(max_attempts):
        new_assignment = deepcopy(original_assignment)
        if num_teams < 2:
            return LeagueSolution(deepcopy(original_assignment), solution.num_teams, solution.team_size, solution.max_budget)
        team_idx1, team_idx2 = random.sample(range(num_teams), 2)
        players_in_team1_indices = [i for i, t_id in enumerate(new_assignment) if t_id == team_idx1]
        players_in_team2_indices = [i for i, t_id in enumerate(new_assignment) if t_id == team_idx2]
        if not players_in_team1_indices or not players_in_team2_indices:
            continue
        player_idx_from_team1 = random.choice(players_in_team1_indices)
        player_idx_from_team2 = random.choice(players_in_team2_indices)
        new_assignment[player_idx_from_team1] = team_idx2
        new_assignment[player_idx_from_team2] = team_idx1
        mutated_solution = LeagueSolution(new_assignment, solution.num_teams, solution.team_size, solution.max_budget)
        if mutated_solution.is_valid(players):
            return mutated_solution
    return LeagueSolution(deepcopy(original_assignment), solution.num_teams, solution.team_size, solution.max_budget)

def mutate_shuffle_within_team_constrained(solution, players, max_attempts=20):
    """Adapts the idea of shuffling a team's composition.
    Selects a team, then attempts to swap one player from this team
    with one player from outside this team, ensuring the overall solution remains valid.
    """
    original_assignment = deepcopy(solution.assignment)
    num_players = len(solution.assignment)
    num_teams = solution.num_teams
    if num_teams == 0 or num_players == 0:
        return LeagueSolution(deepcopy(original_assignment), solution.num_teams, solution.team_size, solution.max_budget)
    for attempt in range(max_attempts):
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
        mutated_solution = LeagueSolution(new_assignment, solution.num_teams, solution.team_size, solution.max_budget)
        if mutated_solution.is_valid(players):
            return mutated_solution
    return LeagueSolution(deepcopy(original_assignment), solution.num_teams, solution.team_size, solution.max_budget)

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

def crossover_one_point_prefer_valid(parent1, parent2, players, max_attempts=10):
    """Performs one-point crossover. Attempts to find a valid child solution.
    If no valid child is found after max_attempts, returns the last generated child.
    """
    len_assignment = len(parent1.assignment)
    if len_assignment < 2:
        return LeagueSolution(deepcopy(parent1.assignment), parent1.num_teams, parent1.team_size, parent1.max_budget)
    last_child_solution = None
    for attempt in range(max_attempts):
        cut_point = random.randint(1, len_assignment - 1)
        child_assignment1 = parent1.assignment[:cut_point] + parent2.assignment[cut_point:]
        child_solution = LeagueSolution(child_assignment1, parent1.num_teams, parent1.team_size, parent1.max_budget)
        if child_solution.is_valid(players):
            return child_solution
        last_child_solution = child_solution
    if last_child_solution:
        return last_child_solution
    else:
        return LeagueSolution(deepcopy(parent1.assignment), parent1.num_teams, parent1.team_size, parent1.max_budget)

def crossover_uniform_prefer_valid(parent1, parent2, players, max_attempts=10):
    """Performs uniform crossover. Attempts to find a valid child solution.
    If no valid child is found after max_attempts, returns the last generated child.
    """
    len_assignment = len(parent1.assignment)
    last_child_solution = None
    for attempt in range(max_attempts):
        child_assignment = []
        for i in range(len_assignment):
            if random.random() < 0.5:
                child_assignment.append(parent1.assignment[i])
            else:
                child_assignment.append(parent2.assignment[i])
        child_solution = LeagueSolution(child_assignment, parent1.num_teams, parent1.team_size, parent1.max_budget)
        if child_solution.is_valid(players):
            return child_solution
        last_child_solution = child_solution
    if last_child_solution:
        return last_child_solution
    else:
        return LeagueSolution(deepcopy(parent1.assignment), parent1.num_teams, parent1.team_size, parent1.max_budget)

# SELECTION OPERATORS --------

def selection_tournament_variable_k(population, players, k):
    """Selects an individual using tournament selection with variable tournament size k."""
    if not population:
        raise ValueError("Population cannot be empty for tournament selection.")
    if k > len(population):
        k = len(population) # Adjust k if it's larger than population size
    if k <= 0:
        raise ValueError("Tournament size k must be positive.")
        
    selected_tournament_individuals = random.sample(population, k)
    # Assuming minimization problem (lower fitness is better)
    winner = min(selected_tournament_individuals, key=lambda sol: sol.fitness(players))
    return winner

def selection_ranking(population, players):
    """Selects an individual using ranking selection."""
    if not population:
        raise ValueError("Population cannot be empty for ranking selection.")
        
    # Sort population by fitness (ascending for minimization)
    sorted_pop = sorted(population, key=lambda s: s.fitness(players))
    
    # Assign ranks (worst fitness gets rank 1, best gets rank N)
    # For selection probability, we want best fitness (lowest rank value here if sorted ascendingly)
    # to have highest probability. So, if ranks are 0, 1, ..., N-1 for best to worst fitness,
    # probabilities should be proportional to N-rank.
    # The original implementation used ranks 1 to N, with best getting highest probability.
    # Let's stick to that for consistency if it worked.
    # Ranks: 1 (best) to N (worst) if sorted_pop is best to worst.
    # Probabilities: proportional to (N - rank + 1) or simply use inverted ranks for weights.

    # Original logic: ranks = list(range(1, len(sorted_pop)+1)), probs = [r / total for r in ranks[::-1]]
    # This means sorted_pop[0] (best fitness) gets rank 1. ranks[::-1] means [N, N-1, ..., 1].
    # So, best individual (sorted_pop[0]) gets weight N, worst (sorted_pop[N-1]) gets weight 1.
    # This is correct for giving higher probability to better fitness.
    
    ranks = list(range(len(sorted_pop), 0, -1)) # Ranks from N down to 1 (N for best, 1 for worst)
    total_rank_sum = sum(ranks)
    
    if total_rank_sum == 0: # Handles case of single individual or all same fitness if ranks are weird
        return random.choice(sorted_pop)
        
    probabilities = [rank / total_rank_sum for rank in ranks]
    
    # random.choices returns a list, so we take the first element
    return random.choices(sorted_pop, weights=probabilities, k=1)[0]

