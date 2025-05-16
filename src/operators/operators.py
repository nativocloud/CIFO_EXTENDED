import random
import numpy as np
from copy import deepcopy
from src.solution.solution import LeagueSolution

# MUTATION OPERATORS --------

def mutate_swap(solution, players_data):
    new_assign = solution.assignment[:]
    i, j = random.sample(range(len(new_assign)), 2)
    new_assign[i], new_assign[j] = new_assign[j], new_assign[i]
    # Refactored to correctly pass players_data as the first argument
    return LeagueSolution(players_data, assignment=new_assign, num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)

def mutate_team_shift(solution, players_data):
    new_assign = solution.assignment[:]
    shift = random.randint(1, solution.num_teams -1) if solution.num_teams > 1 else 0 # Avoid modulo by zero or no shift if 1 team
    if solution.num_teams > 0 : # Ensure num_teams is positive before modulo
        for i in range(len(new_assign)):
            new_assign[i] = (new_assign[i] + shift) % solution.num_teams
    # Refactored to correctly pass players_data as the first argument
    return LeagueSolution(players_data, assignment=new_assign, num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)

def mutate_shuffle_team(solution, players_data):
    new_assign = solution.assignment[:]
    if solution.num_teams == 0:
        # If there are no teams, no shuffling can occur, return a copy or the original.
        return LeagueSolution(players_data, assignment=deepcopy(new_assign), num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)

    chosen_team_id_to_shuffle = random.randint(0, solution.num_teams - 1)
    
    # Get indices of players in the chosen team
    player_indices_in_chosen_team = [i for i, team_id_player in enumerate(new_assign) if team_id_player == chosen_team_id_to_shuffle]
    
    # Get indices of players NOT in the chosen team (these are the potential swap targets for players from the chosen team)
    # This interpretation of "shuffle team" seems to imply reassigning players *from* the chosen team to *other* teams randomly,
    # or swapping players *within* the team if that was the intent. The original logic was:
    # indices = [i for i, team in enumerate(new_assign) if team == chosen_team]
    # shuffled = indices[:] # This copies the indices of players IN the chosen team
    # random.shuffle(shuffled) # This shuffles the order of those indices
    # for old, new in zip(indices, shuffled):
    #    new_assign[old] = chosen_team # This reassigns players from the chosen team... back to the chosen team, using their original indices but in a shuffled order.
                                      # This doesn't actually change team assignments, only the order of players *within* that team list if it were reconstructed.
                                      # The assignment vector itself would not change if players are just re-ordered within the same team.
    # A more meaningful shuffle might be to re-assign players *from* the chosen team to *any* team randomly, or swap players *within* the team.
    # Given the name "shuffle_team", let's assume it means re-assigning players currently in `chosen_team_id_to_shuffle` to *any* valid team slot randomly.
    # This is complex to do while preserving validity easily. 
    # A simpler interpretation, more aligned with other mutations, is to pick a player from the chosen team and swap them with a player from another team.
    # Or, pick two players within the chosen team and swap their roles (if that were part of the representation, but it's not).

    # Re-interpreting based on common mutation types: let's make this a random reassignment of one player from the chosen team.
    if player_indices_in_chosen_team:
        player_to_reassign_idx = random.choice(player_indices_in_chosen_team)
        new_team_for_player = random.randint(0, solution.num_teams - 1)
        new_assign[player_to_reassign_idx] = new_team_for_player # This is a simple reassignment, likely to break validity.

    # The original logic `new_assign[old] = chosen_team` for `old` in `indices` (players in chosen_team) does not change the assignment vector.
    # To make it a meaningful mutation that *might* change the solution towards a new valid one (or be used by prefer_valid operators):
    # Let's stick to a simple random change: pick one player and assign to a random team.
    # This operator is inherently unconstrained and likely to produce invalid solutions often.
    if len(new_assign) > 0:
        player_to_change_idx = random.randrange(len(new_assign))
        new_team_id = random.randrange(solution.num_teams) if solution.num_teams > 0 else 0
        new_assign[player_to_change_idx] = new_team_id

    # Refactored to correctly pass players_data as the first argument
    return LeagueSolution(players_data, assignment=new_assign, num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)


def mutate_swap_constrained(solution, players_data, max_attempts=10):
    original_assignment = deepcopy(solution.assignment)
    num_players = len(solution.assignment)

    for _attempt in range(max_attempts):
        new_assignment = deepcopy(original_assignment)
        idx1, idx2 = random.sample(range(num_players), 2)
        new_assignment[idx1], new_assignment[idx2] = new_assignment[idx2], new_assignment[idx1]
        mutated_solution = LeagueSolution(players_data, assignment=new_assignment, num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)
        if mutated_solution.is_valid(): # MODIFIED: Vectorized call
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
        if mutated_solution.is_valid(): # MODIFIED: Vectorized call
            return mutated_solution
    return LeagueSolution(players_data, assignment=deepcopy(original_assignment), num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)

def mutate_shuffle_within_team_constrained(solution, players_data, max_attempts=20):
    # This name is a bit misleading if it means swapping a player from the team with one outside.
    # If it means shuffling players *within* a team, it wouldn't change the assignment vector.
    # The implementation suggests swapping a player from a chosen team with a player from another team.
    # This is similar to mutate_targeted_player_exchange but with one player fixed to a chosen_team_id.
    original_assignment = deepcopy(solution.assignment)
    num_players = len(solution.assignment)
    num_teams = solution.num_teams
    if num_teams == 0 or num_players == 0:
        return LeagueSolution(players_data, assignment=deepcopy(original_assignment), num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)
    for _attempt in range(max_attempts):
        new_assignment = deepcopy(original_assignment)
        if num_teams < 2: # Need at least two teams for a meaningful exchange between teams
             return LeagueSolution(players_data, assignment=deepcopy(original_assignment), num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)

        chosen_team_id = random.randint(0, num_teams - 1)
        player_indices_in_chosen_team = [i for i, t_id in enumerate(new_assignment) if t_id == chosen_team_id]
        # Ensure there are players in the chosen team
        if not player_indices_in_chosen_team:
            continue

        # Select a player from the chosen team
        p1_idx = random.choice(player_indices_in_chosen_team)
        
        # Select a player from any *other* team
        other_teams_indices = [i for i in range(num_teams) if i != chosen_team_id]
        if not other_teams_indices:
            continue # Only one team, no exchange possible
        
        target_other_team_id = random.choice(other_teams_indices)
        player_indices_in_other_team = [i for i, t_id in enumerate(new_assignment) if t_id == target_other_team_id]

        if not player_indices_in_other_team:
            continue # Target other team is empty

        p2_idx = random.choice(player_indices_in_other_team)

        # Swap these two players' team assignments
        new_assignment[p1_idx] = target_other_team_id 
        new_assignment[p2_idx] = chosen_team_id
        
        mutated_solution = LeagueSolution(players_data, assignment=new_assignment, num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)
        if mutated_solution.is_valid(): # MODIFIED: Vectorized call
            return mutated_solution
    return LeagueSolution(players_data, assignment=deepcopy(original_assignment), num_teams=solution.num_teams, team_size=solution.team_size, max_budget=solution.max_budget)

# CROSSOVER OPERATORS --------

def crossover_one_point(parent1, parent2, players_data):
    len_assignment = len(parent1.assignment)
    if len_assignment < 2:
        # If assignment length is less than 2, crossover is not meaningful or possible.
        # Return a copy of parent1 as a fallback.
        return LeagueSolution(players_data, assignment=deepcopy(parent1.assignment), num_teams=parent1.num_teams, team_size=parent1.team_size, max_budget=parent1.max_budget)
    cut = random.randint(1, len_assignment - 1) # Ensure cut is not at the very beginning or end for a two-part crossover
    child_assign = np.concatenate((parent1.assignment[:cut], parent2.assignment[cut:])) # MODIFIED: Use np.concatenate for NumPy arrays
    # Refactored to correctly pass players_data as the first argument
    return LeagueSolution(players_data, assignment=child_assign, num_teams=parent1.num_teams, team_size=parent1.team_size, max_budget=parent1.max_budget)

def crossover_uniform(parent1, parent2, players_data):
    child_assign = [
        parent1.assignment[i] if random.random() < 0.5 else parent2.assignment[i]
        for i in range(len(parent1.assignment))
    ]
    # Refactored to correctly pass players_data as the first argument
    return LeagueSolution(players_data, assignment=child_assign, num_teams=parent1.num_teams, team_size=parent1.team_size, max_budget=parent1.max_budget)

def crossover_one_point_prefer_valid(parent1, parent2, players_data, max_attempts=10):
    len_assignment = len(parent1.assignment)
    if len_assignment < 2:
        return LeagueSolution(players_data, assignment=deepcopy(parent1.assignment), num_teams=parent1.num_teams, team_size=parent1.team_size, max_budget=parent1.max_budget)
    last_child_solution = None
    for _attempt in range(max_attempts):
        cut_point = random.randint(1, len_assignment - 1)
        child_assignment1 = np.concatenate((parent1.assignment[:cut_point], parent2.assignment[cut_point:])) # MODIFIED: Use np.concatenate for NumPy arrays
        child_solution = LeagueSolution(players_data, assignment=child_assignment1, num_teams=parent1.num_teams, team_size=parent1.team_size, max_budget=parent1.max_budget)
        if child_solution.is_valid(): # MODIFIED: Vectorized call
            return child_solution
        last_child_solution = child_solution # Keep the last generated (possibly invalid) child
    # If no valid child found after attempts, return the last generated one or a copy of parent1
    return last_child_solution if last_child_solution else LeagueSolution(players_data, assignment=deepcopy(parent1.assignment), num_teams=parent1.num_teams, team_size=parent1.team_size, max_budget=parent1.max_budget)

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
        if child_solution.is_valid(): # MODIFIED: Vectorized call
            return child_solution
        last_child_solution = child_solution # Keep the last generated (possibly invalid) child
    # If no valid child found after attempts, return the last generated one or a copy of parent1
    return last_child_solution if last_child_solution else LeagueSolution(players_data, assignment=deepcopy(parent1.assignment), num_teams=parent1.num_teams, team_size=parent1.team_size, max_budget=parent1.max_budget)

# SELECTION OPERATORS --------

def selection_tournament_variable_k(population, players_data, k):
    if not population:
        raise ValueError("Population cannot be empty for tournament selection.")
    # Ensure k is not larger than population size
    actual_k = min(k, len(population))
    if actual_k <= 0:
        # Fallback if k is 0 or population is too small, though k should be positive.
        # Returning a random choice might be one way, or raising error earlier for k.
        # For now, let's assume k is validated to be > 0 before this point if possible.
        # If k becomes 0 due to small population, pick randomly from what's available.
        return random.choice(population) if population else None 
        
    selected_tournament_individuals = random.sample(population, actual_k)
    winner = min(selected_tournament_individuals, key=lambda sol: sol.fitness())
    return winner

def selection_ranking(population, players_data):
    if not population:
        raise ValueError("Population cannot be empty for ranking selection.")
        
    sorted_pop = sorted(population, key=lambda s: s.fitness()) # MODIFIED: Vectorized call
    ranks = list(range(len(sorted_pop), 0, -1)) # Higher rank for better fitness (lower value)
    total_rank_sum = sum(ranks)
    
    if total_rank_sum == 0: # Should only happen if population is empty or all ranks are 0 (not possible with range > 0)
        return random.choice(sorted_pop) # Fallback for empty or problematic ranks
        
    probabilities = [rank / total_rank_sum for rank in ranks]
    return random.choices(sorted_pop, weights=probabilities, k=1)[0]

def selection_boltzmann(population, players_data, temperature, k=1):
    if not population:
        raise ValueError("Population cannot be empty for Boltzmann selection.")
    if temperature <= 1e-6: # Avoid issues with zero or very small temperature
        # At very low temperatures, Boltzmann selection behaves like best selection.
        # Return the best k individuals.
        sorted_pop = sorted(population, key=lambda s: s.fitness()) # MODIFIED: Vectorized call
        return sorted_pop[:k] if k > 1 else sorted_pop[0]

    fitness_values = np.array([sol.fitness() for sol in population]) # MODIFIED: Vectorized call
    
    # Handle potential inf values in fitness - assign them very low probability
    # We are minimizing, so exp(-inf/T) -> 0, exp(+inf/T) -> inf (problematic)
    # If fitness is inf, it's a bad solution. We want its probability to be near zero.
    # exp(-fitness/temp): if fitness is inf, exp(-inf) is 0. This is correct.
    probabilities_unnormalized = np.exp(-fitness_values / temperature)
    
    # Replace NaNs or Infs in probabilities_unnormalized (e.g. from exp(-(-inf))) with 0
    probabilities_unnormalized[np.isnan(probabilities_unnormalized)] = 0
    probabilities_unnormalized[np.isinf(probabilities_unnormalized)] = 0 # Should catch exp(inf)

    sum_probs = np.sum(probabilities_unnormalized)

    if sum_probs == 0 or np.isnan(sum_probs) or np.isinf(sum_probs):
        # If all probabilities are zero (e.g., all fitnesses were inf, or temp too high with large fitnesses)
        # Fallback to random choice or best choice. For diversity, random choice might be better here.
        # Or, if all fitnesses are inf, any choice is equally bad.
        # If only some are inf, they correctly get 0 probability.
        # This case implies all are effectively zero probability.
        return random.choices(population, k=k) if population else (None if k==1 else [])
        
    probabilities = probabilities_unnormalized / sum_probs
    # Ensure probabilities sum to 1 after normalization, handling potential floating point inaccuracies
    probabilities = probabilities / np.sum(probabilities) 
    
    chosen_individuals = random.choices(population, weights=probabilities, k=k)
    return chosen_individuals if k > 1 else chosen_individuals[0]

