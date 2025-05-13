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



def mutate_swap_constrained(solution, players, max_attempts=10):
    """Performs a swap mutation between two players' team assignments.
    Ensures that the resulting assignment is valid. If not, it attempts
    a few more times before returning the original solution if no valid swap is found.
    """
    original_assignment = deepcopy(solution.assignment)
    num_players = len(solution.assignment)

    for attempt in range(max_attempts):
        # Create a new assignment from the original for each attempt
        new_assignment = deepcopy(original_assignment)
        
        # Select two distinct random player indices
        idx1, idx2 = random.sample(range(num_players), 2)
        
        # Swap their team assignments
        new_assignment[idx1], new_assignment[idx2] = new_assignment[idx2], new_assignment[idx1]
        
        # Create a new solution object with the new assignment
        mutated_solution = LeagueSolution(new_assignment, solution.num_teams, solution.team_size, solution.max_budget)
        
        # Check if the mutated solution is valid
        if mutated_solution.is_valid(players):
            return mutated_solution # Return the valid mutated solution
            
    # If no valid swap is found after max_attempts, return the original solution (or a copy)
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

        # 1. Select two different teams
        if num_teams < 2:
            return LeagueSolution(deepcopy(original_assignment), solution.num_teams, solution.team_size, solution.max_budget) # Not enough teams to swap between
        
        team_idx1, team_idx2 = random.sample(range(num_teams), 2)

        # 2. Get players from these teams
        players_in_team1_indices = [i for i, t_id in enumerate(new_assignment) if t_id == team_idx1]
        players_in_team2_indices = [i for i, t_id in enumerate(new_assignment) if t_id == team_idx2]

        if not players_in_team1_indices or not players_in_team2_indices:
            continue # Should not happen if teams are populated

        # 3. Select one player from each team to swap
        player_idx_from_team1 = random.choice(players_in_team1_indices)
        player_idx_from_team2 = random.choice(players_in_team2_indices)

        # 4. Perform the swap in the new_assignment
        # Player from team1 (now at player_idx_from_team1) goes to team2
        # Player from team2 (now at player_idx_from_team2) goes to team1
        new_assignment[player_idx_from_team1] = team_idx2
        new_assignment[player_idx_from_team2] = team_idx1
        
        mutated_solution = LeagueSolution(new_assignment, solution.num_teams, solution.team_size, solution.max_budget)
        
        # 5. Check validity
        # This is a full validity check. A more optimized check would only validate team_idx1 and team_idx2.
        # However, the current LeagueSolution.is_valid() checks all teams.
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

        # 1. Pick a random team to modify
        chosen_team_id = random.randint(0, num_teams - 1)

        # 2. Identify players in the chosen team and players not in it
        player_indices_in_chosen_team = [i for i, t_id in enumerate(new_assignment) if t_id == chosen_team_id]
        player_indices_not_in_chosen_team = [i for i, t_id in enumerate(new_assignment) if t_id != chosen_team_id]

        if not player_indices_in_chosen_team or not player_indices_not_in_chosen_team:
            continue # Not possible to perform a swap

        # 3. Select one player from the chosen team (p1_idx)
        p1_idx = random.choice(player_indices_in_chosen_team)
        
        # 4. Select one player not in the chosen team (p2_idx)
        p2_idx = random.choice(player_indices_not_in_chosen_team)

        # 5. Perform the swap: p1 (from chosen_team) goes to p2's original team, p2 goes to chosen_team.
        p2_original_team_id = new_assignment[p2_idx] # Team p2 was in
        
        new_assignment[p1_idx] = p2_original_team_id # p1 now assigned to p2's old team
        new_assignment[p2_idx] = chosen_team_id    # p2 now assigned to the chosen_team
        
        mutated_solution = LeagueSolution(new_assignment, solution.num_teams, solution.team_size, solution.max_budget)
        
        if mutated_solution.is_valid(players):
            return mutated_solution
            
    return LeagueSolution(deepcopy(original_assignment), solution.num_teams, solution.team_size, solution.max_budget)



def crossover_one_point_prefer_valid(parent1, parent2, players, max_attempts=10):
    """Performs one-point crossover. Attempts to find a valid child solution.
    If no valid child is found after max_attempts, returns the last generated child.
    """
    len_assignment = len(parent1.assignment)
    if len_assignment < 2:
        # Not enough points to cut, return a copy of a parent (e.g., parent1)
        return LeagueSolution(deepcopy(parent1.assignment), parent1.num_teams, parent1.team_size, parent1.max_budget)

    last_child_solution = None

    for attempt in range(max_attempts):
        # Choose a random cut point (ensuring it's not at the very beginning or end)
        cut_point = random.randint(1, len_assignment - 1)
        
        child_assignment1 = parent1.assignment[:cut_point] + parent2.assignment[cut_point:]
        # child_assignment2 = parent2.assignment[:cut_point] + parent1.assignment[cut_point:] # Can also generate the second child
        
        # For now, we focus on generating one child per call, as is common
        child_solution = LeagueSolution(child_assignment1, parent1.num_teams, parent1.team_size, parent1.max_budget)
        
        if child_solution.is_valid(players):
            return child_solution # Return the first valid child found
        
        last_child_solution = child_solution # Keep track of the last generated one

    # If no valid child found after max_attempts, return the last one generated (might be invalid)
    # The GA loop is expected to handle/filter invalid solutions.
    if last_child_solution:
        return last_child_solution
    else:
        # This case should ideally not be reached if len_assignment >= 2, as at least one child is always generated.
        # Fallback to returning a copy of a parent if something unexpected happens.
        return LeagueSolution(deepcopy(parent1.assignment), parent1.num_teams, parent1.team_size, parent1.max_budget)

