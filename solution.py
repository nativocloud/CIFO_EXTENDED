import random
import numpy as np
from copy import deepcopy

class LeagueSolution:
    def __init__(self, players_data_full, assignment=None, num_teams=5, team_size=7, max_budget=750):
        self.num_teams = num_teams
        self.team_size = team_size
        self.max_budget = max_budget
        self.players_data_ref = players_data_full # Store a reference

        if assignment:
            self.assignment = assignment
        else:
            # Try to generate a valid assignment using the new constructive heuristic
            # It might still fail if constraints are too tight or player pool is difficult
            self.assignment = self._random_valid_assignment_constructive(players_data_full)
            # The is_valid check will happen externally, e.g., in GA population generation loop

    def _random_valid_assignment_constructive(self, players_data_full):
        """Attempts to construct a valid assignment using a heuristic approach."""
        # Create a mutable copy of players_data with original indices for assignment
        available_players = [(i, p) for i, p in enumerate(players_data_full)]
        random.shuffle(available_players)
        
        teams_players_indices = [[] for _ in range(self.num_teams)]
        final_assignment = [-1] * len(players_data_full) # Initialize with -1 (unassigned)

        # Define target positional structure for each team
        target_roles = {"GK": 1, "DEF": 2, "MID": 2, "FWD": 2}

        for team_id in range(self.num_teams):
            current_team_indices = []
            current_team_cost = 0
            current_team_roles = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
            players_added_to_this_team = 0

            # Attempt to fill this team according to constraints
            # Iterate through positions to ensure structure
            for role, count_needed in target_roles.items():
                for _ in range(count_needed):
                    player_found_for_role = False
                    # Iterate through a shuffled list of available players to find a match
                    # We need to iterate over a copy if we remove items, or use indices carefully
                    indices_to_remove = []
                    for i, (original_idx, player_data) in enumerate(available_players):
                        if player_data["Position"] == role and \
                           (current_team_cost + player_data["Salary (€M)"]) <= self.max_budget and \
                           players_added_to_this_team < self.team_size:
                            
                            # Add player to team
                            current_team_indices.append(original_idx)
                            final_assignment[original_idx] = team_id
                            current_team_cost += player_data["Salary (€M)"]
                            current_team_roles[role] += 1
                            players_added_to_this_team += 1
                            
                            indices_to_remove.append(i)
                            player_found_for_role = True
                            break # Found a player for this specific role slot
                    
                    # Remove selected players from available_players by their current index in the list
                    # Iterate in reverse to avoid index shifting issues
                    for idx_to_remove in sorted(indices_to_remove, reverse=True):
                        available_players.pop(idx_to_remove)

                    if not player_found_for_role:
                        # Could not find a suitable player for this role slot under budget/size constraints
                        # This indicates a potential issue with the player pool or overly strict constraints
                        # For now, this might lead to an invalid team if not all roles are filled.
                        # The external is_valid() check will catch this.
                        pass 
            
            teams_players_indices[team_id] = current_team_indices

        # Check if all players were assigned (should be if team_size * num_teams == len(players_data_full))
        if len(players_data_full) == self.num_teams * self.team_size and -1 in final_assignment:
            # This means not all players were assigned, which is an issue with the constructive logic
            # or player pool. For now, return the potentially incomplete/invalid assignment.
            # The is_valid check will handle it.
            pass # Or raise an error, or try to fill remaining players randomly (less ideal)

        return final_assignment

    def is_valid(self, players_data_ref_for_check): # players_data_ref_for_check is self.players_data_ref
        if not self.assignment or len(self.assignment) != len(players_data_ref_for_check):
            return False # Basic check for assignment existence and length

        teams = [[] for _ in range(self.num_teams)]
        player_assigned_count = [0] * len(players_data_ref_for_check)

        for player_idx, team_id in enumerate(self.assignment):
            if not (0 <= team_id < self.num_teams):
                return False # Invalid team_id
            teams[team_id].append(players_data_ref_for_check[player_idx])
            player_assigned_count[player_idx] += 1
        
        # Check if each player is assigned to exactly one team
        if any(count != 1 for count in player_assigned_count):
             # This check is implicitly handled if len(assignment) == num_players and all team_ids are valid
             # and team sizes add up. The main concern is double assignment or unassignment.
             # The current assignment structure (list of team_ids for each player) ensures each player has one assignment.
             # The issue would be if not all players are in the assignment list, or list is too long.
             # The initial check for len(self.assignment) != len(players_data_ref_for_check) covers this.
            pass

        for team_idx, team_players in enumerate(teams):
            if len(team_players) != self.team_size:
                return False
            roles = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
            budget = 0
            for p_data in team_players:
                roles[p_data["Position"]] += 1
                budget += p_data["Salary (€M)"]
            
            if roles["GK"] != 1 or roles["DEF"] != 2 or roles["MID"] != 2 or roles["FWD"] != 2:
                return False
            if budget > self.max_budget:
                return False
        return True

    def fitness(self, players_data_ref_for_fitness): # players_data_ref_for_fitness is self.players_data_ref
        if not self.is_valid(players_data_ref_for_fitness):
            return float("inf")
        
        team_skills_values = [[] for _ in range(self.num_teams)]
        for player_idx, team_id in enumerate(self.assignment):
            team_skills_values[team_id].append(players_data_ref_for_fitness[player_idx]["Skill"])
        
        avg_skills = []
        for skills_in_team in team_skills_values:
            if not skills_in_team: # Should not happen if team_size > 0 and solution is valid
                avg_skills.append(0) # Or handle as error, though is_valid should prevent this
            else:
                avg_skills.append(np.mean(skills_in_team))
        
        if not avg_skills: # Should not happen if num_teams > 0
            return float("inf") # Or handle as error
            
        return np.std(avg_skills)

    def copy(self):
        # Crucially, pass the players_data_ref to the new instance
        return LeagueSolution(self.players_data_ref, assignment=deepcopy(self.assignment), \
                              num_teams=self.num_teams, team_size=self.team_size, \
                              max_budget=self.max_budget)

class LeagueHillClimbingSolution(LeagueSolution):
    def __init__(self, players_data_full, assignment=None, num_teams=5, team_size=7, max_budget=750):
        super().__init__(players_data_full, assignment, num_teams, team_size, max_budget)

    def get_neighbors(self, players_data_ref_for_neighbors): # players_data_ref_for_neighbors is self.players_data_ref
        neighbors = []
        num_players = len(self.assignment)
        for i in range(num_players):
            for j in range(i + 1, num_players):
                new_assign = deepcopy(self.assignment)
                new_assign[i], new_assign[j] = new_assign[j], new_assign[i] # Swap player assignments
                
                # Create a new solution instance with the swapped assignment
                # Pass players_data_ref to the constructor
                neighbor = LeagueHillClimbingSolution(players_data_ref_for_neighbors, assignment=new_assign, \
                                                    num_teams=self.num_teams, team_size=self.team_size, \
                                                    max_budget=self.max_budget)
                if neighbor.is_valid(players_data_ref_for_neighbors):
                    neighbors.append(neighbor)
        return neighbors
    
    def copy(self):
        return LeagueHillClimbingSolution(self.players_data_ref, assignment=deepcopy(self.assignment),\
                                          num_teams=self.num_teams, team_size=self.team_size,\
                                          max_budget=self.max_budget)

class LeagueSASolution(LeagueSolution):
    def __init__(self, players_data_full, assignment=None, num_teams=5, team_size=7, max_budget=750):
        super().__init__(players_data_full, assignment, num_teams, team_size, max_budget)

    def get_random_neighbor(self, players_data_ref_for_neighbor): # players_data_ref_for_neighbor is self.players_data_ref
        max_attempts = 100 
        attempts = 0
        num_players = len(self.assignment)

        while attempts < max_attempts:
            idx1, idx2 = random.sample(range(num_players), 2)
            
            candidate_assignment = deepcopy(self.assignment)
            candidate_assignment[idx1], candidate_assignment[idx2] = candidate_assignment[idx2], candidate_assignment[idx1]
            
            potential_neighbor = LeagueSASolution(players_data_ref_for_neighbor, assignment=candidate_assignment, \
                                                num_teams=self.num_teams, team_size=self.team_size, \
                                                max_budget=self.max_budget)
            
            if potential_neighbor.is_valid(players_data_ref_for_neighbor):
                return potential_neighbor
            attempts += 1
            
        # Fallback: if no valid neighbor is found, return a copy of the current solution.
        return self.copy() # Uses the updated copy method
    
    def copy(self):
        return LeagueSASolution(self.players_data_ref, assignment=deepcopy(self.assignment),\
                                num_teams=self.num_teams, team_size=self.team_size,\
                                max_budget=self.max_budget)

