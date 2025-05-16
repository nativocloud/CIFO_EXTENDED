import random
import numpy as np
from copy import deepcopy

class LeagueSolution:
    def __init__(self, players_data_full, assignment=None, num_teams=5, team_size=7, max_budget=750):
        self.num_teams = num_teams
        self.team_size = team_size
        self.max_budget = max_budget
        self.players_data_ref = players_data_full # Store original list of dicts

        # Pre-process player data for vectorized operations
        self.player_salaries_np = np.array([p["Salary (€M)"] for p in players_data_full])
        self.player_skills_np = np.array([p["Skill"] for p in players_data_full])
        
        self.position_map = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
        self.num_unique_roles = len(self.position_map)
        self.target_role_counts_np = np.array([1, 2, 2, 2]) # Order: GK, DEF, MID, FWD

        self.player_positions_numeric_np = np.full(len(players_data_full), -1, dtype=int) # Default to -1 (invalid)
        self._all_player_positions_mapped = True
        for i, p_data in enumerate(players_data_full):
            pos = p_data["Position"]
            if pos in self.position_map:
                self.player_positions_numeric_np[i] = self.position_map[pos]
            else:
                self._all_player_positions_mapped = False
                # print(f"Warning: Unknown position '{pos}' for player {p_data.get('Name', 'N/A')}. This solution instance will be invalid.")
                break 

        if assignment is not None:
            self.assignment = np.array(assignment, dtype=int)
        else:
            # Generate a new assignment if none provided
            # _random_valid_assignment_constructive returns a list or can be made to return np.array
            self.assignment = np.array(self._random_valid_assignment_constructive(players_data_full), dtype=int)

    def _random_valid_assignment_constructive(self, players_data_full_list_of_dicts):
        """Attempts to construct a valid assignment using a heuristic approach."""
        available_players = [(i, p) for i, p in enumerate(players_data_full_list_of_dicts)]
        random.shuffle(available_players)
        
        final_assignment = [-1] * len(players_data_full_list_of_dicts)
        target_roles_dict = {"GK": 1, "DEF": 2, "MID": 2, "FWD": 2}

        for team_id in range(self.num_teams):
            current_team_cost = 0
            players_added_to_this_team = 0

            for role_str, count_needed in target_roles_dict.items():
                for _ in range(count_needed):
                    player_found_for_role = False
                    indices_to_remove_from_available = []
                    for i_available, (original_idx, player_data) in enumerate(available_players):
                        if player_data["Position"] == role_str and \
                           (current_team_cost + player_data["Salary (€M)"]) <= self.max_budget and \
                           players_added_to_this_team < self.team_size:
                            
                            final_assignment[original_idx] = team_id
                            current_team_cost += player_data["Salary (€M)"]
                            players_added_to_this_team += 1
                            indices_to_remove_from_available.append(i_available)
                            player_found_for_role = True
                            break
                    
                    for idx_to_remove in sorted(indices_to_remove_from_available, reverse=True):
                        available_players.pop(idx_to_remove)
        return final_assignment

    def is_valid(self): # Vectorized version
        if not self._all_player_positions_mapped:
            return False # Invalid data mapping from init

        if self.assignment is None or len(self.assignment) != len(self.player_salaries_np):
            return False
        
        assignment_arr = self.assignment # Is already a NumPy array

        if not np.all((assignment_arr >= 0) & (assignment_arr < self.num_teams)):
            return False # Invalid team_id values present
        if -1 in assignment_arr: # Check if any player remained unassigned from constructive heuristic
            return False

        team_counts = np.bincount(assignment_arr, minlength=self.num_teams)
        if not np.all(team_counts == self.team_size):
            return False

        for team_id in range(self.num_teams):
            players_in_team_mask = (assignment_arr == team_id)
            
            team_budget = np.sum(self.player_salaries_np[players_in_team_mask])
            if team_budget > self.max_budget:
                return False
            
            team_positions_numeric = self.player_positions_numeric_np[players_in_team_mask]
            role_counts = np.bincount(team_positions_numeric, minlength=self.num_unique_roles)
            if not np.array_equal(role_counts, self.target_role_counts_np):
                return False
        return True

    def fitness(self): # Vectorized version
        if not self.is_valid():
            return float("inf")
        
        assignment_arr = self.assignment # Is already a NumPy array
        team_skill_sums = np.bincount(assignment_arr, weights=self.player_skills_np, minlength=self.num_teams)
        
        # is_valid() ensures all team_counts are self.team_size, so no division by zero here.
        avg_skills = team_skill_sums / self.team_size 
        return np.std(avg_skills)

    def copy(self):
        # __init__ will re-derive NumPy arrays from self.players_data_ref
        return LeagueSolution(self.players_data_ref, 
                              assignment=deepcopy(self.assignment), # deepcopy for the assignment array itself
                              num_teams=self.num_teams, 
                              team_size=self.team_size, 
                              max_budget=self.max_budget)

class LeagueHillClimbingSolution(LeagueSolution):
    def __init__(self, players_data_full, assignment=None, num_teams=5, team_size=7, max_budget=750):
        super().__init__(players_data_full, assignment, num_teams, team_size, max_budget)

    def get_neighbors(self): # Removed players_data_ref argument
        neighbors = []
        num_players = len(self.assignment)
        for i in range(num_players):
            for j in range(i + 1, num_players):
                # Create a copy of the current assignment for modification
                new_assign_arr = self.assignment.copy() # Use NumPy's copy for the array
                new_assign_arr[i], new_assign_arr[j] = new_assign_arr[j], new_assign_arr[i]
                
                neighbor = LeagueHillClimbingSolution(self.players_data_ref, assignment=new_assign_arr,
                                                    num_teams=self.num_teams, team_size=self.team_size,
                                                    max_budget=self.max_budget)
                if neighbor.is_valid(): # Call new is_valid
                    neighbors.append(neighbor)
        return neighbors
    
    def copy(self):
        return LeagueHillClimbingSolution(self.players_data_ref, assignment=deepcopy(self.assignment),
                                          num_teams=self.num_teams, team_size=self.team_size,
                                          max_budget=self.max_budget)

class LeagueSASolution(LeagueSolution):
    def __init__(self, players_data_full, assignment=None, num_teams=5, team_size=7, max_budget=750):
        super().__init__(players_data_full, assignment, num_teams, team_size, max_budget)

    def get_random_neighbor(self): # Removed players_data_ref argument
        max_attempts = 100 
        attempts = 0
        num_players = len(self.assignment)

        while attempts < max_attempts:
            if num_players < 2:
                 return self.copy() # Cannot swap if less than 2 players
            idx1, idx2 = random.sample(range(num_players), 2)
            
            # Create a copy of the current assignment for modification
            candidate_assignment_arr = self.assignment.copy() # Use NumPy's copy for the array
            candidate_assignment_arr[idx1], candidate_assignment_arr[idx2] = candidate_assignment_arr[idx2], candidate_assignment_arr[idx1]
            
            potential_neighbor = LeagueSASolution(self.players_data_ref, assignment=candidate_assignment_arr,
                                                num_teams=self.num_teams, team_size=self.team_size,
                                                max_budget=self.max_budget)
            
            if potential_neighbor.is_valid(): # Call new is_valid
                return potential_neighbor
            attempts += 1
            
        return self.copy()
    
    def copy(self):
        return LeagueSASolution(self.players_data_ref, assignment=deepcopy(self.assignment),
                                num_teams=self.num_teams, team_size=self.team_size,
                                max_budget=self.max_budget)

