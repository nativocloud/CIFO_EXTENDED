import random
import numpy as np
from copy import deepcopy
#NUM_TEAMS = 5
#TEAM_SIZE = 7
#MAX_BUDGET = 750

class LeagueSolution:
    def __init__(self, assignment=None, num_teams=5, team_size=7, max_budget=750):
        self.num_teams = num_teams
        self.team_size = team_size
        self.max_budget = max_budget
        self.assignment = assignment or self._random_valid_assignment()

    def _random_valid_assignment(self):
        #team_ids = [i % NUM_TEAMS for i in range(35)]
        team_ids = [i % self.num_teams for i in range(self.num_teams * self.team_size)]
        random.shuffle(team_ids)
        return team_ids

    def is_valid(self, players):
        #teams = [[] for _ in range(NUM_TEAMS)]
        teams = [[] for _ in range(self.num_teams)]
        for idx, team_id in enumerate(self.assignment):
            teams[team_id].append(players[idx])
        for team in teams:
            #if len(team) != TEAM_SIZE:
            if len(team) != self.team_size:
                return False
            roles = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
            budget = 0
            for p in team:
                roles[p["Position"]] += 1
                budget += p["Salary"]
            if roles != {"GK": 1, "DEF": 2, "MID": 2, "FWD": 2}:
                return False
            #if budget > MAX_BUDGET:
            if budget > self.max_budget:
                return False
        return True

    def fitness(self, players):
        if not self.is_valid(players):
            return float("inf")
        #team_skills = [[] for _ in range(NUM_TEAMS)]
        team_skills = [[] for _ in range(self.num_teams)]
        for idx, team_id in enumerate(self.assignment):
            team_skills[team_id].append(players[idx]["Skill"])
        avg_skills = [np.mean(team) for team in team_skills]
        return np.std(avg_skills)

    def copy(self):
        return LeagueSolution(self.assignment[:])


class LeagueHillClimbingSolution(LeagueSolution):
    def get_neighbors(self, players):
        neighbors = []
        for i in range(len(self.assignment)):
            for j in range(i + 1, len(self.assignment)):
                new_assign = self.assignment[:]
                new_assign[i], new_assign[j] = new_assign[j], new_assign[i]
                neighbor = LeagueHillClimbingSolution(new_assign)
                if neighbor.is_valid(players):
                    neighbors.append(neighbor)
        return neighbors




                

















class LeagueSASolution(LeagueSolution):
    def get_random_neighbor(self, players):
        # Create a deep copy to avoid modifying the original solution
        new_solution = LeagueSASolution(deepcopy(self.assignment), self.num_teams, self.team_size, self.max_budget)
        
        # Simple swap mutation: pick two random players and swap their team assignments
        # Ensure the new assignment is valid before returning
        attempts = 0
        max_attempts = 100 # To avoid infinite loops if it's hard to find a valid neighbor
        while attempts < max_attempts:
            idx1, idx2 = random.sample(range(len(new_solution.assignment)), 2)
            
            # Perform swap
            original_team_p1 = new_solution.assignment[idx1]
            original_team_p2 = new_solution.assignment[idx2]
            
            # Temporarily assign to check validity (or simply swap and then validate)
            # For SA, we often generate a neighbor and then check its validity and fitness
            # Let's try a direct swap and then validate the whole solution
            # This is simpler than trying to maintain validity during the swap itself for complex constraints
            
            # Create a candidate assignment by swapping
            candidate_assignment = deepcopy(self.assignment)
            candidate_assignment[idx1], candidate_assignment[idx2] = candidate_assignment[idx2], candidate_assignment[idx1]
            
            # Create a new solution instance with the candidate assignment
            neighbor_solution = LeagueSASolution(candidate_assignment, self.num_teams, self.team_size, self.max_budget)
            
            if neighbor_solution.is_valid(players):
                return neighbor_solution
            attempts += 1
        
        # If max_attempts reached and no valid neighbor found, return a copy of the current solution
        # Or, alternatively, one could try a different neighbor generation strategy
        # For now, returning a copy means SA might get stuck if valid neighbors are hard to find
        # A better approach for complex problems might be to ensure neighbor generation always leads to valid, or repair invalid ones.
        # However, the problem statement says: "Invalid configurations ... are not part of the search space and must not be generated during evolution."
        # This implies mutation/neighbor generation should ideally produce valid solutions.
        # The current get_neighbors in HC does this by checking validity before adding.
        # Let's adapt that: generate a random swap, check if valid. Repeat until a valid one is found or max_attempts.

        # Re-attempting with a loop that ensures validity for the *returned* neighbor
        attempts = 0
        while attempts < max_attempts:
            idx1, idx2 = random.sample(range(len(self.assignment)), 2)
            
            # Create a candidate assignment by swapping
            candidate_assignment = deepcopy(self.assignment)
            candidate_assignment[idx1], candidate_assignment[idx2] = candidate_assignment[idx2], candidate_assignment[idx1]
            
            # Create a new solution instance with the candidate assignment
            potential_neighbor = LeagueSASolution(candidate_assignment, self.num_teams, self.team_size, self.max_budget)
            
            if potential_neighbor.is_valid(players):
                return potential_neighbor
            attempts += 1
            
        # Fallback: if no valid neighbor is found after many attempts, return a copy of the current solution.
        # This isn't ideal as SA might not explore effectively.
        # A more robust neighbor generation that guarantees validity or a repair mechanism would be better.
        # For now, this matches the simplicity of the HC neighbor generation.
        return LeagueSASolution(deepcopy(self.assignment), self.num_teams, self.team_size, self.max_budget) # Return a copy of current if no valid found

