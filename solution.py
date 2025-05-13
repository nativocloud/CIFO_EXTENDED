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




                














