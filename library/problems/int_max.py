"""Problem definition

Description: The IntMax problem consists of finding the biggest integer between 1 and some N

Search space: Integers from 1 to 15.

Fitness function: f(x)=x (i.e., the number itself).

Neighbors: Each integer x has at most two neighbors: x-1 and x+1, except for boundaries (1 and 15).

Goal: Maximize f(x).
"""

import random

from solution import Solution

class IntMaxSolution(Solution):
    def __init__(self, repr=None):
        # Check if valid repr was passed
        if repr:
            if (not isinstance(repr, int)) or (repr < 1) or (repr > 15):
                raise ValueError("IntMax Solution representation must be an integer between 1 and 15")

        super().__init__(repr=repr)
    
    def fitness(self):
        return self.repr
    
    def random_initial_representation(self):
        return random.randint(1, 15)

# Algorithm specific
class IntMaxHillClimbingSolution(IntMaxSolution):
    def get_neighbors(self):
        # Boundaries
        if self.repr == 1:
            return [IntMaxHillClimbingSolution(2)]
        elif self.repr == 15:
            return [IntMaxHillClimbingSolution(14)]
        else:
            return [
                IntMaxHillClimbingSolution(self.repr-1),
                IntMaxHillClimbingSolution(self.repr+1)
            ]