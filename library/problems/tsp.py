"""Problem definition

Description: The Traveling Salesperson Problem (TSP) is the challenge of finding the shortest possible route that starts in a given city, visits each of the remaining N-1 cities exactly once, and returns to the starting city.

Search space: All possible permutations of city visit orders, forming valid round-trip routes.

Representation: List of city indexes that compose the route

Fitness function: f(x)= Total distance traveled, computed as the sum of distances between consecutive cities in the route.

Neighbors: A neighbor solution is obtained by swapping the positions of two consecutive cities in the route.

Goal: Minimize f(x).
"""
import random
from copy import deepcopy

from solution import Solution

from problems.data.tsp_data import distance_matrix

class TSPSolution(Solution):
    """The Travel Salesperson Problem (TSP) aims at finding the shortest possible route that starts and
    ends in a given city and visits all other cities once."""
    def __init__(
        self,
        repr = None,
        distance_matrix: list[list[float]] = distance_matrix,
        starting_idx: int = 0
    ):
        self.starting_idx = starting_idx
        self.distance_matrix = distance_matrix

        # Validate repr if it is passed as argument
        if repr:
            self._validate_repr(repr)

        super().__init__(repr=repr)
    
    def _validate_repr(self, repr):
        # Confirm repr is list
        if not isinstance(repr, list):
            raise TypeError('Representation must be a list')
        # Make sure repr is a list of integers
        elif not all([isinstance(idx, int) for idx in repr]):
            raise TypeError('Representation must be a list of integers (indexes of the route)')
        # Validate start and end route points
        if (repr[0] != self.starting_idx) or (repr[-1] != self.starting_idx):
            raise ValueError("TSP route must start and end in the starting index")
        # Validate route length and content
        if (len(repr) != (len(distance_matrix) + 1)) or (set(repr) != set([i for i in range(len(distance_matrix))])):
            raise ValueError("TSP route must pass through all cities")

    def random_initial_representation(self):
        # Route starts in starting idx
        route = [self.starting_idx]
        # Get city idx to visit and shuffle them
        idx_to_visit = [idx for idx in range(len(self.distance_matrix)) if idx != self.starting_idx]
        random.shuffle(idx_to_visit)
        # Add idx to visit to route
        route = route + idx_to_visit
        # Route ends in starting idx
        route = route + [self.starting_idx]
        return route
    
    def fitness(self):
        total_distance = 0
        for i in range(len(self.repr)-1):
            total_distance += self.distance_matrix[self.repr[i]][self.repr[i+1]]
        return total_distance


class TSPHillClimbingSolution(TSPSolution):
    def get_neighbors(self):
        """Neighbors are obtained by swaping the positions of two consecutive cities"""
        neighbors = []
        for i in range(1, len(self.repr)-2):
            new_route = deepcopy(self.repr)
            new_route[i], new_route[i+1] = new_route[i+1], new_route[i]
            neighbor = TSPHillClimbingSolution(route=new_route, distance_matrix=self.distance_matrix)
            neighbors.append(neighbor)

        return neighbors

class TSPSASolution(TSPSolution):
    def get_random_neighbor(self):
        """Random neighbor is obtained by swaping the positions of two random consecutive cities"""
        nr_cities = len(self.distance_matrix)

        # Choose a city idx to switch with the next city 
        random_city_idx = random.randint(1, nr_cities-3)

        new_route = deepcopy(self.repr)
        new_route[random_city_idx] = self.repr[random_city_idx+1]
        new_route[random_city_idx+1] = self.repr[random_city_idx]

        return TSPSASolution(repr=new_route, distance_matrix=self.distance_matrix, starting_idx=self.starting_idx)
    
class TSPGASolution(TSPSolution):
    def __init__(
        self,
        distance_matrix,
        starting_idx,
        mutation_function, # Callable
        crossover_function, # Callable
        repr = None,
    ):
        # Save as attributes for access in methods
        self.mutation_function = mutation_function
        self.crossover_function = crossover_function

        super().__init__(
            distance_matrix=distance_matrix,
            starting_idx=starting_idx,
            repr=repr,
        )
    
    def mutation(self, mut_prob):
        """
        Applies the provided mutation operator to the middle portion 
        of the route (excluding start and end cities).
        """
        # Apply mutation to the middle route segment
        middle_segment = self.repr[1:-1]  # Exclude starting/ending city
        mutated_segment = self.mutation_operator(middle_segment, mut_prob)
        new_repr = [self.starting_idx] + mutated_segment + [self.starting_idx]
        
        return TSPGASolution(
            distance_matrix=self.distance_matrix,
            starting_idx=self.starting_idx,
            mutation_function=self.mutation_function,
            crossover_function=self.crossover_function,
            repr=new_repr
        )

    def crossover(self, other_solution):
        """
        Applies the provided crossover operator to the middle portions
        of two parent routes (excluding start/end cities), and returns
        two new offspring solutions.
        """
        # Apply crossover to the middle route segment of the parents
        parent1_middle = self.repr[1:-1]
        parent2_middle = other_solution.repr[1:-1]

        offspring1_middle, offspring2_middle = self.crossover_function(parent1_middle, parent2_middle)

        offspring1_repr = [self.starting_idx] + offspring1_middle + [self.starting_idx]
        offspring2_repr = [self.starting_idx] + offspring2_middle + [self.starting_idx]

        return (
            TSPGASolution(
                distance_matrix=self.distance_matrix,
                starting_idx=self.starting_idx,
                mutation_function=self.mutation_function,
                crossover_function=self.crossover_function,
                repr=offspring1_repr
            ),
            TSPGASolution(
                distance_matrix=self.distance_matrix,
                starting_idx=self.starting_idx,
                mutation_function=self.mutation_function,
                crossover_function=self.crossover_function,
                repr=offspring2_repr
            )
        )