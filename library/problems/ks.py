"""Problem definition

Description: The Knapsack Problem involves selecting a subset of N items, each with a given value and weight, to pack into a container with a fixed capacity. If the total weight of selected items exceeds the capacity, the solution is invalid. The goal is to maximize the total value of items while ensuring they fit within the container's constraints.

Search space: All possible subsets of items that can be placed in the knapsack.

Representation: Binary string of length N (number of items), where 1 indicates the item is included in the knapsack and 0 indicates the item is excluded.

Fitness function: f(x)= Total value inside the knapsack. If the total size exceeds the knapsack's capacity, the solution is invalid and assigned a fitness of -inf.

Neighbors: A neighbor solution is obtained by flipping a single bit, meaning adding one item to the knapsack, or removing one item from the knapsack.

Goal: Maximize f(x).
"""

import random
from copy import deepcopy

from solution import Solution

from problems.data.ks_data import values, weights, capacity

class KSSolution(Solution):
    """The Knapsack problem aims at finding the best way to pack items in a container
    maximizing the container value while not exceeding the capacity"""
    def __init__(
        self,
        values: list[float] = values,
        weights: list[float] = weights,
        capacity: float = capacity,
        repr: str = None,
    ):
        self.values = values
        self.weights = weights
        self.capacity = capacity

        if repr:
            repr = self._validate_repr(repr)

        super().__init__(repr=repr)
    
    def _validate_repr(self, repr):
        # If repr is given as string, convert to list
        if isinstance(repr, str):
            repr = [int(bit) for bit in repr]
        if not isinstance(repr, list):
            raise TypeError("Representation must be string or list")
        # All list elements should be integers
        if not all([isinstance(bit, int) for bit in repr]):
            repr = [int(bit) for bit in repr]
        # Validate representation length and content
        if (len(repr) != len(self.values)) or (not set(repr).issubset({0, 1})):
            raise ValueError("Representation must be a binary string/list with as many values as objects")
        return repr

    def random_initial_representation(self):
        repr = []
        for _ in range(len(self.values)):
            repr.append(random.choice([0, 1]))
        return repr
    
    def total_weight(self):
        total = 0
        for idx, bin_value in enumerate(self.repr):
            if bin_value == 1:
                total += self.weights[idx]
        return total

    def total_value(self):
        total = 0
        for idx, bin_value in enumerate(self.repr):
            if bin_value == 1:
                total += self.values[idx]
        return total
    
    def fitness(self):
        total_weight = self.total_weight()
        
        if total_weight > self.capacity:
            return -9999999999999999
        
        return self.total_value()

class KSHillClimbingSolution(KSSolution):
    def get_neighbors(self):
        """Neighbors are obtained by flipping a bit in the representation. This means
        adding or removing one item from the container. One neighbor is generated for
        each bit flip."""
        neighbors = []

        for idx, bin_value in enumerate(self.repr):
            neighbor_repr = deepcopy(self.repr)
            if bin_value == 1:
                neighbor_repr[idx] = 0
            else:
                neighbor_repr[idx] = 1

            neighbor = KSHillClimbingSolution(
                repr=neighbor_repr,
                values=self.values,
                weights=self.weights,
                capacity=self.capacity,
            )
            neighbors.append(neighbor)

        return neighbors

class KSSASolution(KSSolution):
    def get_random_neighbor(self):
        """A random neighbor is obtained by flipping a random bit in the representation.
        This means adding or removing one item from the container"""
        neighbor_repr = deepcopy(self.repr)
        # Get random index
        random_idx = random.randint(0, len(self.values)-1)
        # Bit flip
        if neighbor_repr[random_idx] == 1:
            neighbor_repr[random_idx] = 0
        else:
            neighbor_repr[random_idx] = 1
        
        return KSSASolution(
            repr=neighbor_repr,
            values=self.values,
            weights=self.weights,
            capacity=self.capacity,
        )

class KSGASolution(KSSolution):
    def __init__(
        self,
        values,
        weights,
        capacity,
        mutation_function, # Callable
        crossover_function, # Callable
        repr = None
    ):
        super().__init__(
            values=values,
            weights=weights,
            capacity=capacity,
            repr=repr,
        )

        # Save as attributes for access in methods
        self.mutation_function = mutation_function
        self.crossover_function = crossover_function

    
    def mutation(self, mut_prob):
        # Apply mutation function to representation
        new_repr = self.mutation_function(self.repr, mut_prob)
        # Create and return individual with mutated representation
        return KSGASolution(
            values=self.values,
            weights=self.weights,
            capacity=self.capacity,
            mutation_function=self.mutation_function,
            crossover_function=self.crossover_function,
            repr=new_repr
        )

    def crossover(self, other_solution):
        # Apply crossover function to self representation and other solution representation
        offspring1_repr, offspring2_repr = self.crossover_function(self.repr, other_solution.repr)

        # Create and return offspring with new representations
        return (
            KSGASolution(
                values=self.values,
                weights=self.weights,
                capacity=self.capacity,
                mutation_function=self.mutation_function,
                crossover_function=self.crossover_function,
                repr=offspring1_repr
            ),
            KSGASolution(
                values=self.values,
                weights=self.weights,
                capacity=self.capacity,
                mutation_function=self.mutation_function,
                crossover_function=self.crossover_function,
                repr=offspring2_repr
            )
        )