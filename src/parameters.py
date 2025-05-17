"""
parameters.py - Parameter management for the CIFO_EXTENDED project

This module provides classes for managing parameters in a structured way:
- ProblemParameters: Parameters related to the problem definition
- AlgorithmParameters: Parameters related to the optimization algorithms
"""

class ProblemParameters:
    """Parameters related to the Sports League Assignment Problem."""
    
    def __init__(self, num_teams=5, team_size=7, max_budget=750, 
                 positions=None, position_counts=None):
        """
        Initialize problem parameters.
        
        Args:
            num_teams (int): Number of teams in the league
            team_size (int): Number of players per team
            max_budget (int): Maximum budget per team in million euros
            positions (list): List of player positions
            position_counts (dict): Number of players required for each position
        """
        self.NUM_TEAMS = num_teams
        self.TEAM_SIZE = team_size
        self.MAX_BUDGET = max_budget
        
        # Default position configuration if not provided
        self.POSITIONS = positions or ["GK", "DEF", "MID", "FWD"]
        self.POSITION_COUNTS = position_counts or {"GK": 1, "DEF": 2, "MID": 2, "FWD": 2}
        
        # Validate that position counts sum to team size
        total_positions = sum(self.POSITION_COUNTS.values())
        if total_positions != self.TEAM_SIZE:
            raise ValueError(f"Position counts ({total_positions}) must sum to team size ({self.TEAM_SIZE})")
    
    def __str__(self):
        """Return a string representation of the parameters."""
        return (f"Problem Parameters:\n"
                f"  Teams: {self.NUM_TEAMS}\n"
                f"  Players per team: {self.TEAM_SIZE}\n"
                f"  Max budget: {self.MAX_BUDGET}M €\n"
                f"  Positions: {self.POSITIONS}\n"
                f"  Position counts: {self.POSITION_COUNTS}")


class AlgorithmParameters:
    """Parameters related to optimization algorithms."""
    
    def __init__(self, num_runs=30, population_size=100, max_generations=100,
                 mutation_rate=0.1, crossover_prob=0.8, elitism_size=5,
                 tournament_size=3, boltzmann_temp=1.0, cooling_rate=0.99,
                 initial_temp=100.0, min_temp=1.0, max_iterations=1000):
        """
        Initialize algorithm parameters.
        
        Args:
            num_runs (int): Number of independent runs for statistical analysis
            population_size (int): Size of the population for GA
            max_generations (int): Maximum number of generations for GA
            mutation_rate (float): Probability of mutation
            crossover_prob (float): Probability of crossover
            elitism_size (int): Number of elite individuals to preserve
            tournament_size (int): Size of tournament for selection
            boltzmann_temp (float): Temperature for Boltzmann selection
            cooling_rate (float): Cooling rate for simulated annealing
            initial_temp (float): Initial temperature for simulated annealing
            min_temp (float): Minimum temperature for simulated annealing
            max_iterations (int): Maximum iterations for local search algorithms
        """
        # General parameters
        self.NUM_RUNS = num_runs
        
        # GA parameters
        self.POPULATION_SIZE = population_size
        self.MAX_GENERATIONS = max_generations
        self.MUTATION_RATE = mutation_rate
        self.CROSSOVER_PROB = crossover_prob
        self.ELITISM_SIZE = elitism_size
        self.TOURNAMENT_SIZE = tournament_size
        self.BOLTZMANN_TEMP = boltzmann_temp
        
        # SA parameters
        self.INITIAL_TEMP = initial_temp
        self.MIN_TEMP = min_temp
        self.COOLING_RATE = cooling_rate
        
        # HC and general local search parameters
        self.MAX_ITERATIONS = max_iterations
    
    def __str__(self):
        """Return a string representation of the parameters."""
        return (f"Algorithm Parameters:\n"
                f"  General:\n"
                f"    Runs: {self.NUM_RUNS}\n"
                f"  Genetic Algorithm:\n"
                f"    Population size: {self.POPULATION_SIZE}\n"
                f"    Max generations: {self.MAX_GENERATIONS}\n"
                f"    Mutation rate: {self.MUTATION_RATE}\n"
                f"    Crossover probability: {self.CROSSOVER_PROB}\n"
                f"    Elitism size: {self.ELITISM_SIZE}\n"
                f"    Tournament size: {self.TOURNAMENT_SIZE}\n"
                f"  Simulated Annealing:\n"
                f"    Initial temperature: {self.INITIAL_TEMP}\n"
                f"    Minimum temperature: {self.MIN_TEMP}\n"
                f"    Cooling rate: {self.COOLING_RATE}\n"
                f"  Hill Climbing:\n"
                f"    Max iterations: {self.MAX_ITERATIONS}")


# Default parameter instances
default_problem_params = ProblemParameters()
default_algorithm_params = AlgorithmParameters()


def get_problem_parameters():
    """Get default problem parameters."""
    return default_problem_params


def get_algorithm_parameters():
    """Get default algorithm parameters."""
    return default_algorithm_params
