"""
parameters.py - Parameter management for the CIFO_EXTENDED project

This module provides classes for managing parameters in a structured way:
- ProblemParameters: Parameters related to the problem definition
- AlgorithmParameters: Parameters related to the optimization algorithms

Example usage:
    from parameters import ProblemParameters, AlgorithmParameters

    # Create problem parameters with default values
    problem_params = ProblemParameters()
    print(problem_params)

    # Create custom problem parameters
    custom_problem = ProblemParameters(
        num_teams=4,
        team_size=8,
        max_budget=800,
        positions=["GK", "DEF", "MID", "FWD"],
        position_counts={"GK": 1, "DEF": 3, "MID": 2, "FWD": 2}
    )

    # Create algorithm parameters
    algo_params = AlgorithmParameters(
        num_runs=50,
        population_size=200,
        mutation_rate=0.2
    )

    # Convert to dictionary for serialization
    params_dict = custom_problem.to_dict()

    # Create from dictionary
    restored_params = ProblemParameters.from_dict(params_dict)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass(frozen=True)
class ProblemParameters:
    """Parameters related to the Sports League Assignment Problem.

    Attributes:
        NUM_TEAMS: Number of teams in the league
        TEAM_SIZE: Number of players per team
        MAX_BUDGET: Maximum budget per team in million euros
        POSITIONS: List of player positions
        POSITION_COUNTS: Number of players required for each position

    Raises:
        ValueError: If position counts don't sum to team size
        ValueError: If number of teams is not positive
        ValueError: If team size is not positive
        ValueError: If max budget is negative
        ValueError: If invalid positions are specified in position_counts
    """

    # Use __slots__ for memory efficiency
    __slots__ = ["NUM_TEAMS", "TEAM_SIZE", "MAX_BUDGET", "POSITIONS", "POSITION_COUNTS"]
    NUM_TEAMS: int = 5
    TEAM_SIZE: int = 7
    MAX_BUDGET: int = 750
    POSITIONS: List[str] = field(default_factory=lambda: ["GK", "DEF", "MID", "FWD"])
    POSITION_COUNTS: Dict[str, int] = field(
        default_factory=lambda: {"GK": 1, "DEF": 2, "MID": 2, "FWD": 2}
    )

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        # Using object.__setattr__ because the dataclass is frozen
        if self.NUM_TEAMS <= 0:
            raise ValueError("Number of teams must be positive")
        if self.TEAM_SIZE <= 0:
            raise ValueError("Team size must be positive")
        if self.MAX_BUDGET < 0:
            raise ValueError("Budget cannot be negative")

        self._validate_position_counts()

    def _validate_position_counts(self) -> None:
        """Validate that position counts are valid."""
        total_positions = sum(self.POSITION_COUNTS.values())
        if total_positions != self.TEAM_SIZE:
            raise ValueError(
                f"Position counts sum to {total_positions}, but must equal team size {self.TEAM_SIZE}"
            )

        # Check all positions in position_counts are valid
        invalid_positions = set(self.POSITION_COUNTS) - set(self.POSITIONS)
        if invalid_positions:
            raise ValueError(
                f"Invalid positions in position_counts: {invalid_positions}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary for serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of parameters
        """
        return {
            "NUM_TEAMS": self.NUM_TEAMS,
            "TEAM_SIZE": self.TEAM_SIZE,
            "MAX_BUDGET": self.MAX_BUDGET,
            "POSITIONS": self.POSITIONS.copy(),
            "POSITION_COUNTS": self.POSITION_COUNTS.copy(),
        }

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ProblemParameters":
        """Create ProblemParameters from dictionary.

        Args:
            params: Dictionary with parameter values

        Returns:
            ProblemParameters: New instance with values from dictionary

        Raises:
            KeyError: If required parameters are missing from dictionary
        """
        return cls(
            num_teams=params.get("NUM_TEAMS", cls.NUM_TEAMS),
            team_size=params.get("TEAM_SIZE", cls.TEAM_SIZE),
            max_budget=params.get("MAX_BUDGET", cls.MAX_BUDGET),
            positions=params.get("POSITIONS", None),
            position_counts=params.get("POSITION_COUNTS", None),
        )

    def __str__(self) -> str:
        """Return a string representation of the parameters."""
        return (
            f"Problem Parameters:\n"
            f"  Teams: {self.NUM_TEAMS}\n"
            f"  Players per team: {self.TEAM_SIZE}\n"
            f"  Max budget: {self.MAX_BUDGET}M €\n"
            f"  Positions: {self.POSITIONS}\n"
            f"  Position counts: {self.POSITION_COUNTS}"
        )

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (
            f"ProblemParameters(NUM_TEAMS={self.NUM_TEAMS}, "
            f"TEAM_SIZE={self.TEAM_SIZE}, "
            f"MAX_BUDGET={self.MAX_BUDGET}, "
            f"POSITIONS={self.POSITIONS}, "
            f"POSITION_COUNTS={self.POSITION_COUNTS})"
        )


@dataclass(frozen=True)
class AlgorithmParameters:
    """Parameters related to optimization algorithms.

    Attributes:
        NUM_RUNS: Number of independent runs for statistical analysis
        POPULATION_SIZE: Size of the population for GA
        MAX_GENERATIONS: Maximum number of generations for GA
        MUTATION_RATE: Probability of mutation
        CROSSOVER_PROB: Probability of crossover
        ELITISM_SIZE: Number of elite individuals to preserve
        TOURNAMENT_SIZE: Size of tournament for selection
        BOLTZMANN_TEMP: Temperature for Boltzmann selection
        INITIAL_TEMP: Initial temperature for simulated annealing
        MIN_TEMP: Minimum temperature for simulated annealing
        COOLING_RATE: Cooling rate for simulated annealing
        MAX_ITERATIONS: Maximum iterations for local search algorithms

    Raises:
        ValueError: If parameters are outside valid ranges
    """

    # Use __slots__ for memory efficiency
    __slots__ = [
        "NUM_RUNS",
        "POPULATION_SIZE",
        "MAX_GENERATIONS",
        "MUTATION_RATE",
        "CROSSOVER_PROB",
        "ELITISM_SIZE",
        "TOURNAMENT_SIZE",
        "BOLTZMANN_TEMP",
        "INITIAL_TEMP",
        "MIN_TEMP",
        "COOLING_RATE",
        "MAX_ITERATIONS",
    ]
    # General parameters
    NUM_RUNS: int = 30

    # GA parameters
    POPULATION_SIZE: int = 100
    MAX_GENERATIONS: int = 100
    MUTATION_RATE: float = 0.1
    CROSSOVER_PROB: float = 0.8
    ELITISM_SIZE: int = 5
    TOURNAMENT_SIZE: int = 3
    BOLTZMANN_TEMP: float = 1.0

    # SA parameters
    INITIAL_TEMP: float = 100.0
    MIN_TEMP: float = 1.0
    COOLING_RATE: float = 0.99

    # HC and general local search parameters
    MAX_ITERATIONS: int = 1000

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.NUM_RUNS <= 0:
            raise ValueError("Number of runs must be positive")
        if self.POPULATION_SIZE <= 0:
            raise ValueError("Population size must be positive")
        if self.MAX_GENERATIONS <= 0:
            raise ValueError("Maximum generations must be positive")
        if not 0 <= self.MUTATION_RATE <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        if not 0 <= self.CROSSOVER_PROB <= 1:
            raise ValueError("Crossover probability must be between 0 and 1")
        if self.ELITISM_SIZE < 0:
            raise ValueError("Elitism size cannot be negative")
        if self.TOURNAMENT_SIZE <= 0:
            raise ValueError("Tournament size must be positive")
        if self.BOLTZMANN_TEMP <= 0:
            raise ValueError("Boltzmann temperature must be positive")
        if self.INITIAL_TEMP <= 0:
            raise ValueError("Initial temperature must be positive")
        if self.MIN_TEMP <= 0:
            raise ValueError("Minimum temperature must be positive")
        if not 0 < self.COOLING_RATE < 1:
            raise ValueError("Cooling rate must be between 0 and 1 (exclusive)")
        if self.MAX_ITERATIONS <= 0:
            raise ValueError("Maximum iterations must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary for serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of parameters
        """
        return {
            "NUM_RUNS": self.NUM_RUNS,
            "POPULATION_SIZE": self.POPULATION_SIZE,
            "MAX_GENERATIONS": self.MAX_GENERATIONS,
            "MUTATION_RATE": self.MUTATION_RATE,
            "CROSSOVER_PROB": self.CROSSOVER_PROB,
            "ELITISM_SIZE": self.ELITISM_SIZE,
            "TOURNAMENT_SIZE": self.TOURNAMENT_SIZE,
            "BOLTZMANN_TEMP": self.BOLTZMANN_TEMP,
            "INITIAL_TEMP": self.INITIAL_TEMP,
            "MIN_TEMP": self.MIN_TEMP,
            "COOLING_RATE": self.COOLING_RATE,
            "MAX_ITERATIONS": self.MAX_ITERATIONS,
        }

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "AlgorithmParameters":
        """Create AlgorithmParameters from dictionary.

        Args:
            params: Dictionary with parameter values

        Returns:
            AlgorithmParameters: New instance with values from dictionary
        """
        return cls(
            num_runs=params.get("NUM_RUNS", cls.NUM_RUNS),
            population_size=params.get("POPULATION_SIZE", cls.POPULATION_SIZE),
            max_generations=params.get("MAX_GENERATIONS", cls.MAX_GENERATIONS),
            mutation_rate=params.get("MUTATION_RATE", cls.MUTATION_RATE),
            crossover_prob=params.get("CROSSOVER_PROB", cls.CROSSOVER_PROB),
            elitism_size=params.get("ELITISM_SIZE", cls.ELITISM_SIZE),
            tournament_size=params.get("TOURNAMENT_SIZE", cls.TOURNAMENT_SIZE),
            boltzmann_temp=params.get("BOLTZMANN_TEMP", cls.BOLTZMANN_TEMP),
            initial_temp=params.get("INITIAL_TEMP", cls.INITIAL_TEMP),
            min_temp=params.get("MIN_TEMP", cls.MIN_TEMP),
            cooling_rate=params.get("COOLING_RATE", cls.COOLING_RATE),
            max_iterations=params.get("MAX_ITERATIONS", cls.MAX_ITERATIONS),
        )

    def __str__(self) -> str:
        """Return a string representation of the parameters."""
        return (
            f"Algorithm Parameters:\n"
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
            f"    Max iterations: {self.MAX_ITERATIONS}"
        )

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (
            f"AlgorithmParameters(NUM_RUNS={self.NUM_RUNS}, "
            f"POPULATION_SIZE={self.POPULATION_SIZE}, "
            f"MAX_GENERATIONS={self.MAX_GENERATIONS}, "
            f"MUTATION_RATE={self.MUTATION_RATE}, "
            f"CROSSOVER_PROB={self.CROSSOVER_PROB}, "
            f"ELITISM_SIZE={self.ELITISM_SIZE}, "
            f"TOURNAMENT_SIZE={self.TOURNAMENT_SIZE}, "
            f"BOLTZMANN_TEMP={self.BOLTZMANN_TEMP}, "
            f"INITIAL_TEMP={self.INITIAL_TEMP}, "
            f"MIN_TEMP={self.MIN_TEMP}, "
            f"COOLING_RATE={self.COOLING_RATE}, "
            f"MAX_ITERATIONS={self.MAX_ITERATIONS})"
        )


# Default parameter instances
default_problem_params = ProblemParameters()
default_algorithm_params = AlgorithmParameters()


def get_problem_parameters() -> ProblemParameters:
    """Get default problem parameters.

    Returns:
        ProblemParameters: Default problem parameters instance
    """
    return default_problem_params


def get_algorithm_parameters() -> AlgorithmParameters:
    """Get default algorithm parameters.

    Returns:
        AlgorithmParameters: Default algorithm parameters instance
    """
    return default_algorithm_params
