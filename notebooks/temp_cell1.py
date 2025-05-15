import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os # Added for path joining
import sys

# Add the current directory to the path to ensure modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solution import LeagueSolution, LeagueHillClimbingSolution, LeagueSASolution
from evolution import genetic_algorithm, hill_climbing, simulated_annealing 
from operators import (
    # Base Mutations
    mutate_swap,
    mutate_team_shift,
    mutate_shuffle_team,
    # New/Adapted Mutations
    mutate_swap_constrained,
    mutate_targeted_player_exchange,
    mutate_shuffle_within_team_constrained,
    # Base Crossovers
    crossover_one_point,
    crossover_uniform,
    # New/Adapted Crossovers
    crossover_one_point_prefer_valid,
    crossover_uniform_prefer_valid,
    # Selection Operators
    selection_ranking,
    selection_tournament_variable_k,
    selection_boltzmann
)

# Define the directory for saving graphs and results
IMAGES_SP_DIR = "/home/ubuntu/CIFO_EXTENDED_Project/images_sp" # Updated directory
# Ensure the directory exists
if not os.path.exists(IMAGES_SP_DIR):
    os.makedirs(IMAGES_SP_DIR)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Created directory: {IMAGES_SP_DIR}")

# Load player data
players_df = pd.read_csv("players.csv", sep=";")
players_data = players_df.to_dict(orient="records") # Renamed to players_data for clarity
