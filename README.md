# CIFO Extended Project - Single Processor Analysis

This repository contains the code and resources for the Sports League Assignment Problem optimization using metaheuristics. This branch focuses specifically on the single processor implementation.

## Project Structure

```
CIFO_EXTENDED_Project/
├── CIFO_Problem_Statement.txt   # Problem definition
├── data/
│   └── players.csv              # Player data for the league assignment
├── environment.yml              # Conda environment configuration
├── images_sp/                   # Output directory for single processor results
│   ├── all_algorithms_summary_sp.csv       # Summary of all algorithm results
│   ├── hc_convergence_sp.png               # Hill Climbing convergence plot
│   ├── sa_convergence_sp.png               # Simulated Annealing convergence plot
│   └── ga_convergence_sp_*.png             # GA convergence plots for different configurations
├── notebooks/
│   ├── 01_single_processor_analysis.ipynb  # Jupyter notebook for analysis
│   ├── 01_single_processor_analysis.py     # Python script paired with notebook
│   └── set_jupytext_pairing_01.py          # Script to set up Jupytext pairing
├── setup.sh                     # Setup script for environment
└── src/                         # Source code modules
    ├── __init__.py
    ├── evolution/               # Evolutionary algorithms implementation
    │   ├── __init__.py
    │   └── evolution.py         # Hill Climbing, Simulated Annealing, and GA implementations
    ├── operators/               # Genetic operators
    │   ├── __init__.py
    │   └── operators.py         # Mutation, crossover, and selection operators
    └── solution/                # Problem representation
        ├── __init__.py
        └── solution.py          # League solution classes and fitness evaluation
```

## Problem Overview

The Sports League Assignment Problem involves allocating 35 players into 5 teams while satisfying constraints:
- Each team must have exactly 7 players
- Each team must follow a specific positional structure (1 GK, 2 DEF, 2 MID, 2 FWD)
- Each team's total cost must not exceed 750 million €
- Each player must be assigned to exactly one team

The objective is to minimize the standard deviation of average skill ratings across teams.

## Algorithms Implemented

1. **Hill Climbing**: A local search algorithm that iteratively moves to neighboring solutions with better fitness.
2. **Simulated Annealing**: A probabilistic technique that occasionally accepts worse solutions to escape local optima.
3. **Genetic Algorithm**: A population-based approach with multiple configurations testing different:
   - Mutation operators (swap, targeted exchange, shuffle)
   - Crossover operators (one-point, uniform)
   - Selection mechanisms (tournament, ranking, Boltzmann)
   - Population sizes and generation counts

## Setup and Execution

### Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate cifo-extended

# Alternative setup using the provided script
bash setup.sh
```

### Running the Analysis

The analysis can be run either through the Jupyter notebook or directly using the Python script:

```bash
# Run via Python script
python -m notebooks.01_single_processor_analysis

# Or open and run the notebook
jupyter notebook notebooks/01_single_processor_analysis.ipynb
```

## Implementation Details

- **Vectorized Operations**: The solution uses NumPy for vectorized operations to improve performance
- **Cross-Platform Compatibility**: All file paths use relative paths with `os.path.join()` for Windows/Linux compatibility
- **Jupytext Pairing**: The notebook is paired with a Python script for version control and reproducibility

## Results

The analysis generates several outputs in the `images_sp/` directory:
- Convergence plots for each algorithm
- CSV summary of performance metrics (fitness values, execution times)
- Comparative analysis across all implemented algorithms

The single processor analysis serves as a baseline for comparison with multi-processor implementations.
