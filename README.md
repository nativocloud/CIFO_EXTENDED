# CIFO Extended Project - Single Processor Analysis

## Project Overview

This project implements and analyzes metaheuristic algorithms for the Sports League Assignment Problem. The goal is to allocate players into teams while minimizing the standard deviation of team skill levels, subject to constraints on team composition and budget.

## Modular Structure

The project has been organized with a modular structure:

```
CIFO_EXTENDED_Project/
├── data/                          # Input data
│   └── players.csv                # Player information
├── results/                       # Results organized by phase
│   └── phase_1_sp/                # Single processor results
│       ├── images/                # Generated plots
│       └── data/                  # Generated data files
├── notebooks/                     # Jupyter notebooks
│   └── phase_1_sp/                # Single processor notebooks
│       ├── 01_single_processor_analysis.ipynb
│       └── 01_single_processor_analysis.py
│   └── utils/                     # Utility scripts
│       └── set_jupytext_pairing_01.py
├── src/                           # Source code
│   ├── evolution/                 # Evolution algorithms
│   ├── operators/                 # Genetic operators
│   └── solution/                  # Solution representations
├── environment.yml                # Conda environment specification
└── setup.sh                       # Environment setup script
```

## Features

The implementation includes:

1. **Multiple Algorithms**:
   - Hill Climbing
   - Simulated Annealing
   - Genetic Algorithm with multiple configurations

2. **Modular Execution**:
   - Each algorithm can be executed independently
   - All algorithms can be executed in sequence
   - Comparative analysis between algorithms

3. **Visualization**:
   - Convergence plots for each algorithm
   - Comparative plots between configurations
   - Cross-algorithm performance comparison

4. **Statistical Analysis**:
   - Performance metrics (mean, standard deviation)
   - Statistical significance tests (ANOVA, t-tests)
   - Post-hoc analysis (Tukey HSD)

## Setup and Execution

### Environment Setup

1. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate cifo_extended
   ```

2. Run the setup script:
   ```bash
   ./setup.sh
   ```

### Running the Analysis

The notebook `notebooks/phase_1_sp/01_single_processor_analysis.ipynb` provides a modular interface for running the algorithms:

1. **Independent Algorithm Execution**:
   - Run the Hill Climbing cell to execute only Hill Climbing
   - Run the Simulated Annealing cell to execute only Simulated Annealing
   - Run the Genetic Algorithm cell to execute only Genetic Algorithm

2. **Comparative Analysis**:
   - After each algorithm section, run the comparison cell to visualize different configurations
   - At the end, run the cross-algorithm comparison cell to compare all algorithms

3. **Full Execution**:
   - Run the main execution cell to execute all algorithms in sequence
   - Control which algorithms to run using parameters:
     ```python
     main(run_hc=True, run_sa=True, run_ga=True, num_runs=NUM_RUNS)
     ```

## Dependencies

- Python 3.11+
- NumPy, Pandas, Matplotlib
- SciPy (for statistical tests)
- StatsModels (for post-hoc analysis)

## Additional Notes

- The notebook is paired with a Python script using Jupytext for version control
- All paths are relative and cross-platform compatible
- Results are organized by phase for easy comparison
- Statistical tests require multiple runs (NUM_RUNS > 1) for meaningful results
