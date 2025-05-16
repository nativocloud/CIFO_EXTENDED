# CIFO Extended Project

## Overview
This project implements and analyzes optimization algorithms (Hill Climbing, Simulated Annealing, and Genetic Algorithms) for solving a team formation problem in sports leagues.

## Project Structure
```
CIFO_EXTENDED_Project/
├── data/                       # Input data
│   └── players.csv             # Player data with skills and costs
├── notebooks/                  # Jupyter notebooks
│   ├── phase_1_sp/             # Single processor analysis
│   │   ├── 01_single_processor_analysis.ipynb
│   │   └── 01_single_processor_analysis.py
│   └── utils/                  # Utility scripts
│       └── set_jupytext_pairing.py
├── results/                    # Results organized by phase
│   └── phase_1_sp/             # Single processor results
│       ├── data/               # CSV and numerical results
│       └── images/             # Plots and visualizations
├── src/                        # Source code
│   ├── evolution/              # Evolution algorithms
│   ├── operators/              # Genetic operators
│   └── solution/               # Solution representation
├── docs/                       # Documentation
│   └── statistical_analysis_guide.md
├── environment.yml             # Conda environment specification
└── setup.sh                    # Environment setup script
```

## Setup and Installation
1. Clone the repository
2. Create and activate the conda environment:
   ```
   conda env create -f environment.yml
   conda activate cifo_extended
   ```
3. Run the setup script:
   ```
   bash setup.sh
   ```

## Jupytext Pairing
This project uses Jupytext to pair Jupyter notebooks with Python scripts for version control. To set up the pairing:

```
python -m notebooks.utils.set_jupytext_pairing
```

This script automatically detects notebooks and configures Jupytext pairing, working in any environment regardless of directory structure.

## Running the Analysis
To run the single processor analysis:

```
cd CIFO_EXTENDED_Project
python -m notebooks.phase_1_sp.01_single_processor_analysis
```

Or open the notebook in Jupyter:
```
jupyter notebook notebooks/phase_1_sp/01_single_processor_analysis.ipynb
```

## Statistical Analysis
The project implements comprehensive statistical analysis including:
- Paired and unpaired non-parametric tests
- Visualization of algorithm performance
- Comparison across multiple metrics

See `docs/statistical_analysis_guide.md` for details on the statistical methodology.

## Metrics Analyzed
- Best Fitness Value
- Average Fitness
- Standard Deviation
- Convergence Speed
- Execution Time
- Success Rate
- Fitness Over Time Curve
