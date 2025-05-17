# CIFO Extended Project

## Overview
This repository contains the implementation of optimization algorithms for the Sports League Assignment Problem, with both single-processor and multi-processor analysis capabilities.

## Project Structure
```
CIFO_EXTENDED_Project/
├── data/                       # Data files
│   └── players.csv             # Player data with skills and costs
├── notebooks/                  # Jupyter notebooks
│   ├── phase_1_sp/             # Single processor analysis
│   │   ├── 01_single_processor_analysis.ipynb
│   │   └── 01_single_processor_analysis.py
│   └── utils/                  # Utility scripts
│       ├── init_notebook.py    # Notebook initialization
│       ├── pair_notebooks.py   # Notebook-script pairing
│       └── setup_notebooks.py  # Project kernel setup
├── results/                    # Results directory
│   └── phase_1_sp/             # Single processor results
│       ├── images/             # Generated plots
│       └── data/               # Generated data files
├── src/                        # Source code
│   ├── evolution/              # Evolutionary algorithms
│   ├── operators/              # Genetic operators
│   └── solution/               # Problem-specific solution
├── docs/                       # Documentation
│   ├── kernel_management_guide.md  # Kernel setup guide
│   └── notebook_pairing_guide.md   # Notebook pairing guide
├── environment.yml             # Conda environment file
└── setup.sh                    # Setup script
```

## Setup Instructions

### Environment Setup
1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate cifo_extended
```

2. Run the setup script:
```bash
bash setup.sh
```

### Jupyter Notebook Setup
For consistent notebook execution across environments:

1. Set up the project kernel and Jupytext pairing:
```bash
python -m notebooks.utils.setup_notebooks
```

2. Pair all notebooks with Python scripts:
```bash
python -m notebooks.utils.pair_notebooks --all
```

3. When opening notebooks, select the "CIFO Project Kernel"

### Running the Analysis
1. Navigate to the notebooks directory:
```bash
cd notebooks/phase_1_sp
```

2. Open the Jupyter notebook:
```bash
jupyter notebook 01_single_processor_analysis.ipynb
```

3. Make sure to run the initialization cell at the beginning:
```python
# Initialize notebook environment
%run ../utils/init_notebook.py
```

## Documentation

### Guides
- [Kernel Management Guide](docs/kernel_management_guide.md): How to set up and use the project kernel
- [Notebook Pairing Guide](docs/notebook_pairing_guide.md): How to maintain notebook-script synchronization

### Statistical Analysis
The project implements both parametric and non-parametric statistical tests to compare algorithm performance:
- For two algorithms: Wilcoxon (paired) and Mann-Whitney U (unpaired) tests
- For multiple algorithms: Friedman+Nemenyi (paired) and Kruskal-Wallis+Dunn (unpaired) tests

### Metrics
The following metrics are analyzed:
- Best Fitness Value
- Average Fitness
- Standard Deviation
- Convergence Speed
- Execution Time
- Success Rate
- Fitness Over Time Curve

## Development Workflow
This project uses Jupytext to pair notebooks with Python scripts for better version control:

1. Edit either the notebook (.ipynb) or the Python script (.py)
2. Changes are automatically synchronized between the paired files
3. Commit the Python scripts to version control
4. The .gitignore is configured to handle temporary files and outputs
