# CIFO Extended Project

## Project Structure
```
project_root/
├── data/                       # Data files
├── notebooks/                  # Jupyter notebooks
│   ├── phase_1_sp/            # Single processor analysis
│   └── utils/                 # Notebook utility scripts
├── results/                    # Analysis results
│   └── phase_1_sp/            # Single processor results
│       ├── data/              # Generated data
│       └── images/            # Generated plots
├── src/                        # Source code
│   ├── evolution/             # Evolutionary algorithms
│   ├── operators/             # Genetic operators
│   └── solution/              # Problem-specific solution
├── docs/                       # Documentation
│   ├── environment_setup_guide.md
│   ├── kernel_management_guide.md
│   ├── notebook_initialization_guide.md
│   ├── notebook_pairing_guide.md
│   └── statistical_analysis_guide.md
├── environment.yml             # Conda environment file
└── setup.sh                    # Setup script
```

## Setup Instructions

### Prerequisites
- Conda (Anaconda ou Miniconda)
- Git Bash (para utilizadores Windows)

### Environment Setup

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate cifo_extended
```

2. Run the setup script:
   - On Linux/macOS:
   ```bash
   bash setup.sh
   ```
   - On Windows (using Git Bash):
   ```bash
   bash setup.sh
   ```

3. If you encounter path issues, you can manually install the kernel:
```bash
python -m pip install ipykernel
python -m ipykernel install --user --name=project_kernel --display-name="Project Kernel"
```

### Jupyter Notebook Setup

1. Set up the project kernel and Jupytext pairing:
```bash
python -m notebooks.utils.setup_notebooks
```

2. Pair all notebooks with Python scripts:
```bash
python -m notebooks.utils.pair_notebooks --all
```

3. Start Jupyter:
```bash
jupyter lab  # or jupyter notebook
```

4. When opening notebooks, add the following code to the first cell for consistent initialization:
```python
import sys
import os

# Get the absolute path to the utils directory
utils_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'utils'))
if utils_path not in sys.path:
    sys.path.append(utils_path)

# Now import the initialization module
from init_notebook import *

print("✅ Notebook environment initialized successfully!")
```

## Documentation

Refer to the following guides in the docs/ directory:

- environment_setup_guide.md - Detailed environment setup
- kernel_management_guide.md - Managing Jupyter kernels
- notebook_initialization_guide.md - Notebook initialization
- notebook_pairing_guide.md - Using Jupytext for notebook pairing
- statistical_analysis_guide.md - Statistical analysis procedures

## Development Workflow

This project uses Jupytext to pair notebooks with Python scripts for better version control:

1. Edit either the notebook (.ipynb) or the Python script (.py)
2. Changes are automatically synchronized between the paired files
3. Commit both the Python scripts and notebooks to version control
4. The .gitignore is configured to handle temporary files and outputs

## Statistical Analysis

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

## Troubleshooting

### Common Issues

- **Kernel not found in Jupyter**:
  - Ensure the environment is activated
  - Run: `python -m ipykernel install --user --name=project_kernel`
  - Restart Jupyter

- **setup.sh issues on Windows**:
  - Use Git Bash instead of CMD/PowerShell
  - Ensure no spaces in the path (move the project if needed)
  - Run as administrator if permission issues occur

- **Environment activation issues**:
  - Close and reopen your terminal after conda installation
  - Use `conda init` if conda commands aren't recognized
