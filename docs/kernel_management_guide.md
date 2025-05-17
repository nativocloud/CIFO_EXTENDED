# Kernel Management and Notebook Pairing Guide

This guide explains how to set up a project-specific kernel and ensure consistent notebook-script pairing across different environments.

## Table of Contents
- [Quick Setup](#quick-setup)
- [Understanding Kernel Issues](#understanding-kernel-issues)
- [Project Kernel Setup](#project-kernel-setup)
- [Notebook-Script Pairing](#notebook-script-pairing)
- [Troubleshooting](#troubleshooting)

## Quick Setup

For a complete environment setup, run:

```bash
python -m notebooks.utils.setup_notebooks
```

This script will:
1. Install required packages (jupytext, ipykernel)
2. Configure Jupyter to use Jupytext
3. Create a project-specific kernel with the correct PYTHONPATH
4. Set up pre-commit hooks for automatic notebook synchronization

## Understanding Kernel Issues

When working with Jupyter notebooks across different environments, several issues can arise:

1. **Kernel Name Mismatches**: Different environments may have different kernel names
2. **Path Resolution**: The kernel may not have the correct PYTHONPATH
3. **Dependency Management**: Different kernels may have different package versions
4. **Notebook Metadata**: Kernel specifications in notebook metadata may cause conflicts

The project-specific kernel solves these issues by:
- Using a consistent kernel name across environments
- Setting the correct PYTHONPATH in the kernel environment
- Ensuring all dependencies are available to the kernel
- Allowing Jupytext to ignore kernel specification differences

## Project Kernel Setup

### Automatic Setup

Run the setup script:

```bash
python -m notebooks.utils.setup_notebooks
```

### Manual Setup

If you prefer to set up the kernel manually:

1. Create a kernel specification:
   ```bash
   python -m ipykernel install --user --name=cifo_project_kernel --display-name="CIFO Project Kernel"
   ```

2. Edit the kernel.json file to add the project root to PYTHONPATH:
   ```json
   {
     "argv": [
       "python",
       "-m",
       "ipykernel_launcher",
       "-f",
       "{connection_file}"
     ],
     "display_name": "CIFO Project Kernel",
     "language": "python",
     "env": {
       "PYTHONPATH": "/path/to/project/root"
     }
   }
   ```

### Verifying Kernel Setup

To verify the kernel is set up correctly:

```bash
python -c "from notebooks.utils.setup_notebooks import verify_kernel_setup; verify_kernel_setup()"
```

## Notebook-Script Pairing

### Automatic Pairing

To pair all notebooks in the project:

```bash
python -m notebooks.utils.pair_notebooks --all
```

### Using the Project Kernel

When opening a notebook:

1. Select "Kernel" > "Change kernel" > "CIFO Project Kernel"
2. Add the initialization code to the first cell:
   ```python
   # Initialize notebook environment
   %run ../utils/init_notebook.py
   ```

### Handling Kernel Differences

The setup ensures that:
- Kernel specifications are normalized across environments
- Jupytext ignores kernel differences when pairing
- The correct PYTHONPATH is always available

## Troubleshooting

### Kernel Not Found

If the CIFO Project Kernel is not available:

1. Run the setup script again:
   ```bash
   python -m notebooks.utils.setup_notebooks
   ```

2. Restart Jupyter if it was already running

### Pairing Not Working

If notebook-script pairing is not working:

1. Check Jupytext is installed:
   ```bash
   pip install jupytext
   ```

2. Manually pair the notebook:
   ```bash
   python -m notebooks.utils.pair_notebooks path/to/notebook.ipynb
   ```

3. Verify the notebook metadata contains the correct Jupytext configuration

### Import Errors

If you see import errors when running notebooks:

1. Make sure you're using the CIFO Project Kernel
2. Run the initialization script:
   ```python
   %run ../utils/init_notebook.py
   ```
3. Verify the project structure is correct (src/ and data/ directories exist)
