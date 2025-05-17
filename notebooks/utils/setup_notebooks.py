#!/usr/bin/env python3
"""
setup_notebooks.py - Setup script for Jupyter notebook pairing with kernel management
"""

import json
import subprocess
import sys
from pathlib import Path

def install_packages():
    """Install required Python packages."""
    print("Installing required packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "jupytext", "ipykernel"], check=True)
        print("✅ Required packages installed")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error installing packages: {e}")
        print("Please install manually: pip install jupytext ipykernel")

def setup_jupyter_config():
    """Configure Jupyter to use Jupytext by default."""
    print("Configuring Jupyter...")
    jupyter_dir = Path.home() / ".jupyter"
    jupyter_dir.mkdir(exist_ok=True)
    
    config_path = jupyter_dir / "jupyter_notebook_config.py"
    config_content = """# Jupyter Notebook configuration
c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"
c.ContentsManager.default_jupytext_formats = "ipynb,py:percent"
c.ContentsManager.default_notebook_metadata_filter = "all,-language_info"
c.ContentsManager.default_cell_metadata_filter = "-all"
"""
    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"✅ Jupyter configuration saved to {config_path}")

def setup_project_kernel():
    """Create a project-specific kernel with the correct Python path."""
    print("Setting up project kernel...")
    
    # Get the current Python interpreter path
    python_path = sys.executable
    
    # Find project root
    current_dir = Path.cwd()
    project_root = None
    
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / 'src').exists() and (parent / 'data').exists():
            project_root = parent
            break
    
    if project_root is None:
        project_root = current_dir
        print(f"⚠️ Project root not detected, using current directory: {project_root}")
    else:
        print(f"✅ Project root detected: {project_root}")
    
    # Create a kernel spec
    kernel_name = "cifo_project_kernel"
    kernel_spec = {
        "argv": [
            python_path,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}"
        ],
        "display_name": "CIFO Project Kernel",
        "language": "python",
        "env": {
            "PYTHONPATH": str(project_root)  # Add project root to PYTHONPATH
        }
    }
    
    # Create kernel directory
    kernel_dir = Path.home() / ".local/share/jupyter/kernels" / kernel_name
    kernel_dir.mkdir(parents=True, exist_ok=True)
    
    # Save kernel spec
    with open(kernel_dir / "kernel.json", "w") as f:
        json.dump(kernel_spec, f, indent=2)
    
    print(f"✅ Created project kernel at {kernel_dir}")
    print(f"   Kernel name: {kernel_name}")
    print(f"   Display name: {kernel_spec['display_name']}")
    print(f"   Python path: {python_path}")
    print(f"   PYTHONPATH: {project_root}")

def verify_kernel_setup():
    """Verify the project kernel is properly set up."""
    try:
        kernel_path = Path.home() / ".local/share/jupyter/kernels/cifo_project_kernel/kernel.json"
        if not kernel_path.exists():
            print("❌ Project kernel not found. Run setup_notebooks.py first.")
            return False
            
        with open(kernel_path) as f:
            kernel_spec = json.load(f)
            
        python_path = kernel_spec['argv'][0]
        if not Path(python_path).exists():
            print(f"❌ Python interpreter not found at {python_path}")
            return False
            
        print("✅ Project kernel is properly configured")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying kernel: {e}")
        return False

def reset_kernel():
    """Reset the project kernel configuration."""
    import shutil
    kernel_path = Path.home() / ".local/share/jupyter/kernels/cifo_project_kernel"
    if kernel_path.exists():
        shutil.rmtree(kernel_path)
        print("✅ Removed existing project kernel")
    setup_project_kernel()

def setup_precommit_hook():
    """Set up pre-commit hook to sync notebooks."""
    print("Setting up pre-commit hook...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pre-commit"], check=True)
        
        if not Path('.pre-commit-config.yaml').exists():
            with open('.pre-commit-config.yaml', 'w') as f:
                f.write("""repos:
-   repo: local
    hooks:
    -   id: sync-notebooks
        name: Sync Jupyter notebooks
        entry: python -m jupytext --sync
        language: system
        types: [jupyter]
        always_run: true
        pass_filenames: false
""")
        
        subprocess.run(['pre-commit', 'install'], check=True)
        print("✅ Pre-commit hook installed")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error setting up pre-commit hook: {e}")
        print("You can set it up manually later if needed.")
    except FileNotFoundError:
        print("⚠️ pre-commit not found. Skipping pre-commit hook setup.")
        print("You can set it up manually later if needed.")

def main():
    """Main setup function."""
    try:
        install_packages()
        setup_jupyter_config()
        setup_project_kernel()
        setup_precommit_hook()
        
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Restart your Jupyter server if it's running")
        print("2. Select the 'CIFO Project Kernel' when opening notebooks")
        print("3. Your notebooks will now be automatically paired with .py files")
        print("4. When you commit, notebooks will be automatically synced")
        print("\nTo verify the kernel setup, run:")
        print("  python -c \"from notebooks.utils.setup_notebooks import verify_kernel_setup; verify_kernel_setup()\"")
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
