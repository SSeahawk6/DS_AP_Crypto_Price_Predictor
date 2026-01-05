import os
import random
import numpy as np

def ensure_directories_exist(directories):
    """
    Ensures that the specified directories exist.
    
    Args:
        directories (list): A list of directory paths (strings).
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def setup_environment(seed=42):
    """
    Sets random seeds for reproducibility and creates standard directories.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Create necessary directories using the helper function
    ensure_directories_exist(['data', 'results', 'models'])
        
    print(f"[INFO] Environment setup complete. Random Seed: {seed}")