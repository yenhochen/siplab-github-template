import random
import numpy as np
import torch
import yaml


def seed_everything(seed):
    """
    Sets a seed for all random number generators to ensure reproducibility across different libraries.
    
    Args:
        seed (int): The seed to be used for all libraries (random, numpy, torch).
    
    Behavior:
        - Sets the seed for Python's random library.
        - Sets the seed for NumPy.
        - Sets the seed for PyTorch (both CPU and GPU).
        - Ensures PyTorch uses deterministic algorithms by setting `torch.backends.cudnn.deterministic = True`.
        - Disables the cuDNN benchmark for reproducibility.
    """

    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed (if available)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benschmark for reproducibility


def load_config(yaml_path):
    """
    Loads a YAML configuration file and returns the configuration as a dictionary.
    
    Args:
        yaml_path (str): The path to the YAML configuration file.
    
    Returns:
        config (dict): The configuration loaded from the YAML file.
    """
    
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config


