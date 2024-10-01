import torch
import random
import numpy as np
import yaml
import matplotlib.pyplot as plt

def seed_everything(seed):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed (if available)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benschmark for reproducibility

def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def plot_samples_from_dataloader(dataloader, save_path, nrows=5, ncols=5):
    x, y = next(iter(dataloader))

    fig, axs = plt.subplots(nrows, ncols, figsize=(nrows*3,ncols*3), sharex=True, sharey=True)
    axs = axs.flatten()
    for ax, i in zip(axs, x):
        ax.imshow(i.permute(1,2,0), cmap="Greys")

    plt.tight_layout()
    plt.savefig(save_path)