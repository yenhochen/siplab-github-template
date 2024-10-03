import random
import numpy as np
import torch
import yaml
import pickle
from src.models import Autoencoder


def seed_everything(seed):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed (if available)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benschmark for reproducibility


def load_config(yaml_path):
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def save_pickle(results, save_path):
    with open(save_path, 'wb') as file:
        pickle.dump(results, file)

def load_pickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def load_model(config):
    loaded_model = Autoencoder(
        config["img_height"],
        config["img_width"],
        config["hidden_dim"],
        config["latent_dim"],
        config["n_layers"],
        layernorm=config["layernorm"],
    )

    # Load the state dictionary
    loaded_model.load_state_dict(torch.load(config["save_model_path"], weights_only=True))
    loaded_model.eval()
    return loaded_model