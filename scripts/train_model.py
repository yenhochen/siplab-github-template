import os
import sys

# Add root path to Python's system path for module imports
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

import argparse
import logging

from src.load_datasets import get_fmnist_loader, get_mnist_loader
from src.utils import load_config, seed_everything
from src.plotting import plot_reconstruction, plot_loss_curve
from src.train import Trainer
from src.models import Autoencoder
import torch


def main():
    # Set up logging to display info messages in the console with a timestamp and log level
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Plot samples from data loaders.")
    parser.add_argument(
        "--config", type=str, required=True, help="path to yaml config file"
    )
    args = parser.parse_args()

    # Load configurations from the YAML config file
    logger.info(f"Loading Config From {os.path.abspath(args.config)}")
    config = load_config(args.config)

    # Set random seed for reproducibility
    logger.info(f"Setting random seed to {config['seed']}")
    seed_everything(config["seed"])

    # Load the dataset based on the config
    logger.info(
        f"Loading {config['dataset']} dataset with batch size {config['batch_size']}"
    )

    # Conditionally load MNIST or FashionMNIST based on the config
    if config["dataset"] == "mnist":
        trainloader, valloader = get_mnist_loader(
            config["data_dir"], batch_size=config["batch_size"]
        )
    elif config["dataset"] == "fmnist":
        trainloader, valloader = get_fmnist_loader(
            config["data_dir"], batch_size=config["batch_size"]
        )
    else:
        logger.error("Dataset must be either 'mnist' or 'fmnist' ")


    # Log details about the model being initialized
    logger.info(f"Initializing model with:")
    logger.info(f"\timg height, width {config['img_height']}, {config['img_width']}")
    logger.info(f"\thidden dim {config['hidden_dim']}")
    logger.info(f"\tlatent dim {config['latent_dim']}")
    logger.info(f"\tn layers {config['n_layers']}")
    logger.info(f"\tlayernorm {config['layernorm']}")
    logger.info(f"on device: {config['device']}")

    # Initialize the autoencoder model with the config parameters
    model = Autoencoder(
        config["img_height"],
        config["img_width"],
        config["hidden_dim"],
        config["latent_dim"],
        config["n_layers"],
        layernorm=config["layernorm"],
    )
    model = model.to(config["device"])

    # Log training details
    logger.info(f"Training Model with:")
    logger.info(f"\ttrain epochs: {config['train_epochs']}")
    logger.info(f"\tlearning rate: {config['learning_rate']}")
    logger.info(f"\tweight decay: {config['weight_decay']}")
    logger.info(f"\tsched gamma: {config['sched_gamma']}")

    # Initialize the trainer object and start training
    trainer = Trainer(model, trainloader, valloader, logger)
    results = trainer.train(
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        train_epochs=config["train_epochs"],
        sched_gamma=config["sched_gamma"],
        device=config["device"],
    )
    logger.info(f"Training complete!")

    # Save the loss curve plot to a file
    loss_curve_path = f"figs/{config['dataset']}_loss_curve.png"
    plot_loss_curve(results, loss_curve_path)
    logger.info(f"Saving loss curve plot to {loss_curve_path}")
    
    # Plot a reconstruction of a single batch from the validation set
    with torch.no_grad():
        img, label = next(iter(valloader))
        img = img.to(config["device"])
        reconstructed = model(img)
        img = img.cpu()
        reconstructed = reconstructed.cpu()

    # Save the reconstruction plot to a file
    reconsturction_path = f"figs/{config['dataset']}_reconstruction.png"
    plot_reconstruction(img, reconstructed, reconsturction_path)
    logger.info(f"Saving reconstruction plot to {reconsturction_path}")


if __name__ == "__main__":
    main()
