import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

import argparse
import logging

from src.load_datasets import get_fmnist_loader, get_mnist_loader
from src.utils import load_config, seed_everything
from src.plotting import plot_reconstruction, plot_loss_curve
from src.utils import load_model, load_pickle
import torch


def main():
    # initialize logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Plot samples from data loaders.")
    parser.add_argument(
        "--config", type=str, required=True, help="path to yaml config file"
    )
    args = parser.parse_args()

    # load configs
    logger.info(f"Loading Config From {os.path.abspath(args.config)}")
    config = load_config(args.config)

    # set random seeds
    logger.info(f"Setting random seed to {config['seed']}")
    seed_everything(config["seed"])

    # load data
    logger.info(
        f"Loading {config['dataset']} dataset with batch size {config['batch_size']}"
    )

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


    # load results and model
    logger.info(f"Loading pickle from to {config['save_results_path']}")
    loaded_results = load_pickle(config["save_results_path"])

    logger.info(f"Loading model from to {config['save_model_path']}")
    logger.info(f"\timg height, width {config['img_height']}, {config['img_width']}")
    logger.info(f"\thidden dim {config['hidden_dim']}")
    logger.info(f"\tlatent dim {config['latent_dim']}")
    logger.info(f"\tn layers {config['n_layers']}")
    logger.info(f"\tlayernorm {config['layernorm']}")
    logger.info(f"on device: {config['device']}")
    loaded_model = load_model(config)
    loaded_model = loaded_model.to(config["device"])

    # loss_curve_path = f"figs/{config['dataset']}_loss_curve.png"
    # reconsturction_path = f"figs/{config['dataset']}_reconstruction.png"
    plot_loss_curve(loaded_results, config['loss_curve_path'])
    logger.info(f"Saving loss curve plot to {config['loss_curve_path']}")
    
    # plot inference for a single batch from the val set
    with torch.no_grad():
        img, label = next(iter(valloader))
        img = img.to(config["device"])
        reconstructed = loaded_model(img)
        img = img.cpu()
        reconstructed = reconstructed.cpu()
    
    plot_reconstruction(img, reconstructed, config["reconstruction_path"])
    logger.info(f"Saving reconstruction plot to {config['reconstruction_path']}")


if __name__ == "__main__":
    main()
