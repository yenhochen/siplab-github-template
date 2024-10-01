import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

import argparse
import logging

from src.load_datasets import get_fmnist_loader, get_mnist_loader
from src.utils import load_config, plot_samples_from_dataloader, seed_everything


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
    config.get("nrows", 5)
    config.get("ncols", 5)

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

    # plot data from dataloaders
    plot_samples_from_dataloader(
        trainloader,
        f"figs/{config['dataset']}_train_samples.png",
        nrows=config["nrows"],
        ncols=config["ncols"],
    )

    plot_samples_from_dataloader(
        valloader,
        f"figs/{config['dataset']}_val_samples.png",
        nrows=config["nrows"],
        ncols=config["ncols"],
    )


if __name__ == "__main__":
    main()
