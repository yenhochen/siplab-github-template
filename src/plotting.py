import matplotlib.pyplot as plt

def plot_samples_from_dataloader(dataloader, save_path=None, nrows=5, ncols=5):
    """
    Plot a grid of sample images from a given dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader containing batches of images and labels.
        save_path (str, optional): Path to save the plotted figure. If None, the plot will be displayed.
        nrows (int, optional): Number of rows in the image grid. Default is 5.
        ncols (int, optional): Number of columns in the image grid. Default is 5.
    
    This function extracts one batch of images from the dataloader and displays
    a grid of sample images. If `save_path` is specified, the plot is saved instead of shown.
    """

    x, y = next(iter(dataloader))

    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * 3, nrows * 3), sharex=True, sharey=True
    )
    axs = axs.flatten()
    for ax, i in zip(axs, x):
        ax.imshow(i.permute(1, 2, 0), cmap="Greys")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_loss_curve(results, save_path=None):
    """
    Plot the training and validation loss curves over epochs.

    Args:
        results (dict): Dictionary containing 'train_loss' and 'val_loss' as keys with list of loss values.
        save_path (str, optional): Path to save the plotted figure. If None, the plot will be displayed.
    
    This function generates a line plot of training and validation loss over epochs.
    """

    fig, axs = plt.subplots(1, 1, figsize=(5, 3))
    axs.plot(results["train_loss"], label="Train Loss")
    axs.plot(results["val_loss"], label="Val Loss")
    axs.legend()
    axs.set_xlabel("Epoch")
    axs.set_ylabel("MSE")
    plt.locator_params("y", nbins=2)
    plt.locator_params("x", nbins=3)
    plt.title("Loss Curve")
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_reconstruction(img, reconstructed, save_path=None):
    """
    Plot original and reconstructed images side by side for comparison.

    Args:
        img (torch.Tensor): Batch of original images with dimensions (b c h w).
        reconstructed (torch.Tensor): Batch of reconstructed images from the autoencoder with dimensions (b c h w).
        save_path (str, optional): Path to save the plotted figure. If None, the plot will be displayed.
    
    This function generates a grid plot where the first row shows original images, and the second
    row shows corresponding reconstructed images from the autoencoder model.
    """

    nrows = 2
    ncols = 10
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * 3, nrows * 3), sharex=True, sharey=True
    )

    for i in range(ncols):
        axs[0, i].imshow(img[i, 0].numpy(), cmap="Greys")
        axs[1, i].imshow(reconstructed[i, 0].numpy(), cmap="Greys")
        axs[0,i].set_xticks([])
        axs[0,i].set_yticks([])
        
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()