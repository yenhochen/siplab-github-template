import matplotlib.pyplot as plt

def plot_samples_from_dataloader(dataloader, save_path=None, nrows=5, ncols=5):
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
    visualizes two
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