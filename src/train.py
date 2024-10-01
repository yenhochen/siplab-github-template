import torch
import torch.nn.functional as F

class Trainer:
    """
    A class to manage the training and validation of a model using PyTorch.
    
    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        logger (logging.Logger): Logger to record training progress.

    Attributes:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        logger (logging.Logger): Logger to record training progress.
        n_train_samples (int): Number of training samples.
        n_val_samples (int): Number of validation samples.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        logger,
    ):
        """
        Initializes the Trainer with the model, data loaders, and logger.

        Args:
            model (torch.nn.Module): Model to be trained.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
            logger (logging.Logger): Logger to record the training process.
        """

        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.logger = logger

        self.n_train_samples = len(self.train_loader.dataset)
        self.n_val_samples = len(self.val_loader.dataset)

    def configure_optimizer(
        self, learning_rate, weight_decay, train_epochs, sched_gamma
    ):
        """
        Sets up the optimizer and learning rate scheduler for the training.

        Args:
            learning_rate (float): The learning rate for the optimizer.
            weight_decay (float): Weight decay for regularization.
            train_epochs (int): Total number of training epochs.
            sched_gamma (float): The gamma value for the ExponentialLR scheduler.

        Returns:
            opt (torch.optim.Optimizer): Optimizer (Adam) for training.
            sched (torch.optim.lr_scheduler.ExponentialLR): Learning rate scheduler.
        """

        self.logger.info("Setting up Optimizer and Scheduler")
        opt = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=sched_gamma)
        return opt, sched

    def train_step(self, img):
        """
        Performs a single training step, including forward pass, loss computation, and backpropagation.

        Args:
            img (torch.Tensor): Batch of input images (b, c, h, w).

        Returns:
            loss (torch.Tensor): Loss for the current batch.
        """
                
        self.opt.zero_grad()
        img = img.to(self.device)

        x_hat = self.model(img)
        loss = F.mse_loss(x_hat, img)

        loss.backward()
        self.opt.step()
        self.sched.step()
        return loss

    def val_step(self, img):
        """
        Performs a single validation step, computing the loss without backpropagation.

        Args:
            img (torch.Tensor): Batch of input images (b c h w).

        Returns:
            loss (torch.Tensor): Loss for the current batch.
        """

        self.model.eval()
        img = img.to(self.device)
        x_hat = self.model(img)
        loss = F.mse_loss(x_hat, img)
        return loss

    def train_epoch(
        self,
    ):
        """
        Trains the model for one epoch by iterating over the entire training set.

        Returns:
            train_epoch_loss (float): The average training loss for the epoch.
        """

        self.model.train()
        train_epoch_loss = 0
        for img, label in self.train_loader:
            loss = self.train_step(img)
            train_epoch_loss += loss.item()
        train_epoch_loss /= self.n_train_samples
        return train_epoch_loss

    def val_epoch(
        self,
    ):
        """
        Validates the model for one epoch by iterating over the entire validation set.

        Returns:
            val_epoch_loss (float): The average validation loss for the epoch.
        """

        with torch.no_grad():
            self.model.eval()
            val_epoch_loss = 0
            for img, label in self.val_loader:
                loss = self.val_step(img)
                val_epoch_loss += loss.item()
            val_epoch_loss /= self.n_val_samples
        return val_epoch_loss

    def train(
        self,
        learning_rate,
        weight_decay,
        sched_gamma=0.999,
        train_epochs=10,
        device="cpu",
    ):
        """
        Trains the model for a specified number of epochs and logs the results.

        Args:
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for regularization.
            sched_gamma (float, optional): The gamma value for the ExponentialLR scheduler (default: 0.999).
            train_epochs (int, optional): Number of training epochs (default: 10).
            device (str, optional): Device to perform training on ('cpu' or 'cuda') (default: 'cpu').

        Returns:
            results (dict): A dictionary containing the training and validation losses as well as learning rate at each epoch.
        """

        self.log_dict = {"train_loss": [], "val_loss": [], "learning_rate": []}

        self.opt, self.sched = self.configure_optimizer(
            learning_rate, weight_decay, train_epochs, sched_gamma
        )
        self.device = device
        for epoch in range(train_epochs):
            train_epoch_loss = self.train_epoch()
            val_epoch_loss = self.val_epoch()

            lr = self.opt.param_groups[0]["lr"]

            self.logger.info(
                f"epoch: {epoch}\tTrain loss: {train_epoch_loss:.6f}\tVal loss: {val_epoch_loss:.6f}\tlearning rate: {lr:6f}"
            )

            self.log_dict["train_loss"].append(train_epoch_loss)
            self.log_dict["val_loss"].append(val_epoch_loss)
            self.log_dict["learning_rate"].append(lr)

        results = self.log_dict
        return results
