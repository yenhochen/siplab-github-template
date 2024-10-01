import torch
import torch.nn.functional as F


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        logger,
    ):
        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.logger = logger

        self.n_train_samples = len(self.train_loader.dataset)
        self.n_val_samples = len(self.val_loader.dataset)

    def configure_optimizer(
        self, learning_rate, weight_decay, train_epochs, sched_gamma
    ):
        self.logger.info("Setting up Optimizer and Scheduler")
        opt = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=sched_gamma)
        return opt, sched

    def train_step(self, img):
        self.opt.zero_grad()
        img = img.to(self.device)

        x_hat = self.model(img)
        loss = F.mse_loss(x_hat, img)

        loss.backward()
        self.opt.step()
        self.sched.step()
        return loss

    def val_step(self, img):
        self.model.eval()
        img = img.to(self.device)
        x_hat = self.model(img)
        loss = F.mse_loss(x_hat, img)
        return loss

    def train_epoch(
        self,
    ):
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
        #  optimizer args
        learning_rate,
        weight_decay,
        # sched args
        sched_gamma=0.999,
        # train args
        train_epochs=10,
        device="cpu",
    ):
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

        return self.log_dict
