import torch
class WarmupCosineScheduler:
    """
    Custom cosine schedulers with warmup period
    """

    def __init__(self, optimizer, warmup_epochs, max_lr, total_epochs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = (self.max_lr / self.warmup_epochs) * (self.current_epoch + 1)
        else:
            lr = (
                0.5
                * self.max_lr
                * (
                    1
                    + torch.cos(
                        torch.tensor(
                            torch.pi
                            * (self.current_epoch - self.warmup_epochs)
                            / (self.total_epochs - self.warmup_epochs)
                        )
                    )
                )
            )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_epoch += 1

# sched = WarmupCosineScheduler(opt, warmup_epochs, max_lr, train_epochs)