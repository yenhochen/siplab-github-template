import torch
class WarmupCosineScheduler:
    """
    Custom learning rate scheduler that performs warmup followed by a cosine annealing schedule.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be updated.
        warmup_epochs (int): Number of warmup epochs where the learning rate increases linearly.
        max_lr (float): Maximum learning rate reached after the warmup period.
        total_epochs (int): Total number of epochs for the entire training process.

    During the warmup phase, the learning rate increases linearly from 0 to max_lr over
    the specified number of warmup epochs. After warmup, the learning rate follows a 
    cosine annealing schedule, starting at max_lr and decaying to near 0 by the end of the training.
    """

    def __init__(self, optimizer, warmup_epochs, max_lr, total_epochs):
        """
        Initializes the scheduler with the optimizer, warmup period, maximum learning rate, and total epochs.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer whose learning rate is to be updated.
            warmup_epochs (int): Number of epochs over which to linearly increase the learning rate.
            max_lr (float): Maximum learning rate reached after the warmup period.
            total_epochs (int): Total number of training epochs, including warmup and cosine decay.
        """

        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def step(self):
        """
        Update the learning rate based on the current epoch.

        During the warmup period, the learning rate increases linearly. Afterward, it
        follows a cosine annealing schedule to gradually reduce the learning rate.
        """
        
        if self.current_epoch < self.warmup_epochs: 
            # Linear warmup phase
            lr = (self.max_lr / self.warmup_epochs) * (self.current_epoch + 1)
        else:
            # Cosine annealing phase
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
        # Apply the calculated learning rate to each parameter group in the optimizer
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
        self.current_epoch += 1

# sched = WarmupCosineScheduler(opt, warmup_epochs, max_lr, train_epochs)