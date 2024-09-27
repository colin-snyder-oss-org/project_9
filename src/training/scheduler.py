# src/training/scheduler.py
import torch

class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, initial_lr, target_lr, num_steps):
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.num_steps = num_steps
        self.current_step = 0
        super(CustomLRScheduler, self).__init__(optimizer)

    def get_lr(self):
        factor = self.current_step / self.num_steps
        lr = self.initial_lr + factor * (self.target_lr - self.initial_lr)
        return [lr for _ in self.base_lrs]

    def step(self):
        self.current_step += 1
        super(CustomLRScheduler, self).step()
