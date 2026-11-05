import math


class WarmupCosineSchedule:
    def __init__(self,
                 optimizer,
                 warmup_steps,
                 start_lr,
                 ref_lr,
                 T_max,
                 final_lr=0):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            # Linear warmup phase
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # Cosine decay phase
            # Progress should go from 0 to 1 throughout that phase
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * .5 * (1 + math.cos(math.pi * progress)))
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

        return new_lr

    def state_dict(self):
        """Returns the state of the scheduler as a dictionary."""
        return {"_step": self._step}

    def load_state_dict(self, state_dict):
        """Loads the scheduler state."""
        self._step = state_dict["_step"]
