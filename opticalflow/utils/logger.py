from torch.utils.tensorboard import SummaryWriter


class Logger:
    """"Logger from RAFT."""

    def __init__(self, model, scheduler, SUM_FREQ=100, start_step=0):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = start_step
        self.running_loss = {}
        self.writer = None
        self.SUM_FREQ = SUM_FREQ

    def _print_training_status(self):
        metrics_data = [
            self.running_loss[k] / self.SUM_FREQ
            for k in sorted(self.running_loss.keys())
        ]
        training_str = '[{:6d}, {:10.7f}] '.format(
            self.total_steps + 1,
            self.scheduler.get_last_lr()[0])
        metrics_str = ('{:10.4f}, ' * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / self.SUM_FREQ,
                                   self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.SUM_FREQ == self.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()
