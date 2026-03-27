from mmengine.hooks import Hook
from mmengine.registry import HOOKS
'''
@HOOKS.register_module()

class AverageIterLossHook(Hook):
    def __init__(self, interval=295):
        self.interval = interval

    def before_train(self, runner):
        self.total_loss = 0.0
        self.iter_count = 0

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        loss = outputs['loss']
        self.total_loss += loss.item()
        self.iter_count += 1

        if self.iter_count % self.interval == 0:
            avg_loss = self.total_loss / self.interval
            runner.logger.info(f"[Iter {runner.iter + 1}] Avg Training Loss (last {self.interval} iters): {avg_loss:.6f}")
            runner.message_hub.update_scalar('avg_train_loss', avg_loss)
            # Reset counters
            self.total_loss = 0.0
            self.iter_count = 0
'''

from collections import deque

@HOOKS.register_module()
class AverageIterLossHook(Hook):
    def __init__(self, interval_to_log=1000, window=455):
        self.interval_to_log = interval_to_log  # When to log
        self.window = window                    # How many losses to average
        self.loss_queue = deque(maxlen=window)

    def before_train(self, runner):
        self.iter_count = 0

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        loss = outputs['loss']
        self.loss_queue.append(loss.item())
        self.iter_count += 1

        if self.iter_count % self.interval_to_log == 0:
            if len(self.loss_queue) == self.window:
                avg_loss = sum(self.loss_queue) / self.window
                runner.logger.info(
                    f"[Iter {runner.iter + 1}] Avg Training Loss (last {self.window} iters): {avg_loss:.6f}")
                runner.message_hub.update_scalar('avg_train_loss', avg_loss)
            else:
                runner.logger.warning(
                    f"[Iter {runner.iter + 1}] Not enough iterations to compute avg over last {self.window}")
