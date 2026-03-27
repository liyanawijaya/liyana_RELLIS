from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS
import torch
@HOOKS.register_module()
class ValidationLossHook(Hook):
    def __init__(self, log_interval=500):
        self.log_interval = log_interval

    def after_val_iter(self, runner, batch_idx, data_batch, outputs):
        model = runner.model
        model.eval()

        with torch.no_grad():
            try:
                inputs = data_batch["inputs"]
                data_samples = data_batch["data_samples"]

                print(f"inputs type: {type(inputs)}")
                print(f"inputs len: {len(inputs)}")
                print(f"inputs[0] type: {type(inputs[0])}")

                print(f"data_samples type: {type(data_samples)}")
                print(f"data_samples len: {len(data_samples)}")
                print(f"data_samples[0] type: {type(data_samples[0])}")

                # Try just first sample for debug
                losses = model(inputs=inputs, data_samples=data_samples, mode='loss')

                loss, log_vars = model.parse_losses(losses)

                for name, value in log_vars.items():
                    runner.message_hub.update_scalar(f'val/{name}', value)

                if (batch_idx + 1) % self.log_interval == 0:
                    runner.logger.info(
                        f'[Val][Iter {batch_idx + 1}] ' +
                        ', '.join([f'{k}: {v:.4f}' for k, v in log_vars.items()])
                    )
            except Exception as e:
                runner.logger.error(f"ValidationLossHook error at iter {batch_idx}: {e}")
                runner.logger.error(f"data_batch keys: {list(data_batch.keys())}")
                raise e
