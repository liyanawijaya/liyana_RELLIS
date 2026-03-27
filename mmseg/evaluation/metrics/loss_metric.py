from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class CustomLossMetric(BaseMetric):
    def __init__(self, loss_keys=None, name='avg_val_loss', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_keys = loss_keys or ['decode.loss_ce', 'aux_0.loss_ce', 'aux_1.loss_ce', 'aux_2.loss_ce', 'aux_3.loss_ce']
        self.name = name
        #self.results_1 = []

    def process(self, data_batch, data_samples):
        #print(f"Processing batch — current results length: {len(self.results)}")
        #print(len(data_samples))
        #for sample in data_samples:
            #metrics = sample.get('metrics', None)
            #print(f"[DEBUG] Extracted metrics: {metrics}")
            #if metrics:
            #    total = sum(metrics[k] for k in self.loss_keys if k in metrics)
            #    self.results_1.append(total)
            #    print(f"Current loss: {total:.4f}, Running sum: {sum(self.results_1):.4f}")
        """
        for sample in data_samples:
            metrics = sample.get('metrics', {})
            total = sum(metrics[k] for k in self.loss_keys if k in metrics)
            self.results.append(total)  # uses BaseMetric's result collection
            print(f"Total loss per image: {total}")
            print(f"[Sample {idx}] Loss keys: {present_keys}, Total: {total:.4f}, Running sum: {sum(self.results_1):.4f}")
        """
        for idx, sample in enumerate(data_samples):
            metrics = sample.get('metrics', {})
            present_keys = [k for k in self.loss_keys if k in metrics]
            total = sum(metrics[k] for k in present_keys)
            self.results.append(total)


            #print(f"[Sample {idx}] Loss keys: {present_keys}, Total: {total:.4f}, Running sum: {sum(self.results):.4f}")
            
    def compute_metrics(self, results):
        """
        avg_loss = sum(self.results_1) / len(self.results_1) if self.results_1 else 0.0
        print(f"Average_loss: {avg_loss}")
        print(len(self.results_1))
        output={self.name: avg_loss}
        self.results_1 = []
        return output
        """

            
        avg_loss = sum(results) / len(results) if results else 0.0

        #print(f"Total data samples: {len(results)}")
        #print(f"Final Average Loss: {avg_loss:.4f}")
        #print(f"Average Loss: {sum(results)/len(results):.4f}")
        return {self.name:avg_loss}
    def reset(self): 
        print("RESET called — clearing results")
        super().reset()
