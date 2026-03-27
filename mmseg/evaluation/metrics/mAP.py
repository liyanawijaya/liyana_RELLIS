from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
import numpy as np
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F

@METRICS.register_module()
class CustomSegmentationMAP(BaseMetric):
    def __init__(self, num_classes, name='mAP', resize_to=(192, 108), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.num_classes = num_classes
        self.resize_to = resize_to
        self.predictions = []
        self.ground_truths = []
        self.confidences = []

    def downsample(self, tensor, size):
        """Downsample a [H, W] tensor to [H', W']"""
        tensor = tensor.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        resized = F.interpolate(tensor, size=size, mode='nearest')
        return resized.squeeze(0).squeeze(0).long()

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            pred_mask = sample['pred_sem_seg']['data'].squeeze()  # [H, W]
            gt_mask = sample['gt_sem_seg']['data'].squeeze().to(pred_mask)  # [H, W]

            # Downsample both to reduce memory
            pred_mask = self.downsample(pred_mask, self.resize_to)
            gt_mask = self.downsample(gt_mask, self.resize_to)

            # Get confidence map
            if 'seg_logits' in sample:
                logits = sample['seg_logits']['data']  # [C, H, W]
                softmax = logits.softmax(dim=0)        # [C, H, W]
                confidence, _ = softmax.max(dim=0)     # [H, W]
                confidence = F.interpolate(
                    confidence.unsqueeze(0).unsqueeze(0),
                    size=self.resize_to,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            else:
                confidence = (pred_mask >= 0).float()  # dummy confidence = 1.0

            # Move to CPU and store
            self.predictions.append(pred_mask.cpu().numpy())
            self.ground_truths.append(gt_mask.cpu().numpy())
            self.confidences.append(confidence.cpu().numpy())

            # Optional: Free memory
            del pred_mask, gt_mask, confidence, sample
            torch.cuda.empty_cache()

        # Add dummy result to avoid MMEngine warning
        self.results.append(1)
    """
    def compute_metrics(self, results):
        per_class_ap = []

        for cls in range(self.num_classes):
            y_true = []
            y_score = []

            for pred, gt, conf in zip(self.predictions, self.ground_truths, self.confidences):
                mask = (gt == cls)
                y_true.extend(mask.flatten())

                pred_cls = (pred == cls)
                score = pred_cls * conf
                y_score.extend(score.flatten())

            if np.sum(y_true) == 0:
                continue  # skip if class not present in GT

            ap = average_precision_score(y_true, y_score)
            per_class_ap.append(ap)

        mAP = np.mean(per_class_ap) if per_class_ap else 0.0
        return {self.name: mAP}
    """
    def compute_metrics(self, results):
        class_names = [
        "Tree", "Leaves", "Ground", "Fence", "Grass", "Log", "Grasstree", "Sky",
        "Road sign", "Small branch", "Delineator", "Asphalt", "Rubble", "Rock", "Non-nav", "Rough"
        ]

        per_class_ap = []
        class_wise_ap = {}

        for cls in range(self.num_classes):
            y_true = []
            y_score = []

            for pred, gt, conf in zip(self.predictions, self.ground_truths, self.confidences):
                mask = (gt == cls)
                y_true.extend(mask.flatten())

                pred_cls = (pred == cls)
                score = pred_cls * conf
                y_score.extend(score.flatten())

            if np.sum(y_true) == 0:
                continue  # skip if class not present in GT

            ap = average_precision_score(y_true, y_score)
            per_class_ap.append(ap)

            class_name = class_names[cls] if cls < len(class_names) else f'class_{cls}'
            class_wise_ap[f'AP_{class_name}'] = ap

        mAP = np.mean(per_class_ap) if per_class_ap else 0.0

    # Combine results
        metrics = {self.name: mAP}
        metrics.update(class_wise_ap)
        return metrics

    def reset(self):
        super().reset()
        self.predictions = []
        self.ground_truths = []
        self.confidences = []
