import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.decode_heads import FCNHead


'''
class FCNHeadUseAuxGT(FCNHead):
    """FCNHead that uses data_sample.gt_sem_seg_aux for loss."""

    def loss_by_feat(self, seg_logits, batch_data_samples):
        # 1) collect AUX GT -> (N,H,W)
        gts = []
        for s in batch_data_samples:
            if not hasattr(s, 'gt_sem_seg_aux'):
                raise KeyError(
                    "gt_sem_seg_aux not found. "
                    "Make sure LoadAuxAnnotations + PackSegInputs packs it."
                )

            gt = s.gt_sem_seg_aux.data  # Tensor
            # gt can be (1,H,W) or (H,W)
            if gt.dim() == 3:
                gt = gt.squeeze(0)
            gts.append(gt)

        seg_label = torch.stack(gts, dim=0).long()  # (N,H,W)

        # 2) resize logits to GT size if needed
        if seg_logits.shape[-2:] != seg_label.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits,
                size=seg_label.shape[-2:],
                mode='bilinear',
                align_corners=self.align_corners
            )

        # 3) compute loss (ignore_index already set inside loss_decode config)
        loss = self.loss_decode(seg_logits, seg_label)

        # return key name consistent with mmseg head losses
        return dict(loss_seg=loss)
'''
@MODELS.register_module()
class FCNHeadUseAuxGT2(FCNHead):
    """FCNHead that uses data_sample.gt_sem_seg_aux2 for loss."""

    def loss_by_feat(self, seg_logits, batch_data_samples):
        gts = []

        for s in batch_data_samples:
            if not hasattr(s, 'gt_sem_seg_aux2'):
                raise KeyError("gt_sem_seg_aux2 not found.")

            gt = s.gt_sem_seg_aux2.data

            if gt.dim() == 3:
                gt = gt.squeeze(0)

            gts.append(gt)

        seg_label = torch.stack(gts, dim=0).long()

        if seg_logits.shape[-2:] != seg_label.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits,
                size=seg_label.shape[-2:],
                mode='bilinear',
                align_corners=self.align_corners
            )

        loss = self.loss_decode(
            seg_logits,
            seg_label,
            ignore_index=255
        )

        return dict(loss_seg=loss)