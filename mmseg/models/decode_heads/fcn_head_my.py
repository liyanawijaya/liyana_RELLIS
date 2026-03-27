import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.decode_heads import FCNHead

@MODELS.register_module()
class DepthAwareFCNHead(FCNHead):
    """FCNHead that passes depth to a depth-aware loss (e.g., DepthAwareCrossEntropyLoss).

    Requires:
      - data_samples[i].metainfo['depth_map'] exists (H_in, W_in)
      - self.loss_decode accepts depth=...
    """

    def loss_by_feat(self, seg_logits, batch_data_samples):
        # 1) GT label
        seg_label = self._stack_batch_gt(batch_data_samples)  # usually (N,1,H,W)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)  # (N,H,W)
        """
        # 2) Collect depth maps (each stored as tensor (H_in,W_in))
        depth_list = []
        for s in batch_data_samples:
            if 'depth_map' not in s.metainfo:
                raise KeyError("depth_map not found in data_sample.metainfo. "
                               "Attach it in segmentor.loss() before calling decode head loss.")
            depth_list.append(s.metainfo['depth_map'])
        """
        depth_list = []
        for s in batch_data_samples:
            if 'depth_map_raw' not in s.metainfo:
                raise KeyError("depth_map_raw not found in data_sample.metainfo. "
                            "Make sure your data_preprocessor stores it.")
            depth_list.append(s.metainfo['depth_map_raw'])
        depth = torch.stack(depth_list, dim=0).to(seg_logits.device).float()  # (N,H_in,W_in)

        # 3) Align depth to GT resolution (H,W)
        if depth.shape[-2:] != seg_label.shape[-2:]:
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=seg_label.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

        # 4) Align logits to GT resolution (BaseDecodeHead often already does this,
        #    but keep it safe in case your pipeline changes)
        if seg_logits.shape[-2:] != seg_label.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits,
                size=seg_label.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        # 5) Call your depth-aware loss (must support depth=depth)
        loss_seg = self.loss_decode(
            seg_logits,
            seg_label,
            depth=depth,
            ignore_index=self.ignore_index
        )

        # Optional: compute accuracy like MMSeg does (if you want the metric key)
        # acc_seg = accuracy(seg_logits, seg_label, ignore_index=self.ignore_index)
        # return dict(loss_seg=loss_seg, acc_seg=acc_seg)

        return dict(loss_seg=loss_seg)
