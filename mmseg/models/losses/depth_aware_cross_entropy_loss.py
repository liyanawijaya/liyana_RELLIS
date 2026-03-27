import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.losses.utils import get_class_weight
from mmengine.logging import print_log

@MODELS.register_module()
class DepthAwareCrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_ce',
                 avg_non_ignore=False,
                 depth_cfg=None,
                 log_max_depth=False,     # <-- NEW
                 log_interval=50,         # <-- NEW: print every N calls
                 logger='current'):       # <-- NEW
        super().__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        self._loss_name = loss_name

        # logging
        self.log_max_depth = log_max_depth
        self.log_interval = int(log_interval)
        self.logger = logger
        self._forward_calls = 0

        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False. If you ignore labels and want '
                'mean over non-ignored pixels (same as torch CE), set avg_non_ignore=True.')

        self.depth_cfg = depth_cfg or {}
        self.depth_cfg.setdefault('mode', 'linear')
        self.depth_cfg.setdefault('d_max', 1)
        self.depth_cfg.setdefault('alpha', 1.0)
        self.depth_cfg.setdefault('beta', 0.05)
        self.depth_cfg.setdefault('min_w', 0.1)
        self.depth_cfg.setdefault('invalid_depth_val', 0.0)
        self.depth_cfg.setdefault('ignore_invalid_depth', False)
        self.depth_cfg.setdefault('bins', [0.0, 0.2, 0.5, 0.7, 1])
        self.depth_cfg.setdefault('weights', [1.0, 0.8, 0.4, 0.2])

    # ... keep your other methods unchanged ...

    def _make_depth_weight(self, depth: torch.Tensor) -> torch.Tensor:
        cfg = self.depth_cfg
        mode = cfg['mode']
        d_max = float(cfg['d_max'])
        min_w = float(cfg['min_w'])

        invalid = depth <= float(cfg['invalid_depth_val'])

        if mode == 'linear':
            alpha = float(cfg['alpha'])

            # ---- compute dynamic depth range (ignore invalid depths) ----
            valid_mask = depth > float(cfg['invalid_depth_val'])

            if valid_mask.any():
                d_min_img = depth[valid_mask].min()
                d_max_img = depth[valid_mask].max()
            else:
                # fallback to avoid NaNs
                d_min_img = depth.min()
                d_max_img = depth.max()

            # avoid division by zero
           #denom = torch.clamp(d_max_img - d_min_img, min=1e-6)

            # ---- normalize depth to [0, 1] using image-wise min/max ----
           #depth_norm = (depth - d_min_img) / denom
           #depth_norm = (depth - d_min_img) / denom
           #depth_norm = torch.clamp(depth_norm, 0.0, 1.0)

# create tensor constants on the same device & dtype as depth
            d_min = depth.new_tensor(0.0)
            d_max = depth.new_tensor(255.0)

            # denominator (range)
            denom = torch.clamp(d_max - d_min, min=1e-6)

            # ---- normalize depth to [0, 1] ----
            depth_norm = (depth - d_min) / denom
            depth_norm = torch.clamp(depth_norm, 0.0, 1.0)


            # ---- linear decay ----
            # w = 1 - alpha * normalized_depth
            w = 1.0 - alpha * depth_norm

            # clamp to keep gradients stable
            w = torch.clamp(w, min=min_w, max=1.0)


        elif mode == 'inverse':
            beta = float(cfg['beta'])
            w = 1.0 / (1.0 + beta * torch.clamp(depth, min=0.0))
            w = torch.clamp(w, min=min_w, max=1.0)

        elif mode == 'piecewise':
            bins = cfg['bins']
            ws = cfg['weights']
            assert len(ws) == len(bins) - 1, "piecewise: len(weights) must be len(bins)-1"
            w = torch.empty_like(depth)
            w.fill_(float(ws[-1]))
            for i in range(len(ws)):
                lo, hi = float(bins[i]), float(bins[i + 1])
                mask = (depth >= lo) & (depth < hi)
                w[mask] = float(ws[i])
            w = torch.clamp(w, min=min_w, max=1.0)
        else:
            raise ValueError(f"Unknown depth_cfg.mode: {mode}")

        if cfg.get('ignore_invalid_depth', False):
            w = w.masked_fill(invalid, 0.0)
        else:
            w = w.masked_fill(invalid, 1.0)

        return w

    def forward(self,
                cls_score,
                label,
                depth=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):

        self._forward_calls += 1

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        per_pixel = F.cross_entropy(
            cls_score,
            label,
            weight=class_weight,
            ignore_index=ignore_index,
            reduction='none'
        )

        valid_mask = (label != ignore_index).to(per_pixel.dtype)

        if weight is not None:
            per_pixel = per_pixel * weight.to(per_pixel.dtype)


        if depth is not None:
            if depth.dim() == 4 and depth.size(1) == 1:
                depth = depth[:, 0]
            assert depth.shape[-2:] == label.shape[-2:], \
                f"depth spatial size {depth.shape[-2:]} must match label {label.shape[-2:]}"

            # ---- NEW: log max depth (optionally only on valid pixels) ----
            if self.log_max_depth and (self._forward_calls % self.log_interval == 0):
                       
                if depth is None:
                    raise RuntimeError("DepthAwareCrossEntropyLoss called without depth!")

                with torch.no_grad():
                    # log max over pixels that are not ignored
                    d = depth.to(per_pixel.dtype)
                    d_valid = d[label != ignore_index]
                    if d_valid.numel() > 0:
                        dmax = float(d_valid.max().item())
                        dmin = float(d_valid.min().item())
                        print_log(
                            f"[DepthAwareCE] depth(min,max) over non-ignore pixels: ({dmin:.4f}, {dmax:.4f})",
                            logger=self.logger
                        )
                    else:
                        print_log(
                            "[DepthAwareCE] depth: no valid pixels (all ignored).",
                            logger=self.logger
                        )

            depth_w = self._make_depth_weight(depth.to(per_pixel.dtype))
            per_pixel = per_pixel * depth_w

        per_pixel = per_pixel * valid_mask

        if reduction == 'none':
            return per_pixel * self.loss_weight

        if reduction == 'sum':
            return per_pixel.sum() * self.loss_weight

        if avg_factor is not None:
            denom = max(float(avg_factor), 1e-6)
        else:
            if self.avg_non_ignore:
                denom = max(valid_mask.sum().item(), 1e-6)
            else:
                denom = per_pixel.numel()

        loss = per_pixel.sum() / denom
        return loss * self.loss_weight
