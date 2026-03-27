import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.decode_heads import FCNHead

@MODELS.register_module()
class DepthSliceFCNHead(FCNHead):
    """
    FCNHead that computes loss only on a selected depth slice.

    Requires:
      - batch_data_samples[i].metainfo['depth_map_raw'] is present (H,W) or (H_in,W_in)

    Config:
      depth_slice_cfg:
        mode: "index" | "range" | "relative"
        - index: selects a single slice id (0..num_slices-1)
        - range: selects explicit [d_lo, d_hi)
        - relative: selects [r_lo, r_hi) in normalized depth [0,1] per-image using min/max

        num_slices: e.g. 512
        slice_index: e.g. 128
        include_invalid: False
        invalid_depth_val: e.g. -3
    """

    def __init__(self, *args, depth_slice_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)

        cfg = depth_slice_cfg or {}
        self.depth_slice_cfg = cfg
        self.depth_slice_cfg.setdefault("mode", "index")
        self.depth_slice_cfg.setdefault("num_slices", 512)
        self.depth_slice_cfg.setdefault("slice_index", 0)
        self.depth_slice_cfg.setdefault("d_lo", 0.0)
        self.depth_slice_cfg.setdefault("d_hi", 1.0)
        self.depth_slice_cfg.setdefault("r_lo", 0.0)
        self.depth_slice_cfg.setdefault("r_hi", 1.0)
        self.depth_slice_cfg.setdefault("include_invalid", False)
        self.depth_slice_cfg.setdefault("invalid_depth_val", None)
        self.depth_slice_cfg.setdefault("use_per_image_minmax", True)
        self.depth_slice_cfg.setdefault("eps", 1e-6)

    def _collect_depth(self, batch_data_samples, device):
        depth_list = []
        for s in batch_data_samples:
            if "depth_map_raw" not in s.metainfo:
                raise KeyError(
                    "depth_map_raw not found in data_sample.metainfo. "
                    "Make sure your data_preprocessor stores it."
                )
            depth_list.append(s.metainfo["depth_map_raw"])
        depth = torch.stack(depth_list, dim=0).to(device).float()  # (N,H,W)
        return depth

    def _valid_mask(self, depth):
        inv = self.depth_slice_cfg.get("invalid_depth_val", None)
        if inv is None:
            return torch.ones_like(depth, dtype=torch.bool)
        return depth != float(inv)

    def _make_depth_mask(self, depth: torch.Tensor) -> torch.Tensor:
        """
        depth: (N,H,W) float
        returns: mask (N,H,W) bool indicating pixels included in this aux loss.
        """
        cfg = self.depth_slice_cfg
        mode = cfg["mode"]
        eps = float(cfg["eps"])

        valid = self._valid_mask(depth)

        if mode == "index":
            # slice boundaries from per-image min/max or global min/max of valid pixels
            num_slices = int(cfg["num_slices"])
            idx = int(cfg["slice_index"])
            idx = max(0, min(idx, num_slices - 1))

            if cfg.get("use_per_image_minmax", True):
                inf = depth.new_tensor(float("inf"))
                ninf = depth.new_tensor(float("-inf"))
                d_for_min = torch.where(valid, depth, inf)
                d_for_max = torch.where(valid, depth, ninf)
                dmin = d_for_min.amin(dim=(1, 2), keepdim=True)  # (N,1,1)
                dmax = d_for_max.amax(dim=(1, 2), keepdim=True)  # (N,1,1)
            else:
                d_valid = depth[valid]
                if d_valid.numel() == 0:
                    dmin = depth.new_zeros((depth.size(0), 1, 1))
                    dmax = depth.new_ones((depth.size(0), 1, 1))
                else:
                    mn = d_valid.min()
                    mx = d_valid.max()
                    dmin = mn.view(1, 1, 1).expand(depth.size(0), 1, 1)
                    dmax = mx.view(1, 1, 1).expand(depth.size(0), 1, 1)

            dr = (dmax - dmin).clamp_min(eps)
            step = dr / float(num_slices)

            # IMPORTANT: "slice 128 contains pixels of depth 0 to mid depth"
            # That is cumulative, not a thin band.
            # cumulative upper bound:
            hi = dmin + (idx + 1) * step
            mask = (depth <= hi) & valid

        elif mode == "range":
            d_lo = float(cfg["d_lo"])
            d_hi = float(cfg["d_hi"])
            mask = (depth >= d_lo) & (depth < d_hi) & valid

        elif mode == "relative":
            # choose fraction range in [0,1] per-image
            r_lo = float(cfg["r_lo"])
            r_hi = float(cfg["r_hi"])

            inf = depth.new_tensor(float("inf"))
            ninf = depth.new_tensor(float("-inf"))
            d_for_min = torch.where(valid, depth, inf)
            d_for_max = torch.where(valid, depth, ninf)
            dmin = d_for_min.amin(dim=(1, 2), keepdim=True)
            dmax = d_for_max.amax(dim=(1, 2), keepdim=True)
            dr = (dmax - dmin).clamp_min(eps)

            lo = dmin + r_lo * dr
            hi = dmin + r_hi * dr
            mask = (depth >= lo) & (depth < hi) & valid

        else:
            raise ValueError(f"Unknown depth_slice_cfg.mode: {mode}")

        if cfg.get("include_invalid", False):
            # allow invalid pixels to participate (rarely desired)
            mask = mask | (~valid)

        return mask

    def loss_by_feat(self, seg_logits, batch_data_samples):
        # 1) GT label
        seg_label = self._stack_batch_gt(batch_data_samples)  # (N,1,H,W) or (N,H,W)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)  # (N,H,W)

        # 2) Depth
        depth = self._collect_depth(batch_data_samples, seg_logits.device)  # (N,Hd,Wd)

        # 3) Align depth to GT resolution
        if depth.shape[-2:] != seg_label.shape[-2:]:
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=seg_label.shape[-2:],
                mode="nearest"  # for masks, nearest is safer than bilinear
            ).squeeze(1)

        # 4) Align logits to GT resolution
        if seg_logits.shape[-2:] != seg_label.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits,
                size=seg_label.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        # 5) Build depth mask and apply by setting ignore_index outside region
        mask = self._make_depth_mask(depth)  # (N,H,W) bool

        ignore = torch.full_like(seg_label, self.ignore_index)
        seg_label_sliced = torch.where(mask, seg_label, ignore)

        # 6) Loss (standard CE on sliced GT)
        loss_seg = self.loss_decode(
            seg_logits,
            seg_label_sliced,
            ignore_index=self.ignore_index
        )

        return dict(loss_seg=loss_seg)
