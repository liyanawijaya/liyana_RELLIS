# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      build_activation_layer, build_norm_layer)
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from ..utils import resize
import torch.nn.functional as F



from mmcv.cnn import ConvModule


"""
@MODELS.register_module()
class SemanticBranch4Stage(nn.Module):
    '''Lightweight 4-stage semantic branch using standard convs.
    Produces 4 feature maps with channels: [16, 32, 64, 128].
    Each stage downsamples by 2 (stride=2 on first conv of stage).

    Output:
        f1: (N,16, H/2,  W/2)
        f2: (N,32, H/4,  W/4)
        f3: (N,64, H/8,  W/8)
        f4: (N,128,H/16, W/16)
    '''

    def __init__(
        self,
        in_channels=3,
        feat_channels=(16, 32, 64, 128),
        num_convs_per_stage=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        conv_cfg=None,
    ):
        super().__init__()
        assert len(feat_channels) == 4

        self.stages = nn.ModuleList()
        prev_c = in_channels

        for i, out_c in enumerate(feat_channels):
            blocks = []

            # first conv in stage does downsampling
            blocks.append(
                ConvModule(
                    prev_c, out_c, kernel_size=3, stride=2, padding=1,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
                )
            )

            # remaining convs keep resolution
            for _ in range(num_convs_per_stage - 1):
                blocks.append(
                    ConvModule(
                        out_c, out_c, kernel_size=3, stride=1, padding=1,
                        conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
                    )
                )

            self.stages.append(nn.Sequential(*blocks))
            prev_c = out_c

    def forward(self, x):
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        # outs = [f1,f2,f3,f4]
        return tuple(outs)

"""


class DepthRGBSlicer_Shared3x3(nn.Module):
    """
    Output RGB slices per depth slice.

    Output:
      rgb_slices: (N,S,3,H,W)
    """

    def __init__(self, n_slices=128, invalid_depth_val=None, eps=1e-6, use_per_image_minmax=True):
        super().__init__()
        self.S = int(n_slices)
        self.invalid_depth_val = invalid_depth_val
        self.eps = float(eps)
        self.use_per_image_minmax = bool(use_per_image_minmax)

    def _valid_mask(self, depth):
        if self.invalid_depth_val is None:
            return torch.ones_like(depth, dtype=torch.bool)
        return depth != float(self.invalid_depth_val)

    def forward(self, rgb, depth):
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)

        N, _, H, W = rgb.shape
        depth = depth.float()
        valid = self._valid_mask(depth)

        inf = depth.new_tensor(float("inf"))
        ninf = depth.new_tensor(float("-inf"))

        dmin = torch.where(valid, depth, inf).amin(dim=(2, 3), keepdim=True)
        dmax = torch.where(valid, depth, ninf).amax(dim=(2, 3), keepdim=True)

        drange = (dmax - dmin).clamp_min(self.eps)
        step = drange / float(self.S)

        i = torch.arange(1, self.S + 1, device=depth.device, dtype=depth.dtype).view(1, self.S, 1, 1)
        t = dmin + i * step  # (N,S,1,1)

        depth_hw = depth.squeeze(1)  # (N,H,W)
        valid_hw = valid.squeeze(1)  # (N,H,W)

        masks = (depth_hw.unsqueeze(1) <= t) & valid_hw.unsqueeze(1)  # (N,S,H,W)
        masks = masks.to(rgb.dtype)

        rgb_slices = rgb.unsqueeze(1) * masks.unsqueeze(2)  # (N,S,3,H,W)
        return rgb_slices




class DepthRGBSlicer128to128_1(nn.Module):
    """
    Separate 3->1 conv per slice (S slices), applied AFTER masking.
    Uses Conv3d with groups=S so each slice has its own weights (no mixing across slices).
    Output: (N,S,H,W)
    """

    def __init__(self, n_slices=128, invalid_depth_val=None, eps=1e-6, use_per_image_minmax=True):
        super().__init__()
        self.S = int(n_slices)
        self.invalid_depth_val = invalid_depth_val
        self.eps = float(eps)
        self.use_per_image_minmax = bool(use_per_image_minmax)

        # One conv per slice, but implemented efficiently:
        # Input channels = 3*S, Output channels = 1*S, groups = S
        # So each group maps 3 -> 1 independently.
        self.rgb_to_1_per_slice = nn.Conv3d(
            in_channels=3 * self.S,
            out_channels=1 * self.S,
            kernel_size=(1, 1, 1),
            groups=self.S,
            bias=False
        )

    def _valid_mask(self, depth: torch.Tensor) -> torch.Tensor:
        if self.invalid_depth_val is None:
            return torch.ones_like(depth, dtype=torch.bool)
        return depth != float(self.invalid_depth_val)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        assert rgb.dim() == 4 and rgb.size(1) == 3, f"rgb must be (N,3,H,W), got {rgb.shape}"
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        assert depth.dim() == 4 and depth.size(1) == 1, f"depth must be (N,1,H,W), got {depth.shape}"
        assert depth.shape[0] == rgb.shape[0] and depth.shape[-2:] == rgb.shape[-2:], "rgb/depth size mismatch"

        N, _, H, W = rgb.shape
        depth = depth.float()
        valid = self._valid_mask(depth)  # (N,1,H,W)

        # ---- dmin/dmax ----
        if self.use_per_image_minmax:
            inf = depth.new_tensor(float("inf"))
            ninf = depth.new_tensor(float("-inf"))
            d_for_min = torch.where(valid, depth, inf)
            d_for_max = torch.where(valid, depth, ninf)
            dmin = d_for_min.amin(dim=(2, 3), keepdim=True)  # (N,1,1,1)
            dmax = d_for_max.amax(dim=(2, 3), keepdim=True)  # (N,1,1,1)
        else:
            d_valid = depth[valid]
            if d_valid.numel() == 0:
                dmin = depth.new_zeros((N, 1, 1, 1))
                dmax = depth.new_ones((N, 1, 1, 1))
            else:
                mn = d_valid.min()
                mx = d_valid.max()
                dmin = mn.view(1, 1, 1, 1).expand(N, 1, 1, 1)
                dmax = mx.view(1, 1, 1, 1).expand(N, 1, 1, 1)

        drange = (dmax - dmin).clamp_min(self.eps)
        step = drange / float(self.S)

        i = torch.arange(1, self.S + 1, device=depth.device, dtype=depth.dtype).view(1, self.S, 1, 1)
        t = dmin + i * step  # (N,S,1,1)

        depth_hw = depth.squeeze(1)  # (N,H,W)
        valid_hw = valid.squeeze(1)  # (N,H,W)

        masks = (depth_hw.unsqueeze(1) <= t) & valid_hw.unsqueeze(1)  # (N,S,H,W)
        masks = masks.to(rgb.dtype)

        # ---- Mask RGB first: (N,S,3,H,W) ----
        masked_rgb = rgb.unsqueeze(1).expand(N, self.S, 3, H, W) * masks.unsqueeze(2)

        # Arrange for grouped Conv3d:
        # We want channels = 3*S, with a singleton "slice depth" dimension D=1
        # masked_rgb: (N,S,3,H,W) -> (N,3*S,1,H,W)
        x = masked_rgb.permute(0, 1, 2, 3, 4).reshape(N, self.S * 3, 1, H, W)

        y = self.rgb_to_1_per_slice(x)          # (N,S,1,H,W)
        out = y.squeeze(2)                       # (N,S,H,W)
        return out



class DepthRGBSlicer128to128(nn.Module):
    """
    Build 128 cumulative RGB slices using depth thresholds.

    Input:
      rgb   : (N,3,H,W)
      depth : (N,1,H,W) or (N,H,W)

    Output:
      out128 : (N,128,H,W)

    Notes:
      - No slice_to_16 conv.
      - No pooling/avg/sum/max. Returns all 128 channels sequentially.
    """

    def __init__(
        self,
        n_slices: int = 128,
        invalid_depth_val: float = None,   # e.g. -3
        eps: float = 1e-6,
        use_per_image_minmax: bool = True,
    ):
        super().__init__()
        self.S = int(n_slices)
        self.invalid_depth_val = invalid_depth_val
        self.eps = float(eps)
        self.use_per_image_minmax = bool(use_per_image_minmax)

        # RGB -> 1 feature per slice (learned linear mixing of RGB)
        self.rgb_to_slice = nn.Conv2d(3, self.S, kernel_size=1, bias=False)

        # NOTE: removed slice_to_16

    def _valid_mask(self, depth):
        if self.invalid_depth_val is None:
            return torch.ones_like(depth, dtype=torch.bool)
        return depth != float(self.invalid_depth_val)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        assert rgb.dim() == 4 and rgb.size(1) == 3, f"rgb must be (N,3,H,W), got {rgb.shape}"
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        assert depth.dim() == 4 and depth.size(1) == 1, f"depth must be (N,1,H,W), got {depth.shape}"
        assert depth.shape[0] == rgb.shape[0] and depth.shape[-2:] == rgb.shape[-2:], "rgb/depth size mismatch"

        N, _, H, W = rgb.shape
        depth = depth.float()

        valid = self._valid_mask(depth)  # (N,1,H,W)

        # compute dmin/dmax
        if self.use_per_image_minmax:
            inf = depth.new_tensor(float("inf"))
            ninf = depth.new_tensor(float("-inf"))

            d_for_min = torch.where(valid, depth, inf)
            d_for_max = torch.where(valid, depth, ninf)

            dmin = d_for_min.amin(dim=(2, 3), keepdim=True)  # (N,1,1,1)
            dmax = d_for_max.amax(dim=(2, 3), keepdim=True)  # (N,1,1,1)
        else:
            d_valid = depth[valid]
            if d_valid.numel() == 0:
                dmin = depth.new_zeros((N,1,1,1))
                dmax = depth.new_ones((N,1,1,1))
            else:
                mn = d_valid.min()
                mx = d_valid.max()
                dmin = mn.view(1,1,1,1).expand(N,1,1,1)
                dmax = mx.view(1,1,1,1).expand(N,1,1,1)

        drange = (dmax - dmin).clamp_min(self.eps)
        step = drange / float(self.S)

        # thresholds t_i: dmin + i*step, i=1..S
        i = torch.arange(1, self.S + 1, device=depth.device, dtype=depth.dtype).view(1, self.S, 1, 1)
        t = dmin + i * step  # (N,S,1,1)

        depth_hw = depth.squeeze(1)  # (N,H,W)
        valid_hw = valid.squeeze(1)  # (N,H,W)

        masks = (depth_hw.unsqueeze(1) <= t) & valid_hw.unsqueeze(1)  # (N,S,H,W)
        masks = masks.to(rgb.dtype)

        # (N,S,H,W)
        slice_feat = self.rgb_to_slice(rgb)

        # Apply masks and return ALL 128 channels
        out128 = slice_feat * masks  # (N,128,H,W)
        return out128



class DepthRGBSlicer128to16(nn.Module):
    """
    Build 128 cumulative RGB slices using depth thresholds, then project to 16 channels.

    Input:
      rgb   : (N,3,H,W)
      depth : (N,1,H,W) or (N,H,W)

    Output:
      out16 : (N,16,H,W)

    Notes:
      - "last slice contains whole image": achieved by cumulative masks.
      - Handles constant-depth safely.
      - Optionally ignores invalid depth (e.g., pad value -3).
    """

    def __init__(
        self,
        n_slices: int = 128,
        out_channels: int = 16,
        invalid_depth_val: float = None,   # e.g. -3
        eps: float = 1e-6,
        use_per_image_minmax: bool = True,  # per-image dmin/dmax vs per-batch
    ):
        super().__init__()
        self.S = int(n_slices)
        self.outC = int(out_channels)
        self.invalid_depth_val = invalid_depth_val
        self.eps = float(eps)
        self.use_per_image_minmax = bool(use_per_image_minmax)

        # 1) RGB -> 1 feature per slice (learned linear mixing of RGB)
        # weights: (S,3,1,1)
        self.rgb_to_slice = nn.Conv2d(3, self.S, kernel_size=1, bias=False)

        # 2) 128 slices -> 16 channels
        self.slice_to_16 = nn.Conv2d(self.S, self.outC, kernel_size=1, bias=True)

    def _valid_mask(self, depth):
        if self.invalid_depth_val is None:
            return torch.ones_like(depth, dtype=torch.bool)
        return depth != float(self.invalid_depth_val)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        assert rgb.dim() == 4 and rgb.size(1) == 3, f"rgb must be (N,3,H,W), got {rgb.shape}"
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        assert depth.dim() == 4 and depth.size(1) == 1, f"depth must be (N,1,H,W), got {depth.shape}"
        assert depth.shape[0] == rgb.shape[0] and depth.shape[-2:] == rgb.shape[-2:], "rgb/depth size mismatch"

        N, _, H, W = rgb.shape
        depth = depth.float()

        valid = self._valid_mask(depth)  # (N,1,H,W)

        # compute dmin/dmax
        if self.use_per_image_minmax:
            # per-image min/max over valid pixels
            inf = torch.tensor(float("inf"), device=depth.device, dtype=depth.dtype)
            ninf = torch.tensor(float("-inf"), device=depth.device, dtype=depth.dtype)

            d_for_min = torch.where(valid, depth, inf)
            d_for_max = torch.where(valid, depth, ninf)

            dmin = d_for_min.amin(dim=(2, 3), keepdim=True)  # (N,1,1,1)
            dmax = d_for_max.amax(dim=(2, 3), keepdim=True)  # (N,1,1,1)
        else:
            # per-batch min/max
            d_valid = depth[valid]
            if d_valid.numel() == 0:
                dmin = depth.new_zeros((N,1,1,1))
                dmax = depth.new_ones((N,1,1,1))
            else:
                mn = d_valid.min()
                mx = d_valid.max()
                dmin = mn.view(1,1,1,1).expand(N,1,1,1)
                dmax = mx.view(1,1,1,1).expand(N,1,1,1)

        drange = (dmax - dmin).clamp_min(self.eps)  # avoid divide-by-zero if constant depth
        step = drange / float(self.S)               # (N,1,1,1)

        # thresholds t_i: dmin + i*step, i=1..S
        i = torch.arange(1, self.S + 1, device=depth.device, dtype=depth.dtype).view(1, self.S, 1, 1)
        t = dmin + i * step  # (N, S, 1, 1) via broadcasting

        # cumulative masks: depth <= t_i (and valid)
        # depth: (N,1,H,W) -> compare with (N,S,1,1) => (N,S,H,W)
        depth_hw = depth.squeeze(1)  # (N,H,W)
        valid_hw = valid.squeeze(1)  # (N,H,W)

        masks = (depth_hw.unsqueeze(1) <= t) & valid_hw.unsqueeze(1)  # (N,S,H,W)
        masks = masks.to(rgb.dtype)  # float mask

        # project RGB to per-slice feature maps (before masking)
        # slice_feat: (N,S,H,W)
        slice_feat = self.rgb_to_slice(rgb)

        # apply masks: (N,S,H,W)
        slice_feat = slice_feat * masks

        # compress 128 -> 16
        out16 = self.slice_to_16(slice_feat)  # (N,16,H,W)
        return out16




class DepthGuidedConvStage(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 downsample_first=True,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 conv_cfg=None,
                 # DG params
                 use_dg=True,
                 dg_kernel_size=3,
                 dg_sigma=0.1,
                 dg_learnable_spatial=True,
                 dg_shared_spatial=False,
                 dg_residual=True,
                 dg_res_scale=0.1,
                 depth_pad_val=None,

                 # ---- NEW: depth normalization config for DG layer ----
                 dg_normalize_depth=True,
                 dg_norm_mode="per_image",   # "per_image" | "per_batch" | "fixed" | "none"
                 dg_fixed_min=-2.17,
                 dg_fixed_max=2.21,
                 dg_clamp_norm=True):
        super().__init__()
        self.use_dg = bool(use_dg)
        self.dg_residual = bool(dg_residual)
        self.dg_res_scale = float(dg_res_scale)

        self.convs = nn.ModuleList()
        self.dgs = nn.ModuleList() if self.use_dg else None

        for i in range(num_convs):
            stride = 2 if (i == 0 and downsample_first) else 1
            in_c = in_channels if i == 0 else out_channels

            self.convs.append(
                ConvModule(
                    in_c, out_channels,
                    kernel_size=3, stride=stride, padding=1,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
                )
            )

            if self.use_dg:
                self.dgs.append(
                    DepthSimilarityWeightedConv(
                        channels=out_channels,
                        kernel_size=dg_kernel_size,
                        sigma=dg_sigma,
                        learnable_spatial=dg_learnable_spatial,
                        shared_spatial=dg_shared_spatial,
                        depth_pad_val=depth_pad_val,

                        # ---- NEW: pass depth normalisation settings ----
                        normalize_depth=dg_normalize_depth,
                        norm_mode=dg_norm_mode,
                        fixed_min=dg_fixed_min,
                        fixed_max=dg_fixed_max,
                        clamp_norm=dg_clamp_norm,
                    )
                )

    @staticmethod
    def _resize_depth(depth: torch.Tensor, size_hw):
        """depth: (N,1,H,W) or (N,H,W) -> (N,1,h,w)"""
        if depth is None:
            return None
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        return F.interpolate(depth.float(), size=size_hw, mode='nearest')

    def forward(self, x, depth=None):
        # No DG path
        if (not self.use_dg) or (depth is None):
            for conv in self.convs:
                x = conv(x)
            return x

        # DG path (DG after each conv)
        for conv, dg in zip(self.convs, self.dgs):
            x = conv(x)
            d = self._resize_depth(depth, x.shape[-2:])
            if self.dg_residual:
                x = x + self.dg_res_scale * dg(x, d)
            else:
                x = dg(x, d)
        return x



class SemanticBranch4StageNormal(nn.Module):
    """
    Pure CNN semantic branch (NO depth guidance).

    Channels:
        8 → 16 → 32 → 64

    Each stage:
        - First conv in the stage uses stride=2 (downsample)
        - Conv-BN-Act repeated num_convs_per_stage times

    Forward outputs:
        - If return_intermediate=True: returns (f1, f2, f3, f4)
        - Else: returns f4 only
    """

    def __init__(self,
                 in_channels=3,
                 feat_channels=(8, 16, 32, 64),
                 num_convs_per_stage=2,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 conv_cfg=None):
        super().__init__()

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs_per_stage = num_convs_per_stage

        c1, c2, c3, c4 = feat_channels

        self.stage1 = self._make_stage(
            in_channels, c1, num_convs_per_stage,
            norm_cfg, act_cfg, conv_cfg)

        self.stage2 = self._make_stage(
            c1, c2, num_convs_per_stage,
            norm_cfg, act_cfg, conv_cfg)

        self.stage3 = self._make_stage(
            c2, c3, num_convs_per_stage,
            norm_cfg, act_cfg, conv_cfg)

        self.stage4 = self._make_stage(
            c3, c4, num_convs_per_stage,
            norm_cfg, act_cfg, conv_cfg)

    # -------------------------------------------------------
    # build one stage
    # -------------------------------------------------------
    def _make_stage(self,
                    in_channels,
                    out_channels,
                    num_convs,
                    norm_cfg,
                    act_cfg,
                    conv_cfg):

        layers = []
        for i in range(num_convs):
            stride = 2 if i == 0 else 1  # downsample first conv
            in_c = in_channels if i == 0 else out_channels

            layers.append(
                ConvModule(
                    in_c,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_intermediate: bool = True):
        """
        Args:
            x: (N, in_channels, H, W)
            return_intermediate: if True returns (f1,f2,f3,f4), else returns f4

        Returns:
            f1: (N, c1, H/2,  W/2)
            f2: (N, c2, H/4,  W/4)
            f3: (N, c3, H/8,  W/8)
            f4: (N, c4, H/16, W/16)
            (exact scales depend on your input size and stride setup)
        """
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        if return_intermediate:
            return f1, f2, f3, f4
        return f4



@MODELS.register_module()
class SemanticBranch4StageDGOnly34(nn.Module):
    def __init__(self,
                 in_channels=3,
                 feat_channels=(16, 32, 64, 128),
                 num_convs_per_stage=2,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 conv_cfg=None,
                 # DG params
                 dg_kernel_size=3,
                 dg_sigma=0.1,
                 dg_learnable_spatial=True,
                 dg_shared_spatial=False,
                 dg_residual=True,
                 dg_res_scale=0.1,
                 depth_pad_val=-3,

                 # ---- NEW: depth normalization controls ----
                 dg_normalize_depth=True,
                 dg_norm_mode="per_image",
                 dg_fixed_min=-2.17,
                 dg_fixed_max=2.21,
                 dg_clamp_norm=True):
        super().__init__()
        c1, c2, c3, c4 = feat_channels

        # stage 1 & 2: NO DG
        self.stage1 = DepthGuidedConvStage(
            in_channels=in_channels, out_channels=c1,
            num_convs=num_convs_per_stage, downsample_first=True,
            norm_cfg=norm_cfg, act_cfg=act_cfg, conv_cfg=conv_cfg,
            use_dg=False
        )
        self.stage2 = DepthGuidedConvStage(
            in_channels=c1, out_channels=c2,
            num_convs=num_convs_per_stage, downsample_first=True,
            norm_cfg=norm_cfg, act_cfg=act_cfg, conv_cfg=conv_cfg,
            use_dg=False
        )

        # stage 3 & 4: DG after EACH conv
        self.stage3 = DepthGuidedConvStage(
            in_channels=c2, out_channels=c3,
            num_convs=num_convs_per_stage, downsample_first=True,
            norm_cfg=norm_cfg, act_cfg=act_cfg, conv_cfg=conv_cfg,
            use_dg=True,
            dg_kernel_size=dg_kernel_size, dg_sigma=dg_sigma,
            dg_learnable_spatial=dg_learnable_spatial, dg_shared_spatial=dg_shared_spatial,
            dg_residual=dg_residual, dg_res_scale=dg_res_scale,
            depth_pad_val=depth_pad_val,

            # ---- NEW pass-through ----
            dg_normalize_depth=dg_normalize_depth,
            dg_norm_mode=dg_norm_mode,
            dg_fixed_min=dg_fixed_min,
            dg_fixed_max=dg_fixed_max,
            dg_clamp_norm=dg_clamp_norm
        )

        self.stage4 = DepthGuidedConvStage(
            in_channels=c3, out_channels=c4,
            num_convs=num_convs_per_stage, downsample_first=True,
            norm_cfg=norm_cfg, act_cfg=act_cfg, conv_cfg=conv_cfg,
            use_dg=True,
            dg_kernel_size=dg_kernel_size, dg_sigma=dg_sigma,
            dg_learnable_spatial=dg_learnable_spatial, dg_shared_spatial=dg_shared_spatial,
            dg_residual=dg_residual, dg_res_scale=dg_res_scale,
            depth_pad_val=depth_pad_val,

            # ---- NEW pass-through ----
            dg_normalize_depth=dg_normalize_depth,
            dg_norm_mode=dg_norm_mode,
            dg_fixed_min=dg_fixed_min,
            dg_fixed_max=dg_fixed_max,
            dg_clamp_norm=dg_clamp_norm
        )

    def forward(self, x, depth=None):
        f1 = self.stage1(x, depth=None)
        f2 = self.stage2(f1, depth=None)
        f3 = self.stage3(f2, depth=depth)
        f4 = self.stage4(f3, depth=depth)
        #return (f1, f2, f3, f4)
        return (f4)

class DetailBranch(BaseModule):
    """Detail Branch with wide channels and shallow layers to capture low-level
    details and generate high-resolution feature representation.

    Args:
        detail_channels (Tuple[int]): Size of channel numbers of each stage
            in Detail Branch, in paper it has 3 stages.
            Default: (64, 64, 128).
        in_channels (int): Number of channels of input image. Default: 3.
        conv_cfg (dict | None): Config of conv layers.
            Default: None.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Returns:
        x (torch.Tensor): Feature map of Detail Branch.
    """

    def __init__(self,
                 detail_channels=(64, 64, 128),
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        detail_branch = []
        for i in range(len(detail_channels)):
            if i == 0:
                detail_branch.append(
                    nn.Sequential(
                        ConvModule(
                            in_channels=in_channels,
                            out_channels=detail_channels[i],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg),
                        ConvModule(
                            in_channels=detail_channels[i],
                            out_channels=detail_channels[i],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg)))
            else:
                detail_branch.append(
                    nn.Sequential(
                        ConvModule(
                            in_channels=detail_channels[i - 1],
                            out_channels=detail_channels[i],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg),
                        ConvModule(
                            in_channels=detail_channels[i],
                            out_channels=detail_channels[i],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg),
                        ConvModule(
                            in_channels=detail_channels[i],
                            out_channels=detail_channels[i],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg)))
        self.detail_branch = nn.ModuleList(detail_branch)

    def forward(self, x):
        for stage in self.detail_branch:
            x = stage(x)
        return x


class StemBlock(BaseModule):
    """Stem Block at the beginning of Semantic Branch.

    Args:
        in_channels (int): Number of input channels.
            Default: 3.
        out_channels (int): Number of output channels.
            Default: 16.
        conv_cfg (dict | None): Config of conv layers.
            Default: None.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Returns:
        x (torch.Tensor): First feature map in Semantic Branch.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=16,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.conv_first = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.convs = nn.Sequential(
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels // 2,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.pool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse_last = ConvModule(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv_first(x)
        x_left = self.convs(x)
        x_right = self.pool(x)
        x = self.fuse_last(torch.cat([x_left, x_right], dim=1))
        return x


class GELayer(BaseModule):
    """Gather-and-Expansion Layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        exp_ratio (int): Expansion ratio for middle channels.
            Default: 6.
        stride (int): Stride of GELayer. Default: 1
        conv_cfg (dict | None): Config of conv layers.
            Default: None.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Returns:
        x (torch.Tensor): Intermediate feature map in
            Semantic Branch.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 exp_ratio=6,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        mid_channel = in_channels * exp_ratio
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if stride == 1:
            self.dwconv = nn.Sequential(
                # ReLU in ConvModule not shown in paper
                ConvModule(
                    in_channels=in_channels,
                    out_channels=mid_channel,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=in_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.shortcut = None
        else:
            self.dwconv = nn.Sequential(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=mid_channel,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                # ReLU in ConvModule not shown in paper
                ConvModule(
                    in_channels=mid_channel,
                    out_channels=mid_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=mid_channel,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
            )
            self.shortcut = nn.Sequential(
                DepthwiseSeparableConvModule(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    dw_norm_cfg=norm_cfg,
                    dw_act_cfg=None,
                    pw_norm_cfg=norm_cfg,
                    pw_act_cfg=None,
                ))

        self.conv2 = nn.Sequential(
            ConvModule(
                in_channels=mid_channel,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
            ))

        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv2(x)
        if self.shortcut is not None:
            shortcut = self.shortcut(identity)
            x = x + shortcut
        else:
            x = x + identity
        x = self.act(x)
        return x


class CEBlock(BaseModule):
    """Context Embedding Block for large receptive filed in Semantic Branch.

    Args:
        in_channels (int): Number of input channels.
            Default: 3.
        out_channels (int): Number of output channels.
            Default: 16.
        conv_cfg (dict | None): Config of conv layers.
            Default: None.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Returns:
        x (torch.Tensor): Last feature map in Semantic Branch.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=16,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            build_norm_layer(norm_cfg, self.in_channels)[1])
        self.conv_gap = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # Note: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvModule(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        identity = x
        x = self.gap(x)
        x = self.conv_gap(x)
        x = identity + x
        x = self.conv_last(x)
        return x





class DepthSimilarityWeightedConv_1(nn.Module):
    """
    Depth-similarity guided local aggregation (bilateral-style) with OPTIONAL
    per-channel learnable spatial kernels.

    Adds OPTIONAL depth normalization inside the layer so sigma works
    even if depth values are in ranges like [-2.17, 2.21].

    Normalization options:
      - None: use raw depth
      - per_image: normalize each sample to [0,1] using its own min/max
      - per_batch: normalize using batch min/max
      - fixed: normalize using provided fixed_min/fixed_max
    """

    def __init__(
        self,
        channels: int = 128,
        kernel_size: int = 3,
        sigma: float = 0.1,
        learnable_spatial: bool = True,
        shared_spatial: bool = False,
        eps: float = 1e-6,
        depth_pad_val: float = None,

        # --- NEW ---
        normalize_depth: bool = True,
        norm_mode: str = "per_image",      # "per_image" | "per_batch" | "fixed" | "none"
        fixed_min: float = -2.17,          # used only if norm_mode="fixed"
        fixed_max: float = 2.21,           # used only if norm_mode="fixed"
        clamp_norm: bool = True,           # clamp normalized depth to [0,1]
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        assert norm_mode in ("per_image", "per_batch", "fixed", "none")

        self.C = int(channels)
        self.ks = int(kernel_size)
        self.pad = self.ks // 2

        self.sigma = float(sigma)
        self.eps = float(eps)

        self.learnable_spatial = bool(learnable_spatial)
        self.shared_spatial = bool(shared_spatial)
        self.depth_pad_val = depth_pad_val

        # --- NEW ---
        self.normalize_depth = bool(normalize_depth)
        self.norm_mode = norm_mode
        self.clamp_norm = bool(clamp_norm)
        self.fixed_min = float(fixed_min)
        self.fixed_max = float(fixed_max)

        P = self.ks * self.ks

        if self.learnable_spatial:
            if self.shared_spatial:
                self.spatial = nn.Parameter(torch.ones(P))
            else:
                self.spatial = nn.Parameter(torch.ones(self.C, P))
        else:
            self.spatial = None

        if self.spatial is not None:
            with torch.no_grad():
                if self.shared_spatial:
                    self.spatial.fill_(1.0)
                    self.spatial[P // 2] = 1.25
                else:
                    self.spatial.fill_(1.0)
                    self.spatial[:, P // 2] = 1.25

    def _get_ws(self, wd: torch.Tensor):
        """Return spatial weights broadcastable to (N,C,P,HW), or None."""
        if self.spatial is None:
            return None
        P = self.ks * self.ks
        if self.shared_spatial:
            return self.spatial.view(1, 1, P, 1).to(device=wd.device, dtype=wd.dtype)
        return self.spatial.view(1, self.C, P, 1).to(device=wd.device, dtype=wd.dtype)

    def _normalize_depth(self, depth_f: torch.Tensor) -> torch.Tensor:
        """
        depth_f: (N,1,H,W) float
        returns: normalized depth (N,1,H,W) float
        """
        if (not self.normalize_depth) or (self.norm_mode == "none"):
            return depth_f

        # invalid mask (optional)
        if self.depth_pad_val is not None:
            valid = depth_f != float(self.depth_pad_val)
        else:
            valid = torch.ones_like(depth_f, dtype=torch.bool)

        # If no valid pixels, just return as-is (avoid NaNs)
        if not valid.any():
            return depth_f

        # Compute min/max as tensors on same device/dtype
        if self.norm_mode == "fixed":
            dmin = depth_f.new_tensor(self.fixed_min)
            dmax = depth_f.new_tensor(self.fixed_max)

        elif self.norm_mode == "per_batch":
            d_valid = depth_f[valid]
            dmin = d_valid.min()
            dmax = d_valid.max()

        else:  # "per_image"
            # per sample min/max, shape (N,1,1,1)
            # Replace invalid with +inf/-inf so they don't affect min/max
            inf = torch.tensor(float("inf"), device=depth_f.device, dtype=depth_f.dtype)
            ninf = torch.tensor(float("-inf"), device=depth_f.device, dtype=depth_f.dtype)

            d_for_min = torch.where(valid, depth_f, inf)
            d_for_max = torch.where(valid, depth_f, ninf)

            dmin = d_for_min.amin(dim=(2, 3), keepdim=True)
            dmax = d_for_max.amax(dim=(2, 3), keepdim=True)

        denom = (dmax - dmin).clamp_min(depth_f.new_tensor(self.eps))
        depth_n = (depth_f - dmin) / denom

        if self.clamp_norm:
            depth_n = depth_n.clamp(0.0, 1.0)

        # keep invalid values unchanged (or you can set them to 0)
        if self.depth_pad_val is not None:
            depth_n = torch.where(valid, depth_n, depth_f)

        return depth_n

    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        x:     (N,C,H,W)
        depth: (N,1,H,W) or (N,H,W) aligned to x spatially
        """
        assert x.dim() == 4, f"x must be (N,C,H,W), got {x.shape}"
        N, C, H, W = x.shape
        assert C == self.C, f"Expected C={self.C}, got {C}"

        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        assert depth.dim() == 4 and depth.size(1) == 1, f"depth must be (N,1,H,W), got {depth.shape}"
        assert depth.shape[0] == N
        assert depth.shape[-2:] == (H, W), f"Depth must match x size. depth={depth.shape[-2:]} vs x={(H,W)}"

        P = self.ks * self.ks
        HW = H * W

        # --- Unfold x to patches: (N, C*P, HW) -> (N,C,P,HW)
        x_patch = F.unfold(x, kernel_size=self.ks, padding=self.pad).view(N, C, P, HW)

        # --- Depth float + optional normalization (still (N,1,H,W))
        depth_f = depth.float()
        depth_f = self._normalize_depth(depth_f)

        # --- Unfold depth to patches: (N, P, HW) and center (N,1,HW)
        d_patch = F.unfold(depth_f, kernel_size=self.ks, padding=self.pad)  # (N,P,HW)
        d_center = depth_f.view(N, 1, HW)                                   # (N,1,HW)

        # --- Depth similarity weights wd: (N,P,HW)
        diff = d_patch - d_center
        sigma2 = depth_f.new_tensor(self.sigma * self.sigma)
        wd = torch.exp(-(diff * diff) / (2.0 * sigma2 + depth_f.new_tensor(self.eps)))

        # Optional: ignore invalid/padded depth neighbors
        if self.depth_pad_val is not None:
            padv = float(self.depth_pad_val)
            invalid_n = (d_patch == padv)
            invalid_c = (d_center == padv)
            wd = wd.masked_fill(invalid_n | invalid_c, 0.0)

        # --- Expand to channels: (N,1,P,HW)
        wC = wd.unsqueeze(1)

        # --- Apply spatial weights
        ws = self._get_ws(wd)
        if ws is not None:
            wC = wC * ws

        # --- Weighted sum & normalize
        num = (x_patch * wC).sum(dim=2)                  # (N,C,HW)
        den = wC.sum(dim=2).clamp_min(depth_f.new_tensor(self.eps))  # (N,C,HW)
        out = (num / den).view(N, C, H, W)

        return out





class DepthSimilarityWeightedConv(nn.Module):
    """
    Depth-similarity guided local aggregation with OPTIONAL learnable spatial weights.

    This version uses a LINEAR depth weight with LOCAL (per-kernel) min/max:

        dmin_local, dmax_local = min/max of depth in the kxk window (per output pixel)
        d_num = max(|dmax_local - d_center|, |d_center - dmin_local|)
        wd = 1 - |d_patch - d_center| / (d_num + eps)
        wd = clamp(wd, 0, 1)

    Depth normalization (optional) still supported.
    """

    def __init__(
        self,
        channels: int = 128,
        kernel_size: int = 3,
        sigma: float = 0.1,                 # kept for API compatibility (unused by linear wd)
        learnable_spatial: bool = True,
        shared_spatial: bool = True,
        eps: float = 1e-6,
        depth_pad_val: float = None,

        # depth normalization
        normalize_depth: bool = True,
        norm_mode: str = "per_image",       # "per_image" | "per_batch" | "fixed" | "none"
        fixed_min: float = -2.17,
        fixed_max: float = 2.21,
        clamp_norm: bool = True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        assert norm_mode in ("per_image", "per_batch", "fixed", "none")

        self.C = int(channels)
        self.ks = int(kernel_size)
        self.pad = self.ks // 2

        self.sigma = float(sigma)  # unused here
        self.eps = float(eps)

        self.learnable_spatial = bool(learnable_spatial)
        self.shared_spatial = bool(shared_spatial)
        self.depth_pad_val = depth_pad_val

        self.normalize_depth = bool(normalize_depth)
        self.norm_mode = norm_mode
        self.clamp_norm = bool(clamp_norm)
        self.fixed_min = float(fixed_min)
        self.fixed_max = float(fixed_max)

        P = self.ks * self.ks

        if self.learnable_spatial:
            if self.shared_spatial:
                self.spatial = nn.Parameter(torch.ones(P))
            else:
                self.spatial = nn.Parameter(torch.ones(self.C, P))
        else:
            self.spatial = None

        if self.spatial is not None:
            with torch.no_grad():
                if self.shared_spatial:
                    self.spatial.fill_(1.0)
                    self.spatial[P // 2] = 1.25
                else:
                    self.spatial.fill_(1.0)
                    self.spatial[:, P // 2] = 1.25

    def _get_ws(self, wd: torch.Tensor):
        """Return spatial weights broadcastable to (N,C,P,HW), or None."""
        if self.spatial is None:
            return None
        P = self.ks * self.ks
        if self.shared_spatial:
            return self.spatial.view(1, 1, P, 1).to(device=wd.device, dtype=wd.dtype)
        return self.spatial.view(1, self.C, P, 1).to(device=wd.device, dtype=wd.dtype)

    def _normalize_depth(self, depth_f: torch.Tensor) -> torch.Tensor:
        """
        depth_f: (N,1,H,W) float
        returns: normalized depth (N,1,H,W) float
        """
        if (not self.normalize_depth) or (self.norm_mode == "none"):
            return depth_f

        # valid mask (optional)
        if self.depth_pad_val is not None:
            valid = depth_f != float(self.depth_pad_val)
        else:
            valid = torch.ones_like(depth_f, dtype=torch.bool)

        if not valid.any():
            return depth_f

        if self.norm_mode == "fixed":
            dmin = depth_f.new_tensor(self.fixed_min)
            dmax = depth_f.new_tensor(self.fixed_max)

        elif self.norm_mode == "per_batch":
            d_valid = depth_f[valid]
            dmin = d_valid.min()
            dmax = d_valid.max()

        else:  # "per_image"
            inf = depth_f.new_tensor(float("inf"))
            ninf = depth_f.new_tensor(float("-inf"))
            d_for_min = torch.where(valid, depth_f, inf)
            d_for_max = torch.where(valid, depth_f, ninf)
            dmin = d_for_min.amin(dim=(2, 3), keepdim=True)  # (N,1,1,1)
            dmax = d_for_max.amax(dim=(2, 3), keepdim=True)  # (N,1,1,1)

        denom = (dmax - dmin).clamp_min(depth_f.new_tensor(self.eps))
        depth_n = (depth_f - dmin) / denom

        if self.clamp_norm:
            depth_n = depth_n.clamp(0.0, 1.0)

        if self.depth_pad_val is not None:
            depth_n = torch.where(valid, depth_n, depth_f)

        return depth_n

    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        x:     (N,C,H,W)
        depth: (N,1,H,W) or (N,H,W) aligned to x spatially
        """
        assert x.dim() == 4, f"x must be (N,C,H,W), got {x.shape}"
        N, C, H, W = x.shape
        assert C == self.C, f"Expected C={self.C}, got {C}"

        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        assert depth.dim() == 4 and depth.size(1) == 1, f"depth must be (N,1,H,W), got {depth.shape}"
        assert depth.shape[0] == N
        assert depth.shape[-2:] == (H, W), f"Depth must match x size. depth={depth.shape[-2:]} vs x={(H,W)}"

        P = self.ks * self.ks
        HW = H * W
        eps_t = x.new_tensor(self.eps)

        # --- Unfold x to patches: (N, C*P, HW) -> (N,C,P,HW)
        x_patch = F.unfold(x, kernel_size=self.ks, padding=self.pad).view(N, C, P, HW)

        # --- Depth float + optional normalization (still (N,1,H,W))
        depth_f = self._normalize_depth(depth.float())

        # --- Unfold depth to patches: d_patch (N,P,HW), center (N,1,HW)
        d_patch = F.unfold(depth_f, kernel_size=self.ks, padding=self.pad)  # (N,P,HW)
        d_center = depth_f.view(N, 1, HW)                                   # (N,1,HW)

        # --- Build LOCAL (per-kernel) dmin/dmax from d_patch
        if self.depth_pad_val is not None:
            padv = float(self.depth_pad_val)

            # valid neighbors mask in the patch
            valid_n = (d_patch != padv)  # (N,P,HW)
            # valid center mask
            valid_c = (d_center != padv) # (N,1,HW)

            inf = d_patch.new_tensor(float("inf"))
            ninf = d_patch.new_tensor(float("-inf"))

            d_for_min = torch.where(valid_n, d_patch, inf)
            d_for_max = torch.where(valid_n, d_patch, ninf)

            dmin_local = d_for_min.amin(dim=1, keepdim=True)  # (N,1,HW)
            dmax_local = d_for_max.amax(dim=1, keepdim=True)  # (N,1,HW)

            # if ALL neighbors invalid for a location, amin/amax gives inf/-inf; fix that:
            all_invalid = ~valid_n.any(dim=1, keepdim=True)   # (N,1,HW)
            dmin_local = torch.where(all_invalid, d_center, dmin_local)
            dmax_local = torch.where(all_invalid, d_center, dmax_local)

        else:
            # no invalids: simple min/max
            dmin_local = d_patch.amin(dim=1, keepdim=True)  # (N,1,HW)
            dmax_local = d_patch.amax(dim=1, keepdim=True)  # (N,1,HW)
            valid_n = None
            valid_c = None

        # --- Your d_num and LINEAR wd
        d_num = torch.maximum((dmax_local - d_center).abs(), (d_center - dmin_local).abs())  # (N,1,HW)
        diff_abs = (d_patch - d_center).abs()                                                 # (N,P,HW)

        wd = 1.0 - (diff_abs / (d_num + eps_t))  # broadcast (N,1,HW) over P
        wd = wd.clamp(0.0, 1.0)

        # --- If invalids exist, mask them out
        if self.depth_pad_val is not None:
            wd = wd.masked_fill(~valid_n, 0.0)      # invalid neighbors contribute 0
            wd = wd.masked_fill(~valid_c, 0.0)      # if center invalid, kill all weights at that location

        # --- Expand to channels: (N,1,P,HW)
        wC = wd.unsqueeze(1)

        # --- Apply optional learnable spatial weights
        ws = self._get_ws(wd)
        if ws is not None:
           wC = wC * ws
           # wC = wC 

        # --- Weighted sum & normalize
        num = (x_patch * wC).sum(dim=2)              # (N,C,HW)
        den = wC.sum(dim=2).clamp_min(eps_t)         # (N,C,HW)
        out = (num / den).view(N, C, H, W)

        return out



class DepthGuidedStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        stride_first: int = 2,
        kernel_size: int = 3,
        normalize_depth: bool = True,
        norm_mode: str = "per_image",
        fixed_min: float = -2.17,
        fixed_max: float = 2.21,
        clamp_norm: bool = True,
        depth_pad_val: float = -3,
        use_residual: bool = True,
        init_res_scale: float = 0.1,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, stride=stride_first, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_convs - 1):
            layers += [
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]

        self.pre = nn.Sequential(*layers)

        self.refine = DepthGuidedRefine2x(
            channels=out_channels,
            kernel_size=kernel_size,
            normalize_depth=normalize_depth,
            norm_mode=norm_mode,
            fixed_min=fixed_min,
            fixed_max=fixed_max,
            clamp_norm=clamp_norm,
            depth_pad_val=depth_pad_val,
            use_residual=use_residual,
            init_res_scale=init_res_scale,
        )

    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)

        if depth.dim() == 3:
            depth = depth.unsqueeze(1)  # (N,1,H,W)
        if depth.shape[-2:] != x.shape[-2:]:
            depth = F.interpolate(depth, size=x.shape[-2:], mode="nearest")

        return self.refine(x, depth)



class DepthGuidedStage_old(nn.Module):
    """
    One stage:
      - project in_channels -> out_channels (with stride for downsample)
      - optionally extra convs
      - depth-guided refine (keeps out_channels)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        stride_first: int = 2,
        kernel_size: int = 3,
        sigma: float = 0.1,
        normalize_depth: bool = True,
        norm_mode: str = "per_image",
        fixed_min: float = -2.17,
        fixed_max: float = 2.21,
        clamp_norm: bool = True,
        depth_pad_val: float = -3,
        use_residual: bool = True,
        init_res_scale: float = 0.1,
    ):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride_first, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_convs - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        self.pre = nn.Sequential(*layers)

        self.refine = DepthGuidedRefine2x(
            channels=out_channels,
            kernel_size=kernel_size,
            sigma=sigma,
            normalize_depth=normalize_depth,
            norm_mode=norm_mode,
            fixed_min=fixed_min,
            fixed_max=fixed_max,
            clamp_norm=clamp_norm,
            depth_pad_val=depth_pad_val,
            use_residual=use_residual,
            init_res_scale=init_res_scale,
        )

    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)

        # ✅ Ensure depth matches current x spatial size
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)  # (N,1,H,W)
        if depth.shape[-2:] != x.shape[-2:]:
            depth = F.interpolate(depth, size=x.shape[-2:], mode="nearest")

        x = self.refine(x, depth)
        return x







class DepthGuidedBranch4Stage(nn.Module):
    """
    4-stage depth-guided branch:
      8 → 16 → 32 → 64

    Returns:
      f1,f2,f3,f4 if return_intermediate=True else f4

    IMPORTANT:
      depth is resized per stage to match feature spatial size (nearest).
    """
    def __init__(
        self,
        in_channels: int = 1,
        feat_channels=(8, 16, 32, 64),
        num_convs_per_stage: int = 2,
        kernel_size: int = 3,

        # DG params
        normalize_depth: bool = True,
        norm_mode: str = "per_image",
        fixed_min: float = -2.17,
        fixed_max: float = 2.21,
        clamp_norm: bool = True,
        depth_pad_val: float = -3,

        # residual behavior
        use_residual: bool = True,
        init_res_scale: float = 0.1,
    ):
        super().__init__()

        c1, c2, c3, c4 = feat_channels

        stage_kwargs = dict(
            num_convs=num_convs_per_stage,
            stride_first=2,
            kernel_size=kernel_size,
            normalize_depth=normalize_depth,
            norm_mode=norm_mode,
            fixed_min=fixed_min,
            fixed_max=fixed_max,
            clamp_norm=clamp_norm,
            depth_pad_val=depth_pad_val,
            use_residual=use_residual,
            init_res_scale=init_res_scale,
        )

        self.stage1 = DepthGuidedStage(in_channels, c1, **stage_kwargs)
        self.stage2 = DepthGuidedStage(c1, c2, **stage_kwargs)
        self.stage3 = DepthGuidedStage(c2, c3, **stage_kwargs)
        self.stage4 = DepthGuidedStage(c3, c4, **stage_kwargs)

    def _resize_depth(self, depth: torch.Tensor, size_hw):
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)  # (N,1,H,W)
        if depth.shape[-2:] != size_hw:
            depth = F.interpolate(depth, size=size_hw, mode="nearest")
        return depth

    def forward(self, x: torch.Tensor, depth: torch.Tensor, return_intermediate: bool = True):
        d1 = self._resize_depth(depth, x.shape[-2:])
        f1 = self.stage1(x, d1)

        d2 = self._resize_depth(depth, f1.shape[-2:])
        f2 = self.stage2(f1, d2)

        d3 = self._resize_depth(depth, f2.shape[-2:])
        f3 = self.stage3(f2, d3)

        d4 = self._resize_depth(depth, f3.shape[-2:])
        f4 = self.stage4(f3, d4)

        return (f1, f2, f3, f4) if return_intermediate else f4



class DepthGuidedBranch4Stage_old(nn.Module):
    """
    4-stage depth-guided branch:
      8 → 16 → 32 → 64

    Returns:
      f1,f2,f3,f4 if return_intermediate=True else f4

    IMPORTANT:
      depth should be aligned to x spatial size per stage.
      This code resizes depth per stage automatically (nearest).
    """
    def __init__(
        self,
        in_channels: int = 1,
        feat_channels=(8, 16, 32, 64),
        num_convs_per_stage: int = 2,
        sigma: float = 0.1,
        kernel_size: int = 3,
        # DG params
        normalize_depth: bool = True,
        norm_mode: str = "per_image",
        fixed_min: float = -2.17,
        fixed_max: float = 2.21,
        clamp_norm: bool = True,
        depth_pad_val: float = -3,
        use_residual: bool = True,
        init_res_scale: float = 0.1,
    ):
        super().__init__()

        c1, c2, c3, c4 = feat_channels

        self.stage1 = DepthGuidedStage(
            in_channels, c1,
            num_convs=num_convs_per_stage,
            stride_first=2,
            kernel_size=kernel_size,
            sigma=sigma,
            normalize_depth=normalize_depth,
            norm_mode=norm_mode,
            fixed_min=fixed_min,
            fixed_max=fixed_max,
            clamp_norm=clamp_norm,
            depth_pad_val=depth_pad_val,
            use_residual=use_residual,
            init_res_scale=init_res_scale,
        )

        self.stage2 = DepthGuidedStage(
            c1, c2,
            num_convs=num_convs_per_stage,
            stride_first=2,
            kernel_size=kernel_size,
            sigma=sigma,
            normalize_depth=normalize_depth,
            norm_mode=norm_mode,
            fixed_min=fixed_min,
            fixed_max=fixed_max,
            clamp_norm=clamp_norm,
            depth_pad_val=depth_pad_val,
            use_residual=use_residual,
            init_res_scale=init_res_scale,
        )

        self.stage3 = DepthGuidedStage(
            c2, c3,
            num_convs=num_convs_per_stage,
            stride_first=2,
            kernel_size=kernel_size,
            sigma=sigma,
            normalize_depth=normalize_depth,
            norm_mode=norm_mode,
            fixed_min=fixed_min,
            fixed_max=fixed_max,
            clamp_norm=clamp_norm,
            depth_pad_val=depth_pad_val,
            use_residual=use_residual,
            init_res_scale=init_res_scale,
        )

        self.stage4 = DepthGuidedStage(
            c3, c4,
            num_convs=num_convs_per_stage,
            stride_first=2,
            kernel_size=kernel_size,
            sigma=sigma,
            normalize_depth=normalize_depth,
            norm_mode=norm_mode,
            fixed_min=fixed_min,
            fixed_max=fixed_max,
            clamp_norm=clamp_norm,
            depth_pad_val=depth_pad_val,
            use_residual=use_residual,
            init_res_scale=init_res_scale,
        )

    def _resize_depth(self, depth: torch.Tensor, size_hw):
        # depth: (N,1,H,W) or (N,H,W)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.shape[-2:] != size_hw:
            depth = nn.functional.interpolate(depth, size=size_hw, mode="nearest")
        return depth

    def forward(self, x: torch.Tensor, depth: torch.Tensor, return_intermediate: bool = True):
        d1 = self._resize_depth(depth, x.shape[-2:])
        f1 = self.stage1(x, d1)

        d2 = self._resize_depth(depth, f1.shape[-2:])
        f2 = self.stage2(f1, d2)

        d3 = self._resize_depth(depth, f2.shape[-2:])
        f3 = self.stage3(f2, d3)

        d4 = self._resize_depth(depth, f3.shape[-2:])
        f4 = self.stage4(f3, d4)

        if return_intermediate:
            return f1, f2, f3, f4
        return f4



class DepthGuidedRefine2x(nn.Module):
    def __init__(
        self,
        channels=128,
        kernel_size=3,

        # depth normalization controls passed to DG
        normalize_depth=True,
        norm_mode="per_image",     # "per_image" | "per_batch" | "fixed" | "none"
        fixed_min=-2.17,
        fixed_max=2.21,
        clamp_norm=True,
        depth_pad_val=-3,

        # OPTIONAL: depth-weight stability/sharpness (if you added these to DG)
        # If you did NOT add them, remove these two lines and args below.
        # dnum_min=1e-3,
        # wd_gamma=1.0,

        # spatial weights options (same as before)
        learnable_spatial=True,
        shared_spatial=False,

        # block behavior
        use_residual=True,
        init_res_scale=0.1,
    ):
        super().__init__()
        self.use_residual = bool(use_residual)

        self.dg1 = DepthSimilarityWeightedConv(
            channels=channels,
            kernel_size=kernel_size,
            depth_pad_val=depth_pad_val,

            normalize_depth=normalize_depth,
            norm_mode=norm_mode,
            fixed_min=fixed_min,
            fixed_max=fixed_max,
            clamp_norm=clamp_norm,

            learnable_spatial=learnable_spatial,
            shared_spatial=shared_spatial,

            # If your DG class supports these, pass them:
            # dnum_min=dnum_min,
            # wd_gamma=wd_gamma,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.ReLU(inplace=True)

        self.dg2 = DepthSimilarityWeightedConv(
            channels=channels,
            kernel_size=kernel_size,
            depth_pad_val=depth_pad_val,

            normalize_depth=normalize_depth,
            norm_mode=norm_mode,
            fixed_min=fixed_min,
            fixed_max=fixed_max,
            clamp_norm=clamp_norm,

            learnable_spatial=learnable_spatial,
            shared_spatial=shared_spatial,

            # If your DG class supports these, pass them:
            # dnum_min=dnum_min,
            # wd_gamma=wd_gamma,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.ReLU(inplace=True)

        # learnable residual scale (starts small)
        self.res_scale = nn.Parameter(torch.tensor(float(init_res_scale)))

    def forward(self, x, depth):
        identity = x

        out = self.dg1(x, depth)
        out = self.act1(self.bn1(out))

        out = self.dg2(out, depth)
        out = self.act2(self.bn2(out))

        if self.use_residual:
            return identity + self.res_scale * out
        else:
            return self.res_scale * out


class DepthGuidedRefine2x_old(nn.Module):
    def __init__(
        self,
        channels=128,
        kernel_size=3,
        sigma=0.1,

        # --- NEW: normalization controls passed to DG ---
        normalize_depth=True,
        norm_mode="per_image",     # "per_image" | "per_batch" | "fixed" | "none"
        fixed_min=-2.17,
        fixed_max=2.21,
        clamp_norm=True,
        depth_pad_val=-3,

        # block behavior
        use_residual=True,
        init_res_scale=0.1,
    ):
        super().__init__()
        self.use_residual = bool(use_residual)

        self.dg1 = DepthSimilarityWeightedConv(
            channels=channels,
            kernel_size=kernel_size,
            sigma=sigma,
            depth_pad_val=depth_pad_val,
            normalize_depth=normalize_depth,
            norm_mode=norm_mode,
            fixed_min=fixed_min,
            fixed_max=fixed_max,
            clamp_norm=clamp_norm,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.ReLU(inplace=True)

        self.dg2 = DepthSimilarityWeightedConv(
            channels=channels,
            kernel_size=kernel_size,
            sigma=sigma,
            depth_pad_val=depth_pad_val,
            normalize_depth=normalize_depth,
            norm_mode=norm_mode,
            fixed_min=fixed_min,
            fixed_max=fixed_max,
            clamp_norm=clamp_norm,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.ReLU(inplace=True)

        # learnable residual scale (starts small)
        self.res_scale = nn.Parameter(torch.tensor(float(init_res_scale)))

    def forward(self, x, depth):
        identity = x

        out = self.dg1(x, depth)
        out = self.act1(self.bn1(out))

        out = self.dg2(out, depth)
        out = self.act2(self.bn2(out))

        if self.use_residual:
            return identity + self.res_scale * out
        else:
            return self.res_scale * out



class SemanticBranch(BaseModule):
    """Semantic Branch which is lightweight with narrow channels and deep
    layers to obtain　high-level semantic context.

    Args:
        semantic_channels(Tuple[int]): Size of channel numbers of
            various stages in Semantic Branch.
            Default: (16, 32, 64, 128).
        in_channels (int): Number of channels of input image. Default: 3.
        exp_ratio (int): Expansion ratio for middle channels.
            Default: 6.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Returns:
        semantic_outs (List[torch.Tensor]): List of several feature maps
            for auxiliary heads (Booster) and Bilateral
            Guided Aggregation Layer.
    """

    def __init__(self,
                 semantic_channels=(16, 32, 64, 128),
                 in_channels=3,
                 exp_ratio=6,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.semantic_channels = semantic_channels
        self.semantic_stages = []
        for i in range(len(semantic_channels)):
            stage_name = f'stage{i + 1}'
            self.semantic_stages.append(stage_name)
            if i == 0:
                self.add_module(
                    stage_name,
                    StemBlock(self.in_channels, semantic_channels[i]))
            elif i == (len(semantic_channels) - 1):
                self.add_module(
                    stage_name,
                    nn.Sequential(
                        GELayer(semantic_channels[i - 1], semantic_channels[i],
                                exp_ratio, 2),
                        GELayer(semantic_channels[i], semantic_channels[i],
                                exp_ratio, 1),
                        GELayer(semantic_channels[i], semantic_channels[i],
                                exp_ratio, 1),
                        GELayer(semantic_channels[i], semantic_channels[i],
                                exp_ratio, 1)))
            else:
                self.add_module(
                    stage_name,
                    nn.Sequential(
                        GELayer(semantic_channels[i - 1], semantic_channels[i],
                                exp_ratio, 2),
                        GELayer(semantic_channels[i], semantic_channels[i],
                                exp_ratio, 1)))

        self.add_module(f'stage{len(semantic_channels)}_CEBlock',
                        CEBlock(semantic_channels[-1], semantic_channels[-1]))
        self.semantic_stages.append(f'stage{len(semantic_channels)}_CEBlock')

    def forward(self, x):
        semantic_outs = []
        for stage_name in self.semantic_stages:
            semantic_stage = getattr(self, stage_name)
            x = semantic_stage(x)
            semantic_outs.append(x)
        return semantic_outs


class BGALayer(BaseModule):
    """Bilateral Guided Aggregation Layer to fuse the complementary information
    from both Detail Branch and Semantic Branch.

    Args:
        out_channels (int): Number of output channels.
            Default: 128.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        conv_cfg (dict | None): Config of conv layers.
            Default: None.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Returns:
        output (torch.Tensor): Output feature map for Segment heads.
    """

    def __init__(self,
                 out_channels=128,
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        #self.alpha = nn.Parameter(torch.tensor(0.0))  # scalar in ℝ
        #self.mix_conv = nn.Sequential(nn.Conv2d(C1 + C2, C_out, kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(C_out),nn.ReLU(inplace=True))
        ##self.mix_conv = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.out_channels = out_channels
        self.align_corners = align_corners
        """
        self.detail_dwconv = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dw_norm_cfg=norm_cfg,
                dw_act_cfg=None,
                pw_norm_cfg=None,
                pw_act_cfg=None,
            ))
        
        self.detail_down = nn.Sequential(
            ConvModule(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        """
        """
        self.semantic_conv = nn.Sequential(
            ConvModule(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None))
        
        self.semantic_dwconv = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dw_norm_cfg=norm_cfg,
                dw_act_cfg=None,
                pw_norm_cfg=None,
                pw_act_cfg=None,
            ))
        """
        
        self.semantic_dwconv_rgb = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dw_norm_cfg=norm_cfg,
                dw_act_cfg=None,
                pw_norm_cfg=None,
                pw_act_cfg=None,
            ))
        
        """
        self.thicker = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.SiLU(inplace=True),
        )
        """
        """
        self.conv = ConvModule(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            inplace=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        """
        '''
        self.mix_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=128,   # input feature maps
                out_channels=128,  # output feature maps
                kernel_size=3,
                padding=2,         # keeps spatial size unchanged
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        '''
        
    def forward(self, x_my):
        #detail_dwconv = self.detail_dwconv(x_d) # my comment
        #detail_dwconv = self.detail_dwconv(x_d) # my comment
       #detail_down = self.detail_down(x_d)
        #semantic_conv = self.semantic_conv(x_s)
        #semantic_dwconv = self.semantic_dwconv(x_s) #my comment
        #etail_dwconv_= self.detail_dwconv(x_my)# my comment
       #detail_down_= self.detail_down(x_my)
        #semantic_dwconv = self.semantic_dwconv(x_s) 
        rgb_dwconv = self.semantic_dwconv_rgb(x_my) 
        
        #dil = torch.nn.functional.max_pool2d(rgb_dwconv, kernel_size=5, stride=1, padding=5 // 2)
        #rgb_dwconv=rgb_dwconv+self.thicker(dil)
        """
        semantic_conv = resize(
            input=semantic_conv,
            size=detail_dwconv.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        """
        #fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv) # my comment
        #fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv) # my comment
        #####fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv) #
        #fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv)
        ##fuse_2 = detail_down * torch.sigmoid(semantic_dwconv) #my comment
        ###fuse_2 = semantic_dwconv #my comment
        #$$fuse_2 = semantic_dwconv * torch.sigmoid(detail_down)#my comment
        fuse_2 =torch.sigmoid(rgb_dwconv)
        ###fuse_3 = semantic_dwconv * torch.sigmoid(depth_dwconv)
        #######fuse_3 =  semantic_dwconv * torch.sigmoid(rgb_dwconv)
        #fuse_4= detail_down * torch.sigmoid(depth_dwconv)
        ###fuse_3 =  depth_dwconv
        #fuse_2 = semantic_dwconv
        #fuse_3 = detail_down * depth_dwconv
        ##fuse_3 = semantic_dwconv * depth_dwconv
        ##fuse_3 =   detail_down * depth_dwconv
        ###fuse_3 =   fuse_2 * torch.sigmoid(depth_dwconv)
        #fuse_3 = torch.sigmoid(depth_dwconv)
        #alpha = torch.sigmoid(self.alpha)    # self.alpha = nn.Parameter(torch.tensor(0.0))
        #gate  = torch.sigmoid(semantic_dwconv)
        #fuse_2 = (1 - alpha) * detail_down + alpha * (detail_down * gate)
        """
        fuse_2 = resize(
            input=fuse_2,
            size=fuse_1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        """
        """
        fuse_3 = resize(
            input=fuse_3,
            size=fuse_1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        """
        #output = self.conv(fuse_1 + fuse_2)
        #x = torch.cat([fuse_2], dim=1)              # [B, C+C2, H, W]
        #output = self.mix_conv(x)  
        output=fuse_2                          # nn.Conv2d(C+C2, C_out, kernel_size=1)
        return output


class SemanticBranchNormal(nn.Module):
    """
    Generic CNN semantic branch with ANY number of stages.

    feat_channels example:
        (8,16)
        (8,16,32)
        (8,16,32,64)
        (8,16,32,64,128)

    Each stage:
        first conv stride=2 (downsample)
    """

    def __init__(self,
                 in_channels=3,
                 feat_channels=(8, 16, 32, 64),
                 num_convs_per_stage=2,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 conv_cfg=None):

        super().__init__()

        self.stages = nn.ModuleList()

        prev_c = in_channels

        for out_c in feat_channels:
            self.stages.append(
                self._make_stage(
                    prev_c,
                    out_c,
                    num_convs_per_stage,
                    norm_cfg,
                    act_cfg,
                    conv_cfg
                )
            )
            prev_c = out_c

    # -------------------------------------------------------
    def _make_stage(self,
                    in_channels,
                    out_channels,
                    num_convs,
                    norm_cfg,
                    act_cfg,
                    conv_cfg):

        layers = []

        for i in range(num_convs):
            stride = 2 if i == 0 else 1
            in_c = in_channels if i == 0 else out_channels

            layers.append(
                ConvModule(
                    in_c,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

        return nn.Sequential(*layers)

    # -------------------------------------------------------
    def forward(self, x, return_intermediate=True):

        feats = []

        for stage in self.stages:
            x = stage(x)
            feats.append(x)

        if return_intermediate:
            return feats  # list of features

        return feats[-1]


class FuseConcatSE(nn.Module):
    def __init__(self, c, out_c=None, r=16):
        super().__init__()
        out_c = out_c or c
        self.mix = nn.Sequential(
            nn.Conv2d(2*c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        # SE gate
        self.fc1 = nn.Conv2d(out_c, max(out_c // r, 4), 1)
        self.fc2 = nn.Conv2d(max(out_c // r, 4), out_c, 1)

    def forward(self, f1, f2):
        x = torch.cat([f1, f2], dim=1)
        x = self.mix(x)
        g = F.adaptive_avg_pool2d(x, 1)
        g = F.relu(self.fc1(g), inplace=True)
        g = torch.sigmoid(self.fc2(g))
        return x * g

@MODELS.register_module()
class BiSeNetV2(BaseModule):
    """BiSeNetV2: Bilateral Network with Guided Aggregation for
    Real-time Semantic Segmentation.

    This backbone is the implementation of
    `BiSeNetV2 <https://arxiv.org/abs/2004.02147>`_.

    Args:
        in_channels (int): Number of channel of input image. Default: 3.
        detail_channels (Tuple[int], optional): Channels of each stage
            in Detail Branch. Default: (64, 64, 128).
        semantic_channels (Tuple[int], optional): Channels of each stage
            in Semantic Branch. Default: (16, 32, 64, 128).
            See Table 1 and Figure 3 of paper for more details.
        semantic_expansion_ratio (int, optional): The expansion factor
            expanding channel number of middle channels in Semantic Branch.
            Default: 6.
        bga_channels (int, optional): Number of middle channels in
            Bilateral Guided Aggregation Layer. Default: 128.
        out_indices (Tuple[int] | int, optional): Output from which stages.
            Default: (0, 1, 2, 3, 4).
        align_corners (bool, optional): The align_corners argument of
            resize operation in Bilateral Guided Aggregation Layer.
            Default: False.
        conv_cfg (dict | None): Config of conv layers.
            Default: None.
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 #C=128, #my code
                 #in_channels=3, # my comments
                 in_channels=6, #my code
                 detail_channels=(64, 64, 128),
                 semantic_channels=(16, 32, 64, 128),
                 semantic_expansion_ratio=6,
                 bga_channels=128,
                 #out_indices=(0, 1, 2, 3, 4),
                 out_indices=(0, 1,2,3,4),
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):

        if init_cfg is None:
            init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(
                    type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
            ]
        super().__init__(init_cfg=init_cfg)
        #self.fuser = nn.Sequential(nn.Conv2d(2*C, C, kernel_size=1, bias=False),nn.BatchNorm2d(C),nn.Sigmoid(),) #my code
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.detail_channels = detail_channels
        self.semantic_channels = semantic_channels
        self.semantic_expansion_ratio = semantic_expansion_ratio
        self.bga_channels = bga_channels
        self.align_corners = align_corners
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg


        """
        #my code-->
        self.detail = DetailBranch(self.detail_channels, self.in_channels//2)
        self.semantic = SemanticBranch(self.semantic_channels,
                                       self.in_channels//2,
                                       self.semantic_expansion_ratio)
        """
        """
        self.semantic_rgb = SemanticBranch(self.semantic_channels,
                                       self.in_channels//2,
                                       self.semantic_expansion_ratio)
        """
        #my code^^^
        # in __init__
        """"
        self.semantic_branch = SemanticBranch4Stage(
            in_channels=self.in_channels//2,                 # or 4/6 if you feed RGB+extra channels
            feat_channels=(16, 32, 64, 128),
            num_convs_per_stage=2,
            norm_cfg=norm_cfg
        )
        """
        """
        self.semantic_branch_1 = SemanticBranch4StageDGOnly34(
            in_channels=8,
            feat_channels=(8, 16, 32, 64),
            num_convs_per_stage=2,
            norm_cfg=norm_cfg,
            dg_sigma=0.1,
            depth_pad_val=-3,
            dg_residual=True,
            dg_normalize_depth=True,
            dg_norm_mode="per_image",
            dg_fixed_min=-2.17,
            dg_fixed_max=2.21,
            dg_clamp_norm=True
        )

        self.semantic_branch_2 = SemanticBranch4StageDGOnly34(
            in_channels=8,
            feat_channels=(8, 16, 32, 64),
            num_convs_per_stage=2,
            norm_cfg=norm_cfg,
            dg_sigma=0.1,
            depth_pad_val=-3,
            dg_residual=True,
            dg_normalize_depth=True,
            dg_norm_mode="per_image",
            dg_fixed_min=-2.17,
            dg_fixed_max=2.21,
            dg_clamp_norm=True
        )

        self.semantic_branch_3 = SemanticBranch4StageDGOnly34(
            in_channels=8,
            feat_channels=(8, 16, 32, 64),
            num_convs_per_stage=2,
            norm_cfg=norm_cfg,
            dg_sigma=0.1,
            depth_pad_val=-3,
            dg_residual=True,
            dg_normalize_depth=True,
            dg_norm_mode="per_image",
            dg_fixed_min=-2.17,
            dg_fixed_max=2.21,
            dg_clamp_norm=True
        )

        self.semantic_branch_4 = SemanticBranch4StageDGOnly34(
            in_channels=8,
            feat_channels=(8, 16, 32, 64),
            num_convs_per_stage=2,
            norm_cfg=norm_cfg,
            dg_sigma=0.1,
            depth_pad_val=-3,
            dg_residual=True,
            dg_normalize_depth=True,
            dg_norm_mode="per_image",
            dg_fixed_min=-2.17,
            dg_fixed_max=2.21,
            dg_clamp_norm=True
        )
        """
        self.semantic_branch_1 = SemanticBranchNormal(in_channels=3, feat_channels=(8, 16, 32, 64), num_convs_per_stage=2, norm_cfg=norm_cfg)
        self.semantic_branch_2 = SemanticBranchNormal(in_channels=1 , feat_channels=(8, 16, 32, 64), num_convs_per_stage=2, norm_cfg=norm_cfg)
        self.semantic_branch_3= SemanticBranchNormal(in_channels=64 , feat_channels=(64, 128), num_convs_per_stage=2, norm_cfg=norm_cfg)
        #self.semantic_branch_3 = SemanticBranchNormal(in_channels=3, feat_channels=(8, 16), num_convs_per_stage=2, norm_cfg=norm_cfg)
        #self.semantic_branch_4 = SemanticBranchNormal(in_channels=3, feat_channels=(8, 16), num_convs_per_stage=2, norm_cfg=norm_cfg)
        #self.semantic_branch_5 = SemanticBranchNormal(in_channels=16, feat_channels=(32,32), num_convs_per_stage=2, norm_cfg=norm_cfg)
        #self.semantic_branch_6 = SemanticBranchNormal(in_channels=32, feat_channels=(64,64), num_convs_per_stage=2, norm_cfg=norm_cfg)
        #self.semantic_branch_7 = SemanticBranchNormal(in_channels=64, feat_channels=(128, 128), num_convs_per_stage=2, norm_cfg=norm_cfg)
        #self.conv_16=nn.Conv2d(16, 32, 3, stride=2, padding=1)
        #self.conv_32=nn.Conv2d(32, 64, 3, stride=2, padding=1)
        #self.conv_64=nn.Conv2d(64, 128, 3, stride=2, padding=1)

        #self.semantic_branch_5 = SemanticBranch4StageNormal(in_channels=1, feat_channels=(8,16,32,64), num_convs_per_stage=2, norm_cfg=norm_cfg)
        """
        self.refine = DepthGuidedRefine2x(
            channels=64,
            kernel_size=3,

            normalize_depth=True,
            norm_mode="per_image",   # recommended
            fixed_min=-2.17,
            fixed_max=2.21,
            clamp_norm=True,
            depth_pad_val=-3,

            learnable_spatial=True,  # per-channel spatial kernels (strongest)
            shared_spatial=False,

            use_residual=True,
            init_res_scale=0.1,
        )
        """

        '''
        self.detail = DetailBranch(self.detail_channels, self.in_channels)
        self.semantic = SemanticBranch(self.semantihannels,
                                       self.in_channels,
                                       self.semantic_expansion_ratio)
        
        #'''
        """
        self.depth_guided = DepthSimilarityWeightedConv(
            channels=128,
            kernel_size=3,
            sigma=0.1,
            learnable_spatial=True
        )
        """
        """
        self.depth_refine = DepthGuidedRefine2x(
                channels=64,
                kernel_size=3,
                sigma=0.1,

                # ---- NEW depth normalization options ----
                normalize_depth=True,
                norm_mode="per_image",     # or "fixed"
                fixed_min=-2.17,
                fixed_max=2.21,
                clamp_norm=True,

                depth_pad_val=-3,

                # optional behaviour
                use_residual=True,
                init_res_scale=0.1
            )
        """
        """
        self.depth_rgb_slicer = DepthRGBSlicer128to128(
            n_slices=128,
            out_channels=128,
            invalid_depth_val=-3,        # if you use -3 as pad/invalid depth
            use_per_image_minmax=True
        )
        
        self.depth_rgb_slicer = DepthRGBSlicer128to128(
            n_slices=4,
            invalid_depth_val=-3,
            use_per_image_minmax=True,
                )c_c
        self.depth_rgb_slicer_1 = DepthRGBSlicer128to128(
            n_slices=4,
            invalid_depth_val=-3,
            use_per_image_minmax=True,
                )
        
        self.depth_rgb_slicer = DepthRGBSlicer128to128_1(
                n_slices=4,
                invalid_depth_val=None,   # set if you have invalid depth flag
                eps=1e-6,
                use_per_image_minmax=True
            )
        """
        """
        self.semantic_branch_1 = DepthGuidedBranch4Stage(
            in_channels=1,
            feat_channels=(8, 16, 32, 64),
            num_convs_per_stage=2,
            kernel_size=3,

            normalize_depth=True,
            norm_mode="per_image",   # or "fixed" if you want consistent scaling
            fixed_min=-2.17,
            fixed_max=2.21,
            clamp_norm=True,
            depth_pad_val=-3,

            use_residual=True,
            init_res_scale=0.1,
        )
        """

        """
        self.semantic_branch_2 = DepthGuidedBranch4Stage(
            in_channels=1,
            feat_channels=(8, 16, 32, 64),
            num_convs_per_stage=2,
            sigma=0.1
        )
        self.semantic_branch_3 = DepthGuidedBranch4Stage(
            in_channels=1,
            feat_channels=(8, 16, 32, 64),
            num_convs_per_stage=2,
            sigma=0.1
        )
        self.semantic_branch_4 = DepthGuidedBranch4Stage(
            in_channels=1,
            feat_channels=(8, 16, 32, 64),
            num_convs_per_stage=2,
            sigma=0.1
        )
        
        """
        
        self.depth_slicer = DepthRGBSlicer_Shared3x3(
            n_slices=4,                 # number of depth slices you want
            invalid_depth_val=None,     # or e.g. -3 if you use invalid depth
            eps=1e-6,
            use_per_image_minmax=True
        )
        

        """
        self.depth_rgb_slicer_1 = DepthRGBSlicer128to128(
            n_slices=512,
            out_channels=512,
            invalid_depth_val=-3,        # if you use -3 as pad/invalid depth
            use_per_image_minmax=True
        )
        """
        #self.fuse = nn.Conv2d(64 * 4, 64, kernel_size=1, bias=False)
        #self.fuse_bn = nn.BatchNorm2d(64)
        #self.fuse_act = nn.ReLU(inplace=True)
        #self.fuse4 = nn.Conv2d(64*4, 64, kernel_size=1)
        #self.fuse3 = nn.Conv2d(64*3, 64, kernel_size=1)
        #self.fuse2 = nn.Conv2d(64*2, 64, kernel_size=1)
        #self.rgb_to_4 = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False)

        #self.channel_mix = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        #self.channel_mix_1 = nn.Conv2d(32, 16, kernel_size=1, bias=False)
        #self.channel_mix_2 = nn.Conv2d(64, 16, kernel_size=1, bias=False)
        self.concat=FuseConcatSE(c=64)
        self.bga = BGALayer(self.bga_channels, self.align_corners)

    def forward(self, x):
        #  stole refactoring code from Coin Cheung, thanks
        #x_detail = self.detail(x) # my comments
        #x_semantic_lst = self.semantic(x) # my comments
        """
        #-> my code
        # Assume input x is [B, 6, H, W] => split into 2x [B, 3, H, W]
        assert x.shape[1] == 6, f"Expected 6 channels, but got {x.shape[1]}"
        x_detail_input = x[:, :3, :, :]     
        
        x_semantic_input =  x[:, 3:, :, :] 
        x_detail = self.detail(x_detail_input)
        x_semantic_lst = self.semantic(x_semantic_input)
        #^^^ my code
        #"""
        #"""
        #latest
        #-> my code to include depth and rgb semantic branch
        # Assume input x is [B, 6, H, W] => split into 2x [B, 3, H, W]
        assert x.shape[1] == 6, f"Expected 6 channels, but got {x.shape[1]}"
        #x_detail_input = x[:, :3, :, :]   
        x_detail_input=x[:,3: ,:, :]
  
        x_semantic_input_depth = x[:, 3:4 , :, :] 
        x_semantic_input_rgb =x[:, :3, :, :]
        x_depth_raw=x[:,3:4, :, :] 
        x_depth_raw = x[:, 3:4, :, :]
        
        # in forward(self, x):
        # x is the input tensor (N,C,H,W)
        #f1, f2, f3, f4 = self.semantic_branch(x_semantic_input_rgb,x_depth_raw)
        
        #x16 = self.depth_rgb_slicer(x_semantic_input_rgb, x_depth_raw)
        #x32 = self.depth_rgb_slicer(x_semantic_input_rgb, x_depth_raw)   # (N,32,H,W)
        #x512 = self.depth_rgb_slicer(x_semantic_input_rgb, x_depth_raw)
        
        rgb_slices = self.depth_slicer(x_semantic_input_rgb, x_depth_raw)
        x4_1 = rgb_slices[:, 0]
        x4_2 = rgb_slices[:, 1]
        x4_3 = rgb_slices[:, 2]
        x4_4 = rgb_slices[:, 3]
        
        """
        N, S, C, H, W = rgb_slices.shape

        x_1 = rgb_slices[:, 0].reshape(N, 3, H, W)
        x_2 = rgb_slices[:, 1].reshape(N, 3, H, W)
        x_3 = rgb_slices[:, 2].reshape(N, 3, H, W)
        x_4 = rgb_slices[:, 3].reshape(N, 3, H, W)

        y_1 = self.rgb_to_4(x_1)
        y_2 = self.rgb_to_4(x_2)
        y_3 = self.rgb_to_4(x_3)
        y_4 = self.rgb_to_4(x_4)

        out_1 = y_1.reshape(N, 1, 1, H, W)
        out_2 = y_2.reshape(N, 1, 1, H, W)
        out_3 = y_3.reshape(N, 1, 1, H, W)
        out_4 = y_4.reshape(N, 1, 1, H, W)
        """
        
        #g1 = x32[:, 0:8,  :, :]   # channels 1-8   (0-7)
        #g2 = x32[:, 8:16, :, :]   # channels 9-16  (8-15)
        #g3 = x32[:, 16:24, :, :]  # channels 17-24 (16-23)
        #g4 = x32[:, 24:32, :, :]  # channels 25-32 (24-31)
        """
        g5 = x512_1[:, 0:1, :, :] # channels 25-32 (24-31)
        g6 = x512_1[:, 1:2, :, :]  # channels 25-32 (24-31)
        g7 = x512_1[:, 2:3, :, :]  # channels 25-32 (24-31)
        g8 = x512_1[:, 3:4, :, :]
        """
        #g5 = x512_1[:, 0]
        #g6 = x512_1[:, 1]
        #g7 = x512_1[:, 2]
        #g8 = x512_1[:, 3]
        #f1, f2, f3, f4 = self.semantic_branch(x16, x_depth_raw)
        #_, _, _, f1 = self.semantic_branch_1(g1, x_depth_raw)
        #_, _, _, f2 = self.semantic_branch_2(g2, x_depth_raw)
        #_, _, _, f3 = self.semantic_branch_3(g3, x_depth_raw)
        #_, _, _, f4 = self.semantic_branch_4(g4, x_depth_raw)
        """
        _, _, _, f1 = self.semantic_branch_1(out[:, 0])
        _, _, _, f2 = self.semantic_branch_1(out[:, 1])
        _, _, _, f3 = self.semantic_branch_1(out[:, 2])
        _, _, _, f4 = self.semantic_branch_1(out[:, 3])
        """
        #rgb_= torch.cat([x_semantic_input_rgb, x4_3], dim=1)  
        _, _, f32_1, f1 = self.semantic_branch_1(x_semantic_input_rgb)
        _, _, f32_2, f2 = self.semantic_branch_2(x_semantic_input_depth)
        #x_cat= torch.cat([f1,f2], dim=1)
        out=self.concat(f1,f2)
        #out = self.channel_mix(x_cat)
        #out=torch.sigmoid(f2)*f1
        _, f3 = self.semantic_branch_3(out)
        #_, f4 = self.semantic_branch_3(f2)

        #out_1 = out_1.squeeze(2)
        #out_2 = out_2.squeeze(2)
        #out_3 = out_3.squeeze(2)
        #out_4 = out_4.squeeze(2)
        #Ht, Wt = f1.shape[-2:]   # choose highest-res among finals (likely branch4)
        #x_depth = F.interpolate(x_depth_raw, size=(Ht,Wt), mode="bilinear", align_corners=False)
        #f_refined=self.refine(f1,x_depth)
        #f5=torch.sigmoid(f3)*f4
        #f4=torch.sigmoid(f2)*f1
        #x_cat= torch.cat([f1, f2], dim=1)  
        #out = self.channel_mix(x_cat)
        #x_cat_= torch.cat([x4_1, x4_2, x4_3, x4_4], dim=1)  
        #f8, f16, f32, f64 = self.semantic_branch_1(out)                 # all
        """
        _, f16_1= self.semantic_branch_4(x4_1)

        _, f16_2= self.semantic_branch_3(x4_2)
        f16_2=f16_1+f16_2
        f32_2=self.conv_16(f16_2)
        """
        #Ht, Wt = f32_2.shape[-2:]   # choose highest-res among finals (likely branch4)
        #f32_1 = F.interpolate(f16_1, size=(Ht,Wt), mode="bilinear", align_corners=False)
        #x_cat_64=x_cat= torch.cat([f32_1, f32_2], dim=1)   
        """
        _, f16_3, f32_3 = self.semantic_branch_2(x4_3)
        f32_3=f32_3+f32_2
        f64_3=self.conv_32(f32_3)
    
        _, f16_4, f32_4, f64_4 = self.semantic_branch_1(x4_4)
        f64_4=f64_3+f64_4
        f128=self.conv_64(f64_4)
    
        """
        #f5 = self.semantic_branch_5(out_5, return_intermediate=False)
        #f2 = self.semantic_branch_1(out[:, 1], return_intermediate=False) 
        #f3 = self.semantic_branch_1(out[:, 2], return_intermediate=False) 
        #f4 = self.semantic_branch_1(out[:, 3], return_intermediate=False) 
        #x_detail = self.detail(x_detail_input)
       #x_detail_= self.detail(x_detail_rgb)
        #x_semantic_lst_1 = self.semantic(x_semantic_input_depth)
         # (N,16,H,W)
        #x_semantic_lst_2 = [f1, f2, f3, f4]
        #x_semantic_lst_2 = [f8, f16, f32, f64]
        """
        x_sem_1=[f4,f3]
        x_sem_2=[f3,f2]
        x_sem_3=[f2,f1]
        x_cat_1 = torch.cat(x_sem_1, dim=1)    
        x_cat_2 = torch.cat(x_sem_2, dim=1)    
        x_cat_3 = torch.cat(x_sem_3, dim=1)    
        out_1 = self.channel_mix_1(x_cat_1) 
        out_2 = self.channel_mix_2(x_cat_2) 
        out_3 = self.channel_mix_3(x_cat_3)
        x_sem_4=[out_1,out_2, out_3]
        """
       #f2 = F.interpolate(f2, size=(Ht,Wt), mode="bilinear", align_corners=False)
        #3 = F.interpolate(f3, size=(Ht,Wt), mode="bilinear", align_corners=False)

        #f1 = self.proj1(f1)  # -> (N,64,Ht,Wt)
        #f2 = self.proj2(f2)
        #f3 = self.proj3(f3)
        #f4 = self.proj4(f4)
        #x_cat= torch.cat([f1], dim=1)    
        #x_cat_1= torch.cat([f32_1, f32_2, f32_3, f32_4], dim=1) 
        #x_cat_2= torch.cat([f16_1, f16_2, f16_3, f16_4], dim=1)    
   
        
        #s1 = self.fuse4(torch.cat([f1,f2,f3,f4], dim=1))
        #s2 = self.fuse3(torch.cat([f2,f3,f4], dim=1))
        #s3 = self.fuse2(torch.cat([f3,f4], dim=1))

        ##x_sem=[f4,f1,f2,f3]
        ##x_cat = torch.cat(x_sem, dim=1)      # (N,256,H,W)
        #x_semantic = self.fuse_act(self.fuse_bn(self.fuse(x_cat)))  # (N,64,H,W)
        #out = self.channel_mix(x_cat)   # x: (N,256,H,W)
        #out_1 = self.channel_mix_1(x_cat_1)
        #out_2 = self.channel_mix_2(x_cat_2)
        x_semantic_lst_2 = [f32_1,f1,f2, f32_2]

        #x_semantic_lst_2 = self.semantic_rgb(x_semantic_input_rgb)
        #x_cat = torch.cat([x_semantic_lst_2[-1], x_semantic_lst_1[-1]], dim=1)   #conv_4   # (N, 2C, H, W)
        #x_semantic_lst = self.fuser(x_cat) #conv_4
        #x_semantic_lst=x_semantic_lst_1[-1]*x_semantic_lst_1[-1]
        # x_sem: (N,128,Hs,Ws) from semantic branch
        # depth_raw: (N,1,H,W) or (N,H,W)

        depth_sem = x_depth_raw
        if depth_sem.dim() == 3:
            depth_sem = depth_sem.unsqueeze(1)  # (N,1,H,W)

        # resize depth to semantic feature resolution
        depth_sem = F.interpolate(
            depth_sem.float(),
            size=x_semantic_lst_2[-1].shape[-2:],
            mode='nearest'
        )

        # apply depth-guided refinement
       #x_sem =x_semantic_lst_1[-1]+ self.depth_guided(x_semantic_lst_1[-1],depth_sem)
        #x_sem = self.depth_refine(x_semantic,depth_sem)


        x_head = self.bga(f3)
        #x_head = self.bga(x_detail, x_semantic_lst_1[-1], x_semantic_lst_2[-1])
        #outs = [x_head] + x_semantic_lst_2[2:4]+x_semantic_lst_1[2:4] #conv_5
        #outs = [x_head] + x_semantic_lst_2[1:3]+ x_semantic_lst_1[2:4]
        #outs = [x_head] + x_semantic_lst_2[:-1]+x_semantic_lst_1[:-1] #conv_4

        #""" latest
        #x_fused = x_head + x_detail  #my code  
        #outs = [x_fused] + x_semantic_lst[:-1] #my comment#
        #x_head = self.bga(x_detail, x_semantic_lst[-1])
        outs = [x_head] + x_semantic_lst_2[0:]#+x_semantic_lst_1[2:4]
        #outs = [outs[i] for i in self.out_indices]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
        
    '''def forward(self, inputs):
    # inputs is now a dict with two image streams
        img_detail = inputs['img_detail']
        img_semantic = inputs['img_semantic']

        x_detail = self.detail(img_detail)
        x_semantic_lst = self.semantic(img_semantic)
        x_head = self.bga(x_detail, x_semantic_lst[-1])
        outs = [x_head] + x_semantic_lst[:-1]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)'''

