# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      build_activation_layer, build_norm_layer)
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from ..utils import resize
import torch.nn.functional as F

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




import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthSimilarityWeightedConv(nn.Module):
    """
    Depth-similarity guided local aggregation (bilateral-style) with OPTIONAL
    per-channel learnable spatial kernels.

    Keeps channels the same: (N,C,H,W) -> (N,C,H,W)

    For each pixel p and channel c:
        out[p,c] = sum_{q in kxk}  x[q,c] * ws[c, q] * wd[p,q]  /  sum_{q in kxk} ws[c,q] * wd[p,q]

    where:
      - wd[p,q] = exp( -(D(q)-D(p))^2 / (2*sigma^2) )   (depth similarity)
      - ws[c,q] is a learnable weight for each channel and kernel position (optional)

    Notes:
      - This layer is depthwise (no channel mixing), but spatially adaptive via depth.
      - If you want channel mixing afterwards, add a 1x1 conv after this layer.
    """

    def __init__(
        self,
        channels: int = 128,
        kernel_size: int = 3,
        sigma: float = 0.1,
        learnable_spatial: bool = True,
        shared_spatial: bool = False,
        eps: float = 1e-6,
        depth_pad_val: float = None,   # if not None, treat this value as invalid and ignore neighbors
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.C = int(channels)
        self.ks = int(kernel_size)
        self.pad = self.ks // 2
        self.sigma = float(sigma)
        self.eps = float(eps)
        self.learnable_spatial = bool(learnable_spatial)
        self.shared_spatial = bool(shared_spatial)
        self.depth_pad_val = depth_pad_val

        P = self.ks * self.ks

        if self.learnable_spatial:
            if self.shared_spatial:
                # One kernel shared across all channels: (P,)
                self.spatial = nn.Parameter(torch.ones(P))
            else:
                # One kernel per channel: (C,P)
                self.spatial = nn.Parameter(torch.ones(self.C, P))
        else:
            self.spatial = None

        # Nice init: center slightly higher helps stability
        # (only if learnable spatial enabled)
        if self.spatial is not None:
            with torch.no_grad():
                if self.shared_spatial:
                    self.spatial.fill_(1.0)
                    self.spatial[P // 2] = 1.25
                else:
                    self.spatial.fill_(1.0)
                    self.spatial[:, P // 2] = 1.25

    def _get_ws(self, wd: torch.Tensor) -> torch.Tensor:
        """
        Build spatial weights tensor to match wd's dtype/device.

        wd: (N,P,HW)
        returns:
          if shared_spatial: (1,1,P,1)
          else (per-channel): (1,C,P,1)
        """
        P = self.ks * self.ks
        if self.spatial is None:
            return None

        if self.shared_spatial:
            ws = self.spatial.view(1, 1, P, 1).to(device=wd.device, dtype=wd.dtype)
        else:
            ws = self.spatial.view(1, self.C, P, 1).to(device=wd.device, dtype=wd.dtype)
        return ws

    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        x:     (N,C,H,W)
        depth: (N,1,H,W) or (N,H,W) aligned to x spatially
        """
        assert x.dim() == 4, f"x must be (N,C,H,W), got {x.shape}"
        N, C, H, W = x.shape
        assert C == self.C, f"Expected C={self.C}, got {C}"

        if depth.dim() == 3:
            depth = depth.unsqueeze(1)  # (N,1,H,W)
        assert depth.dim() == 4 and depth.size(1) == 1, f"depth must be (N,1,H,W), got {depth.shape}"
        assert depth.shape[0] == N, "Batch size mismatch"
        assert depth.shape[-2:] == (H, W), (
            f"Depth must match x spatial size. depth={depth.shape[-2:]} vs x={(H,W)}. "
            "Resize depth to feature resolution before calling."
        )

        P = self.ks * self.ks
        HW = H * W

        # --- Unfold x to patches: (N, C*P, HW) -> (N,C,P,HW)
        x_patch = F.unfold(x, kernel_size=self.ks, padding=self.pad)  # (N, C*P, HW)
        x_patch = x_patch.view(N, C, P, HW)

        # --- Unfold depth to patches: (N, P, HW)
        depth_f = depth.float()
        d_patch = F.unfold(depth_f, kernel_size=self.ks, padding=self.pad)  # (N, P, HW)
        d_center = depth_f.view(N, 1, HW)                                   # (N, 1, HW)

        # --- Depth similarity weights wd: (N,P,HW)
        diff = d_patch - d_center
        wd = torch.exp(-(diff * diff) / (2.0 * (self.sigma ** 2) + self.eps))  # (N,P,HW)

        # Optional: ignore invalid/padded depth neighbors if you padded depth with a special value
        if self.depth_pad_val is not None:
            invalid_n = (d_patch == float(self.depth_pad_val))
            invalid_c = (d_center == float(self.depth_pad_val))
            invalid = invalid_n | invalid_c
            wd = wd.masked_fill(invalid, 0.0)

        # --- Expand depth weights to channels: (N,1,P,HW)
        wC = wd.unsqueeze(1)

        # --- Apply spatial weights (shared or per-channel)
        ws = self._get_ws(wd)
        if ws is not None:
            # shared: ws (1,1,P,1) broadcasts across C
            # per-ch: ws (1,C,P,1) matches channels
            wC = wC * ws

        # --- Weighted sum & normalize
        num = (x_patch * wC).sum(dim=2)                      # (N,C,HW)
        den = wC.sum(dim=2).clamp_min(self.eps)              # (N,C,HW)
        out = (num / den).view(N, C, H, W)                   # (N,C,H,W)

        return out




class DepthGuidedRefine2x(nn.Module):
    def __init__(self, channels=128, kernel_size=3, sigma=0.1):
        super().__init__()
        self.dg1 = DepthSimilarityWeightedConv(channels, kernel_size, sigma)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.ReLU(inplace=True)

        self.dg2 = DepthSimilarityWeightedConv(channels, kernel_size, sigma)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.ReLU(inplace=True)

        # optional learnable residual scale (starts small)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, depth):
       #identity = x
        x = self.act1(self.bn1(self.dg1(x, depth)))
        x = self.act2(self.bn2(self.dg2(x, depth)))
        return self.res_scale * x


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
        """
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
        self.mix_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=256,   # input feature maps
                out_channels=128,  # output feature maps
                kernel_size=3,
                padding=2,         # keeps spatial size unchanged
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        
        
    def forward(self, x_d, x_s,x_my):
        #detail_dwconv = self.detail_dwconv(x_d) # my comment
        detail_dwconv = self.detail_dwconv(x_d) # my comment
       #detail_down = self.detail_down(x_d)
        semantic_conv = self.semantic_conv(x_s)
        semantic_dwconv = self.semantic_dwconv(x_s) #my comment
        #etail_dwconv_= self.detail_dwconv(x_my)# my comment
       #detail_down_= self.detail_down(x_my)
        #semantic_dwconv = self.semantic_dwconv(x_s) 
        rgb_dwconv = self.semantic_dwconv_rgb(x_my) 
        
        #dil = torch.nn.functional.max_pool2d(rgb_dwconv, kernel_size=5, stride=1, padding=5 // 2)
        #rgb_dwconv=rgb_dwconv+self.thicker(dil)
        semantic_conv = resize(
            input=semantic_conv,
            size=detail_dwconv.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        #fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv) # my comment
        #fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv) # my comment
        #####fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv) #
        fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv)
        ##fuse_2 = detail_down * torch.sigmoid(semantic_dwconv) #my comment
        ###fuse_2 = semantic_dwconv #my comment
        #$$fuse_2 = semantic_dwconv * torch.sigmoid(detail_down)#my comment
        fuse_2 =torch.sigmoid(semantic_dwconv)+torch.sigmoid(rgb_dwconv)
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
        
        fuse_2 = resize(
            input=fuse_2,
            size=fuse_1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        """
        fuse_3 = resize(
            input=fuse_3,
            size=fuse_1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        """
        #output = self.conv(fuse_1 + fuse_2)
        x = torch.cat([fuse_1,fuse_2], dim=1)              # [B, C+C2, H, W]
        output = self.mix_conv(x)                            # nn.Conv2d(C+C2, C_out, kernel_size=1)
        return output


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
                 out_indices=(0, 1, 2, 3, 4),
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
        
        #my code-->
        self.detail = DetailBranch(self.detail_channels, self.in_channels//2)
        self.semantic = SemanticBranch(self.semantic_channels,
                                       self.in_channels//2,
                                       self.semantic_expansion_ratio)
        
        self.semantic_rgb = SemanticBranch(self.semantic_channels,
                                       self.in_channels//2,
                                       self.semantic_expansion_ratio)
        
        #my code^^^
        '''
        self.detail = DetailBranch(self.detail_channels, self.in_channels)
        self.semantic = SemanticBranch(self.semantic_channels,
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
        self.depth_refine = DepthGuidedRefine2x(channels=128, kernel_size=3, sigma=0.1)


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
  
        x_semantic_input_depth = x[:,3: , :, :] 
        x_semantic_input_rgb =x[:, :3, :, :]
        x_depth_raw=x[:,3:4, :, :] 
        
        
        x_detail = self.detail(x_detail_input)
       #x_detail_= self.detail(x_detail_rgb)
        x_semantic_lst_1 = self.semantic(x_semantic_input_depth)
        x_semantic_lst_2 = self.semantic_rgb(x_semantic_input_rgb)
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
        x_sem = self.depth_refine(x_semantic_lst_2[-1],depth_sem)






        x_head = self.bga(x_detail,x_semantic_lst_1[-1],x_sem)
        #x_head = self.bga(x_detail, x_semantic_lst_1[-1], x_semantic_lst_2[-1])
        #outs = [x_head] + x_semantic_lst_2[2:4]+x_semantic_lst_1[2:4] #conv_5
        #outs = [x_head] + x_semantic_lst_2[1:3]+ x_semantic_lst_1[2:4]
        #outs = [x_head] + x_semantic_lst_2[:-1]+x_semantic_lst_1[:-1] #conv_4

        #""" latest
        #x_fused = x_head + x_detail  #my code  
        #outs = [x_fused] + x_semantic_lst[:-1] #my comment#
        #x_head = self.bga(x_detail, x_semantic_lst[-1])
        outs = [x_head] + x_semantic_lst_2[1:3]+x_semantic_lst_1[2:4]
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

