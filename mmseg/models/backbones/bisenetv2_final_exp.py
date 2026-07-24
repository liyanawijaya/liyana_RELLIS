# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      build_activation_layer, build_norm_layer)
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from ..utils import resize


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

class SimpleAttention(nn.Module):
    def __init__(self, channels=128, reduction=16):
        super().__init__()

        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        x = x * self.channel_att(x)

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)

        x = x * self.spatial_att(spatial)

        return x   
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), "kernel_size should be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.spatial_att = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=1,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [B, C, H, W]

        avg_out = torch.mean(x, dim=1, keepdim=True)      # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)    # [B, 1, H, W]

        spatial = torch.cat([avg_out, max_out], dim=1)    # [B, 2, H, W]

        att_map = self.spatial_att(spatial)               # [B, 1, H, W]

        x = x * att_map                                   # [B, C, H, W]

        return x   



class RGBGuidedLiDARAttention(nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, lidar):
        # rgb   : B × 128 × H × W
        # lidar : B × 128 × H × W

        x = torch.cat([lidar, rgb], dim=1)      # B × 256 × H × W

        lidar_att = self.attention(x)           # B × 128 × H × W

        lidar_guided = rgb * lidar_att        # B × 128 × H × W

        fused = torch.cat([lidar_guided,rgb], dim=1)  # B × 256 × H × W

        out = self.fuse(fused)                  # B × 128 × H × W

        return out


class AttentionFusion128(nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        self.att = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.out = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat1, feat2):
        # feat1: B x 128 x H x W
        # feat2: B x 128 x H x W

        x = torch.cat([feat1, feat2], dim=1)   # B x 256 x H x W

        att = self.att(x)                      # B x 128 x H x W

        feat2_weighted = feat2 * att           # B x 128 x H x W

        fused = torch.cat([feat1, feat2_weighted], dim=1)

        out = self.out(fused)                  # B x 128 x H x W

        return out

import torch.nn.functional as F

class CommonFeatureModule(nn.Module):
    def __init__(self, channels=128, common_channels=128):
        super().__init__()

        self.rgb_proj = nn.Sequential(
            nn.Conv2d(channels, common_channels, kernel_size=1),
            nn.BatchNorm2d(common_channels),
            nn.ReLU(inplace=True)
        )

        self.lidar_proj = nn.Sequential(
            nn.Conv2d(channels, common_channels, kernel_size=1),
            nn.BatchNorm2d(common_channels),
            nn.ReLU(inplace=True)
        )

        self.common_fuse = nn.Sequential(
            nn.Conv2d(common_channels * 2, common_channels, kernel_size=1),
            nn.BatchNorm2d(common_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_feat, lidar_feat):
        rgb_coarse = F.avg_pool2d(
            rgb_feat,
            kernel_size=5,
            stride=1,
            padding=2
        )

        rgb_proj = self.rgb_proj(rgb_coarse)
        lidar_proj = self.lidar_proj(lidar_feat)

        #common = self.common_fuse(
        #    torch.cat([rgb_proj, lidar_proj], dim=1)
        #)
        rgb_norm = F.normalize(rgb_coarse, dim=1)
        lidar_norm = F.normalize(lidar_feat, dim=1)
        common_1=lidar_norm*torch.sigmoid(rgb_norm)
        common_2=torch.sigmoid(lidar_norm)*rgb_norm
        
        return common_1,common_2
"""
class LidarGuidedFusion(nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        self.lidar_attention = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_feat, lidar_feat):
        attention = self.lidar_attention(lidar_feat)
        lidar_weighted = lidar_feat * attention
        rgb_coarse = F.avg_pool2d(
            rgb_feat,
            kernel_size=5,
            stride=1,
            padding=2
        )

        
        fused = self.fusion(
            torch.cat([rgb_coarse, lidar_weighted], dim=1)
        )

        return  attention, fused

"""
class LidarGuidedFusion(nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        # Generate one spatial weight map for each RGB feature channel
        # using the LiDAR feature tensor
        self.lidar_attention = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # Fuse LiDAR-guided RGB features with the original LiDAR features
        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * 2,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_feat, lidar_feat):
        """
        Args:
            rgb_feat:   [B, 128, H, W]
            lidar_feat: [B, 128, H, W]

        Returns:
            attention: [B, 128, H, W]
            fused:     [B, 128, H, W]
        """

        if rgb_feat.shape != lidar_feat.shape:
            raise ValueError(
                f'RGB and LiDAR features must have the same shape, '
                f'but received RGB {rgb_feat.shape} and '
                f'LiDAR {lidar_feat.shape}.'
            )

        # LiDAR generates 128 channel-specific attention maps
        attention = self.lidar_attention(lidar_feat)

        # Optionally reduce fine RGB texture
        rgb_coarse = F.avg_pool2d(
            rgb_feat,
            kernel_size=5,
            stride=1,
            padding=2
        )

        # LiDAR attention identifies important RGB locations/features
        rgb_weighted = rgb_feat * attention

        # Combine LiDAR-guided RGB features with LiDAR features
        fusion_input = torch.cat(
            [rgb_weighted, lidar_feat],
            dim=1
        )

        fused = self.fusion(fusion_input)

        return rgb_weighted

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
        self.rgb_bn = nn.BatchNorm2d(128)
        self.depth_bn = nn.BatchNorm2d(128)
        self.detail_dwconv_depth = nn.Sequential(
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
        self.detail_down_depth = nn.Sequential(
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
        self.semantic_conv_depth = nn.Sequential(
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
        
      
        self.semantic_dwconv_depth = nn.Sequential(
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
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        self.mix_conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        self.mix_conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128*2,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        self.mix_conv_att = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SpatialAttention(kernel_size=7)
        )
        self.fusion_layer = RGBGuidedLiDARAttention(channels=128)
        self.fusion = AttentionFusion128(channels=128)
        
        self.common = CommonFeatureModule(
            channels=128,
            common_channels=128
        )
        self.lidar_guided= LidarGuidedFusion(channels=128)
        
    def forward(self, x_d, x_depth, x_s, x_my):
        #detail_dwconv = self.detail_dwconv(x_d) # my comment

        detail_dwconv = self.detail_dwconv(x_d) # my comment
        detail_dwconv_depth= self.detail_dwconv_depth(x_depth)
        detail_down = self.detail_down(x_d)
        semantic_conv = self.semantic_conv(x_s)
        semantic_dwconv = self.semantic_dwconv(x_s) #my comment
        detail_down_depth=self.detail_down_depth(x_depth)
        semantic_dwconv_depth=self.semantic_dwconv_depth(x_my)
        #semantic_dwconv = self.semantic_dwconv(x_s) 
        depth_conv = self.semantic_conv_depth(x_my) 
        
        #dil = torch.nn.functional.max_pool2d(rgb_dwconv, kernel_size=5, stride=1, padding=5 // 2)
        #rgb_dwconv=rgb_dwconv+self.thicker(dil)
        semantic_conv = resize(
            input=semantic_conv,
            size=detail_dwconv.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        depth_conv = resize(
            input=depth_conv,
            size=detail_dwconv_depth.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        #fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv) # my comment
        #fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv) # my comment
        #####fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv) #
        #final fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv) #
        #final fuse_3 = detail_down * torch.sigmoid(semantic_dwconv) #my comment
        ###fuse_2 = semantic_dwconv #my comment
        #$$fuse_2 = semantic_dwconv * torch.sigmoid(detail_down)#my comment
        #final fuse_2 =(detail_dwconv_depth)*torch.sigmoid(depth_conv)
        #final fuse_4 = detail_down_depth * torch.sigmoid(semantic_dwconv_depth)
        
        fuse_1 = detail_dwconv * torch.sigmoid(semantic_conv) #
        fuse_2 = detail_down * torch.sigmoid(semantic_dwconv) #my comment
        ###fuse_2 = semantic_dwconv #my comment
        #$$fuse_2 = semantic_dwconv * torch.sigmoid(detail_down)#my comment
        fuse_3 =(detail_dwconv_depth)*torch.sigmoid(depth_conv)
        fuse_4 = detail_down_depth * torch.sigmoid(semantic_dwconv_depth)
        
        
        
        """
        rgb_sem=semantic_dwconv
        depth_sem=semantic_dwconv_depth
        output_3 = torch.cat([(torch.sigmoid(semantic_dwconv)),(torch.sigmoid(semantic_dwconv_depth))], dim=1) 
        output_3=self.mix_conv_2(output_3)
        output_com=torch.sigmoid(rgb_sem)*depth_sem
        
        #rgb_weighted = self.lidar_guided(torch.sigmoid(rgb_sem), torch.sigmoid(depth_sem))
        
        rgb_sem_conv = resize(
            input=output_3,
            size=detail_dwconv.shape[2:],
           mode='bilinear',
            align_corners=self.align_corners)
        
        rgb_sem_dwconv = resize(
            input=output_3,
            size=detail_down.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        rgb_1 = detail_dwconv * torch.sigmoid(rgb_sem_conv) #
        rgb_2 = detail_down * torch.sigmoid(rgb_sem_dwconv) #my comment

        rgb_2_up = resize( #new
            input=rgb_2,
            size=rgb_1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output_4 = torch.cat([(rgb_1),(rgb_2_up)], dim=1) 
        """
        #output_3 = torch.cat([((rgb_1)),(rgb_2_up)], dim=1) 
        #output_3=self.mix_conv_2(output_3)
        #fuse_1 = detail_dwconv  # new
        #fuse_2 = semantic_dwconv #new
        ###fuse_2 = semantic_dwconv #my comment
        #$$fuse_2 = semantic_dwconv * torch.sigmoid(detail_down)#my comment
        #fuse_3 =detail_dwconv_depth #new
        #fuse_4 = semantic_dwconv_depth #new
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
        
        fuse_2_up = resize(
            input=fuse_2,
            size=fuse_1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        #fuse_3 = resize(
        #    input=fuse_3,
        #    size=fuse_1.shape[2:],
        #    mode='bilinear',
        #    align_corners=self.align_corners)
        fuse_4_up = resize(
            input=fuse_4,
            size=fuse_1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        #fuse_1 = self.rgb_bn(fuse_1)
        #fuse_2 = self.depth_bn(fuse_2)
        #output = self.conv(fuse_1 + fuse_2)
        ##x = torch.cat([torch.sigmoid(fuse_1),torch.sigmoid(fuse_2)], dim=1)              # [B, C+C2, H, W]
        ##output_1 = fuse_1                            # nn.Conv2d(C+C2, C_out, kernel_size=1)
        x = torch.cat([(fuse_1),(fuse_2_up)], dim=1) 
        x_1 = torch.cat([(fuse_3),(fuse_4_up)], dim=1) 
        #output_2=fuse_2_up*fuse_4_up
   
        output_1 = self.mix_conv(x) #detail resolution
        output_2 = self.mix_conv_1(x_1) #semantic resolution
        #output_1=torch.sigmoid(fuse_1)*fuse_3
        #output_2=torch.sigmoid(fuse_2)*fuse_4
        
        #output_1_new=fuse_1+fuse_2_up #new
        #output_2_new=fuse_3+fuse_4_up #new 
        #output_1 = fuse_1+fuse_3 #detail resolution
        #output_2 = fuse_2+fuse_4 #semantic resolution
        output_2_up = resize( #new
            input=output_2,
            size=output_1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        #fuse_3 = resize(
        #    input=fuse_3,
        #    size=fuse_1.shape[2:],
        #    mode='bilinear',
        #    align_corners=self.align_corners)
        output_1_down = resize( #new
            input=output_1,
            size=output_2.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        
        output_2_resized = resize( #new
            input=output_2,
            size=output_1.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
                
        #common_1, common_2 = self.common(
        #    output_1,
        #    output_2_up
        #)
        
        #attention, fused = self.lidar_guided(torch.sigmoid(output_1), torch.sigmoid(output_2_up))

        #output_com = torch.cat([(torch.sigmoid(output_1)),(torch.sigmoid(output_2_resized))], dim=1) #new
        #output_com = self.mix_conv_2(output_com)
        #output_com=torch.sigmoid(output_1)*torch.sigmoid(output_2_up) 
        #final output_com=torch.sigmoid(fuse_1)*torch.sigmoid(fuse_3)
        #output_com=torch.sigmoid(output_1*output_2_up)
        #output_com = torch.cat([(torch.sigmoid(output_1)),(torch.sigmoid(output_2_up))], dim=1) #new
        #output_1=output_1*torch.sigmoid(output_2_up) #new
        #output_2=output_1_down*torch.sigmoid(output_2) #new

        #output_com=torch.sigmoid(output_1)
        output_com=torch.sigmoid(output_1*output_2_up)
        #output_3 = torch.cat([(torch.sigmoid(output_1)+output_com),torch.sigmoid(output_2_up)+output_com], dim=1) #new
        output_3=torch.sigmoid(output_1)+torch.sigmoid(output_2_up)+output_com
        #finaloutput_3 = torch.cat([(torch.sigmoid(output_1)*output_2_up),(torch.sigmoid(output_2_up)*output_1)], dim=1) #new
        #output_3 =self.mix_conv_2(torch.sigmoid(output_1)+common) #new
        #output_3 = torch.cat([(torch.sigmoid(output_1)),torch.sigmoid(fused)], dim=1)
        #output_3 = torch.cat([(torch.sigmoid(fuse_2)),torch.sigmoid(fuse_4)], dim=1)
        #output_3 =self.mix_conv_2(output_3)
        #output_3 = torch.cat([(torch.sigmoid(output_1)),torch.sigmoid(output_2_up)], dim=1)
        ####output_com= torch.sigmoid(output_1*output_2_up)
        ####output_3 = torch.cat([(torch.sigmoid(output_1)+output_com),torch.sigmoid(output_2_up)+output_com], dim=1)
        #output_3=torch.sigmoid(output_1)+torch.sigmoid(output_2_up)+output_com
        ####output_3=self.mix_conv_2(output_3)
        #output_3=output_1+torch.sigmoid(output_1)*torch.sigmoid(output_2)
        #lastoutput_3=torch.sigmoid(output_1)+torch.sigmoid(output_2)+torch.sigmoid(output_1)*torch.sigmoid(output_2)
        #finaloutput_3=torch.sigmoid(output_1)+torch.sigmoid(output_2)+torch.sigmoid(output_1)*torch.sigmoid(output_2)
        #output_3=torch.sigmoid(output_1)*torch.sigmoid(output_2)
        #output_3=output_1+torch.sigmoid(output_1)*output_2
        #output_3=output_1
        #output_3 = torch.cat([(output_1),(output_3)], dim=1) 
        #output_3 = torch.cat([(torch.sigmoid(output_1)), torch.sigmoid((output_2))], dim=1)
        #output_3=output_1+self.mix_conv_2(output_3)
        #output_3 = self.fusion_layer(output_1, output_3)

        #output_3=self.mix_conv_att(output_3)
        #output_3 = self.fusion(output_3, output_1)
        #output_3=self.mix_conv_2(output_3)
        #lastreturn output_2, output_3
        #return torch.sigmoid(fuse_2_up)*torch.sigmoid(fuse_4_up), output_3, output_com
        #return output_2_up, output_3, output_com
        return output_3, output_com

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
                 #lastout_indices=(0, 1, 2, 3, 4, 5, 6),
                 out_indices=(0, 1, 2, 3, 4, 5),
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
        self.detail_depth = DetailBranch(self.detail_channels, self.in_channels//2)
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
        x_detail_input=x[:, :3,:, :]
        x_detail_depth=x[:, 3:, :]
        #x_detail_depth=x[:, 3:,:, :]
  
        x_semantic_input_depth = x[:,3: , :, :] 
        x_semantic_input_rgb =  x[:, :3,:, :] 
        
        
        x_detail = self.detail(x_detail_input)
        x_depth =self.detail_depth(x_detail_depth)
        x_semantic_lst_1 = self.semantic(x_semantic_input_rgb)
        x_semantic_lst_2 = self.semantic_rgb(x_semantic_input_depth)
        #x_semantic_lst_2 = self.semantic_rgb(x_semantic_input_rgb)
        #x_cat = torch.cat([x_semantic_lst_2[-1], x_semantic_lst_1[-1]], dim=1)   #conv_4   # (N, 2C, H, W)
        #x_semantic_lst = self.fuser(x_cat) #conv_4
        #x_semantic_lst=x_semantic_lst_1[-1]*x_semantic_lst_1[-1]
        """
        import torch.nn.functional as F

        x_detail = F.interpolate(
            x_detail,
            size=x_semantic_lst_2[3].shape[2:],
            mode='bilinear',
            align_corners=False
        )
        """
        #x_head = self.bga(x_detail, x_semantic_lst_1[-1]) #my comment
        #x_head_2, 
        #finalx_head_2, x_head_3, x_head_com= self.bga(x_detail, x_depth, x_semantic_lst_1[-1], x_semantic_lst_2[-1])
        output_3, output_com= self.bga(x_detail, x_depth, x_semantic_lst_1[-1], x_semantic_lst_2[-1])
        #outs = [x_head] + x_semantic_lst_2[2:4]+x_semantic_lst_1[2:4] #conv_5
        #final outs = [x_head_3]+[x_head_2] +[x_head_com] + x_semantic_lst_1[1:3] + x_semantic_lst_2[2:4]
        outs = [output_3] + [output_com]+ x_semantic_lst_1[1:3] + x_semantic_lst_2[2:4]
        #outs = [x_head] + x_semantic_lst_2[:-1]+x_semantic_lst_1[:-1] #conv_4

        #""" latest
    
        #x_fused = x_head + x_detail  #my code  
        #outs = [x_fused] + x_semantic_lst[:-1] #my comment#
        #x_head = self.bga(x_detail, x_semantic_lst[-1])
        #outs = [x_head] + x_semantic_lst[:-1]
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

