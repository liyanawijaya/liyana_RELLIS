from numbers import Number
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F  # <-- NEW
from mmengine.model import BaseDataPreprocessor
from mmseg.registry import MODELS
from mmseg.utils import stack_batch


@MODELS.register_module()
class SegDataPreProcessorWithRawDepth(BaseDataPreprocessor):
    """Same as SegDataPreProcessor but also stores raw depth channel in metainfo,
    aligned to the stacked/padded batch resolution.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = -1,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
        depth_channel_idx: int = 5,
        depth_pad_val: float = 0.0,   # <-- NEW: what to pad depth with
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.depth_channel_idx = depth_channel_idx
        self.depth_pad_val = float(depth_pad_val)

        assert not (bgr_to_rgb and rgb_to_bgr)
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None
            self._enable_normalize = True
            self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std', torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        self.batch_augments = batch_augments
        self.test_cfg = test_cfg

    def _align_depth_list_to(self, depth_list, Ht, Wt):
        """Pad/crop each (H,W) depth tensor to (Ht,Wt) using bottom/right padding."""
        aligned = []
        for d in depth_list:
            h, w = d.shape

            # crop if larger
            d = d[:Ht, :Wt]

            # pad if smaller
            pad_h = Ht - d.shape[0]
            pad_w = Wt - d.shape[1]
            if pad_h > 0 or pad_w > 0:
                d = F.pad(d, (0, pad_w, 0, pad_h), mode='constant', value=self.depth_pad_val)

            aligned.append(d)
        return aligned

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        data = self.cast_data(data)
        inputs = data['inputs']              # list[Tensor(C,H,W)]
        data_samples = data.get('data_samples', None)

        # ----- Capture RAW depth BEFORE any conversion/normalisation -----
        raw_depth_list = []
        for _inp in inputs:
            if _inp.size(0) <= self.depth_channel_idx:
                raise ValueError(
                    f'Expected >= {self.depth_channel_idx+1} channels, got {_inp.size(0)}')
            raw_depth_list.append(_inp[self.depth_channel_idx].clone())  # (H,W)

        # Optional channel conversion (only for pure RGB inputs; for multi-channel it's usually False anyway)
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]

        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            # ---- NEW: align raw depth to stacked batch resolution ----
            Ht, Wt = inputs.shape[-2], inputs.shape[-1]  # e.g., 600,600
            raw_depth_list = self._align_depth_list_to(raw_depth_list, Ht, Wt)

        else:
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs)

            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)

                # align raw depth to test-time padded size too
                Ht, Wt = inputs.shape[-2], inputs.shape[-1]
                raw_depth_list = self._align_depth_list_to(raw_depth_list, Ht, Wt)

                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                # no padding in test, but still ensure uniform stack
                inputs = torch.stack(inputs, dim=0)
                Ht, Wt = inputs.shape[-2], inputs.shape[-1]
                raw_depth_list = self._align_depth_list_to(raw_depth_list, Ht, Wt)

        # ----- Attach aligned raw depth to metainfo -----
        if data_samples is not None:
            for ds, d in zip(data_samples, raw_depth_list):
                ds.set_metainfo({'depth_map_raw': d})

        return dict(inputs=inputs, data_samples=data_samples)
