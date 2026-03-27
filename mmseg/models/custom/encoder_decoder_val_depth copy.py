import torch
from mmseg.models import EncoderDecoder
from mmseg.structures import SegDataSample
from mmseg.registry import MODELS
@MODELS.register_module()
class EncoderDecoderWithValLoss(EncoderDecoder):

    def loss(self, inputs, data_samples, **kwargs):
        """
        inputs: Tensor (N,C,H,W) from data_preprocessor
        data_samples: list[SegDataSample]
        """

        # --- 1) Extract depth channel (assumes depth is channel index 3) ---
        if not isinstance(inputs, torch.Tensor) or inputs.dim() != 4:
            raise TypeError(f"Expected inputs as (N,C,H,W) tensor, got {type(inputs)} with shape {getattr(inputs, 'shape', None)}")

        if inputs.size(1) < 4:
            raise ValueError(f"Expected >=4 channels (RGB+Depth). Got C={inputs.size(1)}")

        x_depth = inputs[:, 5:6, :, :]   # (N,1,H,W)

        # --- 2) Attach depth map to each sample's metainfo ---
        if len(data_samples) != x_depth.size(0):
            raise ValueError(f"Batch size mismatch: depth N={x_depth.size(0)} vs data_samples={len(data_samples)}")

        for i, s in enumerate(data_samples):
            # ensure it's a SegDataSample (not strictly required, but safer)
            if not isinstance(s, SegDataSample):
                raise TypeError(f"data_samples[{i}] is not SegDataSample, got {type(s)}")

            # Store (H,W) tensor. Keep it on GPU, no detach needed.
            s.set_metainfo({'depth_map': x_depth[i, 0]})

        # --- 3) Run normal MMSeg loss (decode_head / aux_head will now see depth_map) ---
        return super().loss(inputs, data_samples, **kwargs)

    def val_step(self, data):
        # Your existing val_step (unchanged)
        data = self.data_preprocessor(data, False)
        inputs = data['inputs']
        data_samples = data['data_samples']
        batch_img_metas = [sample.metainfo for sample in data_samples]

        losses = self.loss(inputs, data_samples)
        loss_dict = {k: v.item() if hasattr(v, 'item') else v for k, v in losses.items()}

        seg_logits = self.inference(inputs, batch_img_metas)
        pred_samples = self.postprocess_result(seg_logits, data_samples)

        for pred in pred_samples:
            pred.set_metainfo({'metrics': loss_dict})

        return pred_samples
