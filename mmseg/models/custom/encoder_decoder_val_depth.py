import torch
from mmseg.models import EncoderDecoder
from mmseg.structures import SegDataSample
from mmseg.registry import MODELS

@MODELS.register_module()
class EncoderDecoderWithValLoss(EncoderDecoder):

    def loss(self, inputs, data_samples, **kwargs):
        """
        inputs: Tensor (N,C,H,W) from data_preprocessor (may be normalized)
        data_samples: list[SegDataSample] with metainfo containing raw depth
                      (e.g., 'depth_map_raw') injected by the custom data_preprocessor
        """

        if data_samples is None:
            raise ValueError("data_samples is None. Raw depth must be passed via data_samples.metainfo.")

        # Attach RAW depth to the key expected by your DepthAwareFCNHead/loss: 'depth_map'
        for i, s in enumerate(data_samples):
            if not isinstance(s, SegDataSample):
                raise TypeError(f"data_samples[{i}] is not SegDataSample, got {type(s)}")

            if 'depth_map_raw' not in s.metainfo:
                raise KeyError(
                    "depth_map_raw not found in data_samples.metainfo. "
                    "Use SegDataPreProcessorWithRawDepth (or similar) to store raw depth."
                )

            # raw depth is typically (H,W); keep it as-is
            s.set_metainfo({'depth_map': s.metainfo['depth_map_raw']})

        # Run normal MMSeg loss (decode_head / aux_head can read depth_map now)
        return super().loss(inputs, data_samples, **kwargs)

    def val_step(self, data):
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
