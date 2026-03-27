from mmseg.models import EncoderDecoder
from mmseg.registry import MODELS
from mmengine.structures import PixelData
'''
@MODELS.register_module()
class EncoderDecoderWithValLoss(EncoderDecoder):
    def val_step(self, data):
        data = self.data_preprocessor(data, False)
        inputs = data['inputs']
        data_samples = data['data_samples']
        losses = self.loss(inputs, data_samples)

        # Attach losses to each data sample (optional)
        for sample in data_samples:
            sample.losses = losses

        return data_samples  # processed by evaluator/hook
        '''
from mmseg.structures import SegDataSample  # confirmed
'''
@MODELS.register_module()
class EncoderDecoderWithValLoss(EncoderDecoder):
    def val_step(self, data):
        data = self.data_preprocessor(data, False)
        inputs = data['inputs']
        data_samples = data['data_samples']

        losses = self.loss(inputs, data_samples)
        loss_dict = {k: v.item() if hasattr(v, 'item') else v for k, v in losses.items()}

        for sample in data_samples:
            assert isinstance(sample, SegDataSample)
            # Store metrics using set_metainfo
            sample.set_metainfo({'metrics': loss_dict})
            #print(f"[DEBUG] Sample types: {[type(s) for s in data_samples]}")
        return data_samples
'''
'''
@MODELS.register_module()
class EncoderDecoderWithValLoss(EncoderDecoder):
    def val_step(self, data):
        data = self.data_preprocessor(data, False)
        inputs = data['inputs']
        data_samples = data['data_samples']
        batch_img_metas = [sample.metainfo for sample in data_samples]

        # Compute losses
        losses = self.loss(inputs, data_samples)
        loss_dict = {k: v.item() if hasattr(v, 'item') else v for k, v in losses.items()}

        # Compute segmentation predictions using inherited `inference()` logic
        seg_logits = self.inference(inputs, batch_img_metas)
        pred_labels = seg_logits.argmax(dim=1)

        for sample, pred in zip(data_samples, pred_labels):
            sample.pred_sem_seg = PixelData(data=pred.unsqueeze(0))  # unsqueeze to [1, H, W]
            sample.set_metainfo({'metrics': loss_dict})  # For your CustomLossMetric

        return data_samples
'''

@MODELS.register_module()
class EncoderDecoderWithValLoss(EncoderDecoder):
    def val_step(self, data):
        # Preprocess inputs
        data = self.data_preprocessor(data, False)
        inputs = data['inputs']
        data_samples = data['data_samples']
        batch_img_metas = [sample.metainfo for sample in data_samples]

        # Compute losses
        losses = self.loss(inputs, data_samples)
        loss_dict = {k: v.item() if hasattr(v, 'item') else v for k, v in losses.items()}

        # Compute raw segmentation logits
        seg_logits = self.inference(inputs, batch_img_metas)

        # Use official postprocessing to produce final predictions
        pred_samples = self.postprocess_result(seg_logits, data_samples)

        # Attach loss metrics to the corresponding sample
        for pred, sample in zip(pred_samples, data_samples):
            assert isinstance(sample, SegDataSample)
            pred.set_metainfo({'metrics': loss_dict})

        return pred_samples

