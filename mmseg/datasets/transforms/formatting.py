# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample

@TRANSFORMS.register_module()
# Copyright (c) OpenMMLab. All rights reserved.
class PackSegInputs(BaseTransform):

    def __init__(self,
                 meta_keys=('img_path',
                            'seg_map_path',
                            'seg_map_path_aux',
                            'seg_map_path_aux2',   # ✅ added
                            'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'flip_direction',
                            'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()

        # -------------------- INPUT --------------------
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)

            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = to_tensor(img.transpose(2, 0, 1)).contiguous()

            packed_results['inputs'] = img

        data_sample = SegDataSample()

        # -------------------- MAIN GT --------------------
        if 'gt_seg_map' in results:
            gt = results['gt_seg_map']

            if gt.ndim == 2:
                data = to_tensor(gt[None, ...].astype(np.int64))
            else:
                warnings.warn(f'gt_seg_map expected 2D but got {gt.shape}')
                data = to_tensor(gt.astype(np.int64))

            data_sample.gt_sem_seg = PixelData(data=data)

        # -------------------- AUX GT 1 --------------------
        if 'gt_seg_map_aux' in results:
            gt_aux = results['gt_seg_map_aux']

            if gt_aux.ndim == 2:
                data_aux = to_tensor(gt_aux[None, ...].astype(np.int64))
            else:
                warnings.warn(f'gt_seg_map_aux expected 2D but got {gt_aux.shape}')
                data_aux = to_tensor(gt_aux.astype(np.int64))

            data_sample.set_data(dict(
                gt_sem_seg_aux=PixelData(data=data_aux)
            ))

        # -------------------- AUX GT 2 ✅ --------------------
        if 'gt_seg_map_aux2' in results:
            gt_aux2 = results['gt_seg_map_aux2']

            if gt_aux2.ndim == 2:
                data_aux2 = to_tensor(gt_aux2[None, ...].astype(np.int64))
            else:
                warnings.warn(f'gt_seg_map_aux2 expected 2D but got {gt_aux2.shape}')
                data_aux2 = to_tensor(gt_aux2.astype(np.int64))

            data_sample.set_data(dict(
                gt_sem_seg_aux2=PixelData(data=data_aux2)
            ))

        # -------------------- EDGE / DEPTH --------------------
        if 'gt_edge_map' in results:
            data_sample.set_data(dict(
                gt_edge_map=PixelData(
                    data=to_tensor(results['gt_edge_map'][None, ...].astype(np.int64))
                )
            ))

        if 'gt_depth_map' in results:
            data_sample.set_data(dict(
                gt_depth_map=PixelData(
                    data=to_tensor(results['gt_depth_map'][None, ...])
                )
            ))

        # -------------------- META --------------------
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)

        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(meta_keys={self.meta_keys})'
'''
class PackSegInputs(BaseTransform):

    def __init__(self,
                 meta_keys=('img_path',
                            'seg_map_path',
                            'seg_map_path_aux',   # ✅ keep
                            'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'flip_direction',
                            'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()

        # -------------------- INPUT --------------------
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = to_tensor(img.transpose(2, 0, 1)).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()

        # -------------------- MAIN GT --------------------
        if 'gt_seg_map' in results:
            gt = results['gt_seg_map']
            if gt.ndim == 2:
                data = to_tensor(gt[None, ...].astype(np.int64))
            else:
                warnings.warn(f'gt_seg_map expected 2D but got {gt.shape}')
                data = to_tensor(gt.astype(np.int64))
            data_sample.gt_sem_seg = PixelData(data=data)

        # -------------------- AUX GT ✅ --------------------
        if 'gt_seg_map_aux' in results:
            gt_aux = results['gt_seg_map_aux']
            if gt_aux.ndim == 2:
                data_aux = to_tensor(gt_aux[None, ...].astype(np.int64))
            else:
                warnings.warn(f'gt_seg_map_aux expected 2D but got {gt_aux.shape}')
                data_aux = to_tensor(gt_aux.astype(np.int64))

            # ✅ safest way
            data_sample.set_data(dict(gt_sem_seg_aux=PixelData(data=data_aux)))

        # -------------------- EDGE / DEPTH (unchanged) --------------------
        if 'gt_edge_map' in results:
            data_sample.set_data(dict(
                gt_edge_map=PixelData(data=to_tensor(results['gt_edge_map'][None, ...].astype(np.int64)))
            ))

        if 'gt_depth_map' in results:
            data_sample.set_data(dict(
                gt_depth_map=PixelData(data=to_tensor(results['gt_depth_map'][None, ...]))
            ))

        # -------------------- META --------------------
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)

        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(meta_keys={self.meta_keys})'

'''

'''
@TRANSFORMS.register_module()
class PackSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """



    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        if 'gt_depth_map' in results:
            gt_depth_data = dict(
                data=to_tensor(results['gt_depth_map'][None, ...]))
            data_sample.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str

'''