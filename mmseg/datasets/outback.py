from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class OUTBACKDataset(BaseSegDataset):

    METAINFO = dict(
        classes=("Sky", "Asphalt", "Ground", "Grass", "Rough", "Obstacle"),
        palette=[
            [0, 102, 0],
            [245, 230, 200],
            [125, 82, 8],
            [125, 125, 8],
            [25, 82, 255],
            [100, 180, 75]
        ]
    )

    def __init__(
        self,
        seg_map_path_aux=None,
        seg_map_path_aux2=None,
        data_prefix=None,
        **kwargs
    ):
        if data_prefix is None:
            data_prefix = dict(img_path='', seg_map_path='')

        data_prefix = dict(data_prefix)

        if seg_map_path_aux is not None:
            data_prefix['seg_map_path_aux'] = seg_map_path_aux

        if seg_map_path_aux2 is not None:
            data_prefix['seg_map_path_aux2'] = seg_map_path_aux2

        super().__init__(
            img_suffix='.npy',
            seg_map_suffix='.png',
            data_prefix=data_prefix,
            **kwargs
        )
