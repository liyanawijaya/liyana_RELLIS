from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import os

@DATASETS.register_module()
class OUTBACKDataset(BaseSegDataset):

    METAINFO = dict(
        #classes=("BACKGROUND", "grass tree", "pole", "tree", "leaves", "fence net", "log", "grass", 
        #    "road sign", "small branch", "gravel", "ground", "horizon", "roots", "sky", 
        #    "delineator", "rock",),
        
        #classes=("Tree", "Leaves", "Ground", "Fence", "Grass", "Log", "Grasstree", "Sky", 
        #    "Road sign", "Small branch", "Delineator", "Asphalt", "Rubble", "Rock", "Non-nav", 
        #    "Rough", "Nan",),
        #palette=[[0, 102, 0], [245, 230, 200], [125, 82, 8], [25, 82, 255], [100, 180, 75], [255, 173, 10], 
        #   [54, 255, 25], [26, 172, 255], [204, 108, 231], [115, 250, 235], [233, 108, 108],
        #  [84, 84, 84], [226, 255, 25], [6, 57, 112], [255, 197, 25],
        #  [140, 255, 25], [255, 154, 25]]
        
        classes=("Sky", "Asphalt", "Ground","Grass","Rough","Obstacle"),
        palette=[[0, 102, 0], [245, 230, 200],[125, 82, 8], [125, 82, 8],[25, 82, 255],[100, 180, 75]]
)
    # inside OUTBACKDataset class


    def __init__(self, **kwargs):
        super(OUTBACKDataset, self).__init__(
            img_suffix='.png',
            #img_suffix='.npy',
            seg_map_suffix='.png',
            #seg_map_path_aux='.png',
            **kwargs)
        
