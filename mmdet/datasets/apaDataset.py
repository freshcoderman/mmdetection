import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class ApaDataset(CustomDataset):

    CLASSES = (
    'thick_cylinder', 
    'thin_cylinder', 
    'triangle', 
    'special_car',
    'truck_box',
    'motortricycle',
    'tricycle',
    'biking_person',
    'car',
    'truck_nobox_full',
    'bus',
    'ride_bicycle_person',
    'walking_person',
    'other')

    def load_annotations(self, ann_file):
        data_infos = mmcv.load(ann_file)
        for item in data_infos:
            item['ann']['bboxes'] = np.array(item['ann']['bboxes']).astype(np.float32)
            item['ann']['anchors'] = np.array(item['ann']['anchors']).astype(np.float32)
            item['ann']['bboxes'][:,2:4] = item['ann']['bboxes'][:,2:4] +item['ann']['bboxes'][:,0:2]
            item['ann']['labels'] = np.array(item['ann']['labels']).astype(np.float32) - 1
        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['anchor_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []