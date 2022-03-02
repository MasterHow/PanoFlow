import os
import os.path as osp
from glob import glob

from .base_flow import FlowDataset


class Flow360(FlowDataset):

    def __init__(self,
                 aug_params=None,
                 split='train',
                 root='datasets/Flow360',
                 dstype='sunny'):
        super(Flow360, self).__init__(aug_params)

        flow_root = osp.join(root, split, dstype, 'flow')
        image_root = osp.join(root, split, dstype, 'img')

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.jpg')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))
