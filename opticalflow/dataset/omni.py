import os.path as osp
from glob import glob

from .base_flow import FlowDataset


class OmniDataset(FlowDataset):

    def __init__(self,
                 aug_params=None,
                 root='datasets/OMNIFLOWNET_DATASET',
                 dstype='Forest',
                 is_test=False):
        super(OmniDataset, self).__init__(aug_params)

        self.is_test = is_test
        self.dstype = dstype

        for id in range(5):
            if id == 0:
                name = self.dstype
            else:
                name = f'{self.dstype}_{id}'

            flow_root = osp.join(root, dstype, name, 'ground_truth')
            image_root = osp.join(root, dstype, name, 'images')

            image_list = sorted(glob(osp.join(image_root, '*.png')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [i]  # frame_id

            self.flow_list += sorted(
                glob(osp.join(flow_root, '*.flo')))[0:-1]
