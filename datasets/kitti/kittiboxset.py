import os
from PIL import Image
from torch.utils.data import Dataset
if __name__ == '__main__':
    import kitti_utils as ku
else:
    from . import kitti_utils as ku

__all__ = [
    'KittiBoxSet'
]

class KittiBoxSet(Dataset):

    def __init__(self, kittiroot, split, transform=None, target_transform=None):
        assert split in ['train', 'val']
        super(KittiBoxSet, self).__init__()
        self.split = split
        self.image_dir = os.path.join(kittiroot, 'boxes', split, 'image')
        self.label_dir = os.path.join(kittiroot, 'boxes', split, 'label')

        self.box_ids = [filename[:-4] for filename in os.listdir(self.image_dir)]
        self.num_boxes = len(self.box_ids)

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.num_boxes
    
    def __getitem__(self, idx):
        return self.box_ids[idx]