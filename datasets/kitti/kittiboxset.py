import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from . import kitti_utils as ku

__all__ = [
    'KittiBoxSet'
]


class KittiBoxSet(Dataset):

    def __init__(self, kitti_root, split, transform=None, label_transform=None):
        assert split in ['train', 'val']
        super(KittiBoxSet, self).__init__()
        self.split = split
        self.image_dir = os.path.join(kitti_root, 'boxes', split, 'image')
        self.label_dir = os.path.join(kitti_root, 'boxes', split, 'label')

        self.image_files = os.listdir(self.image_dir)

        self.transform = transform
        self.label_transform = label_transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # get image
        box_image = Image.open(os.path.join(self.image_dir, self.image_files[idx]))
        if self.transform is not None:
            box_image = self.transform(box_image)
        
        # get label
        box_label = ku.read_box_label(os.path.join(self.label_dir, self.image_files[idx][:-4]+'.txt'))
        if self.label_transform is not None:
            box_label = self.label_transform(box_label)
        
        return box_image, box_label