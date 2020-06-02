import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from . import kitti_utils as ku

__all__ = [
    'KittiBoxSet'
]


class KittiBoxSet(Dataset):

    def __init__(self, kitti_root, split, transform=None, label_transform=None, 
                 augment=False, augment_type=[]):
        assert split in ['train', 'val']
        super(KittiBoxSet, self).__init__()
        self.split = split
        self.image_dir = os.path.join(kitti_root, 'boxes', split, 'image')
        self.label_dir = os.path.join(kitti_root, 'boxes', split, 'label')

        self.image_files = os.listdir(self.image_dir)

        self.transform = transform
        self.label_transform = label_transform

        self.augment = augment
        self.augment_type = augment_type
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        box_image = Image.open(os.path.join(self.image_dir, self.image_files[idx]))
        box_label = ku.read_box_label(os.path.join(self.label_dir, self.image_files[idx][:-4]+'.txt'))

        if self.augment and 'flip' in self.augment_type and torch.rand(1).item() < 0.5:
            box_image = box_image.transpose(Image.FLIP_LEFT_RIGHT)
            box_label['theta_l'] = -box_label['theta_l']

        if self.transform is not None:
            box_image = self.transform(box_image)
        if self.label_transform is not None:
            box_label = self.label_transform(box_label)
        
        return box_image, box_label