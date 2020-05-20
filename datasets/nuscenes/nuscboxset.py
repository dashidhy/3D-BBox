import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

__all__ = [
    'NuscBoxSet'
]


class NuscBoxSet(Dataset):

    def __init__(self, nusc_root, split, transform=None, label_transform=None):
        assert split in ['train', 'val']
        super(NuscBoxSet, self).__init__()
        self.split = split
        self.nusc_root = nusc_root
        image_anns_file = os.path.join(nusc_root, 'boxes', split, 'image_annotations.json')

        print('Loading image annotations...')
        with open(image_anns_file, 'r') as f:
            self.image_anns = json.load(f)
        print('Done. {} annotations loaded.'.format(self.__len__()))

        self.transform = transform
        self.label_transform = label_transform
    
    def __len__(self):
        return len(self.image_anns)
    
    def __getitem__(self, idx):
        
        # get image
        box_image = Image.open(os.path.join(self.nusc_root, self.image_anns[idx]['box_image_file']))
        if self.transform is not None:
            box_image = self.transform(box_image)
        
        # get label
        box_label = self.image_anns[idx].copy()
        if self.label_transform is not None:
            box_label = self.label_transform(box_label)
        
        return box_image, box_label


# debug
if __name__ == '__main__':
    nusc_box_set = NuscBoxSet('/home/srip19-pointcloud/datasets/NuScenes', 'train')
    _, box_label = nusc_box_set[12837]
    print(box_label)