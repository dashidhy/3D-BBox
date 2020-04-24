import os
from PIL import Image
import torch
from torch.utils.data import Dataset
if __name__ == '__main__':
    import kitti_utils as ku
else:
    from . import kitti_utils as ku

__all__ = [
    'box_label2tensor', 'KittiBoxSet'
]


def box_label2tensor(box_label):
    for key, val in box_label.items():
        if not isinstance(val, str):
            box_label[key] = torch.tensor(val)
    return box_label


class KittiBoxSet(Dataset):

    def __init__(self, kitti_root, split, transform=None, target_transform=None):
        assert split in ['train', 'val']
        super(KittiBoxSet, self).__init__()
        self.split = split
        self.image_dir = os.path.join(kitti_root, 'boxes', split, 'image')
        self.label_dir = os.path.join(kitti_root, 'boxes', split, 'label')

        self.image_files = os.listdir(self.image_dir)

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # get image
        box_image = Image.open(os.path.join(self.image_dir, self.image_files[idx]))
        if self.transform is not None:
            box_image = self.transform(box_image)
        
        # get label
        box_label = ku.read_box_label(os.path.join(self.label_dir, self.image_files[idx][:-4]+'.txt'))
        if self.target_transform is not None:
            box_label = self.target_transform(box_label)
        
        return box_image, box_label


# debug
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    kitti_root = '/home/srip19-pointcloud/datasets/KITTI'
    trans = T.ToTensor()

    boxset = KittiBoxSet(kitti_root, 'train', transform=trans, target_transform=box_label2tensor)
    loader = DataLoader(boxset, batch_size=4, shuffle=True)

    for batch in loader:
        batch_box_tensor, batch_box_label = batch
        break

    print(batch_box_tensor.size())
    print(batch_box_label)