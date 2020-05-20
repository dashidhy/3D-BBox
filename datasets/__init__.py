import torch
from torchvision import transforms as T
from . import kitti, nuscenes


class box_label2tensor(object):

    def __init__(self, del_labels=()):
        self.del_labels = del_labels

    def __call__(self, box_label):
        for key in self.del_labels:
            del box_label[key]
        for key, val in box_label.items():
            if not isinstance(val, str):
                box_label[key] = torch.tensor(val)
        return box_label


class box_image2input(object):

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    
    def __call__(self, box_image):
        return self.transform(box_image)


__all__ = [
    'kitti', 'nuscenes', 'box_label2tensor', 'box_label2tensor'
]