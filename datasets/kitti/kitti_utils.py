import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T

__all__ = [
    'read_txt_lines', 'read_scene_labels', 'read_box_label', 'label_scene2box',
    'write_box_label', 'write_box_image', 'box_label2tensor', 'box_image2input'
]


def read_txt_lines(filepath):
    assert filepath.endswith('.txt')
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    return tuple(lines)


def read_scene_labels(filepath):
    sample = filepath[-10:-4]
    lines = read_txt_lines(filepath)
    labels = []
    for line in lines:
        eles = line.split()
        label = {
            'sample': sample,
            'type': eles[0],
            'class': -1,
            'truncated': float(eles[1]),
            'occluded': int(eles[2]),
            'obs_angle': float(eles[3]), # don't use
            'bbox2D': (float(eles[4]), float(eles[5]), float(eles[6]), float(eles[7])), # (l, t, r, b)
            'dimensions': (float(eles[8]), float(eles[9]), float(eles[10])), # (h, w, l)
            'location': (float(eles[11]), float(eles[12]), float(eles[13])), # (x, y, z) in camera frame, i.e., x -> right, y -> down, and z: front
            'ry': float(eles[14]) # right-handed rotation along y, [-pi, pi]
        }
        if label['type'] in ['Car', 'Van']:
            label['class'] = 0
        elif label['type'] in ['Pedestrian', 'Person_sitting']:
            label['class'] = 1
        elif label['type'] == 'Cyclist':
            label['class'] = 2
        labels.append(label)
    return tuple(labels)


def read_box_label(filepath):
    eles = read_txt_lines(filepath)[0].split()
    label = {
        'sample': eles[0],
        'type': eles[1],
        'class': int(eles[2]),
        'dimensions': (float(eles[3]), float(eles[4]), float(eles[5])),
        'theta_l': float(eles[6])
    }
    return label


def label_scene2box(label):
    theta_ray = np.arctan2(label['location'][2], label['location'][0])
    box_label = {
        'sample': label['sample'],
        'type': label['type'],
        'class': label['class'],
        'dimensions': label['dimensions'],
        'theta_l': np.pi - theta_ray - label['ry']
    }
    return box_label


def write_box_label(box_label, filepath, force=False):
    if os.path.exists(filepath) and not force:
        raise FileExistsError('Label file {} exists! Use force=True to conver it.')

    assert filepath.endswith('.txt')

    write_string = ' '.join([box_label['sample'], 
                             box_label['type'], 
                             str(box_label['class']), 
                             str(box_label['dimensions'][0]),
                             str(box_label['dimensions'][1]),
                             str(box_label['dimensions'][2]),
                             str(box_label['theta_l'])])
    with open(filepath, 'w') as f:
        f.write(write_string)
    
    return


def write_box_image(box_image, filepath, force=False):
    if os.path.exists(filepath) and not force:
        raise FileExistsError('Image file {} exists! Use force=True to conver it.')

    assert isinstance(box_image, Image.Image)
    assert filepath.endswith('.png')
    box_image.save(filepath)
    return


class box_label2tensor(object):

    def __call__(self, box_label):
        for key, val in box_label.items():
            if not isinstance(val, str):
                box_label[key] = torch.tensor(val)
        return box_label


class box_image2input(object):

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    
    def __call__(self, box_image):
        return self.transform(box_image)