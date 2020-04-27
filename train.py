import os
import argparse
from torch.utils.data import DataLoader
from miscs import config_utils as cu
from datasets.kitti import KittiBoxSet
import models
from models.builder import build_from, build_loss
from datasets.kitti.kitti_utils import box_label2tensor, box_image2input

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg_file', type=str, required=True, help='Config file path, required flag.')
parser.add_argument('--kitti_root', type=str, help='KITTI dataset root.')
parser.add_argument('--batch_size', type=int, help='Mini-batch size')
parser.add_argument('--num_workers', type=int, help='Number of workers for DataLoader')
FLAGS = parser.parse_args()

# parse configs
cfg_dict = cu.file2dict(FLAGS.cfg_file)
dataset_cfg = cu.parse_args_update(FLAGS, cfg_dict['dataset_cfg'])
model_cfg = cfg_dict['model_cfg']
loss_cfg = cfg_dict['loss_cfg']
training_cfg = cfg_dict['training_cfg']
loader_cfg = cu.parse_args_update(FLAGS, training_cfg['loader_cfg'])

# build dataset and dataloader
train_set = KittiBoxSet(kitti_root=dataset_cfg['kitti_root'], split='train', 
                        transform=box_image2input(**dataset_cfg['img_norm']), 
                        label_transform=box_label2tensor(dataset_cfg['del_labels']))

train_loader = DataLoader(train_set, shuffle=True, **loader_cfg)


# build model
posenet = build_from(models, model_cfg)

# build loss
dimension_loss = build_loss(loss_cfg['dimension_loss_cfg'])
pose_loss = build_loss(loss_cfg['pose_loss_cfg'])