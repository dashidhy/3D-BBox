import os
import argparse
import torch
from torch.utils.data import DataLoader
from miscs import config_utils as cu, X_Logger
from datasets.kitti import KittiBoxSet
import models
from models.builder import build_from, build_loss
from datasets.kitti.kitti_utils import box_label2tensor, box_image2input

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg_file', type=str, required=True, help='Config file path, required flag.')
parser.add_argument('--log_dir', type=str, help='Folder to save experiment records.')
parser.add_argument('--kitti_root', type=str, help='KITTI dataset root.')
parser.add_argument('--batch_size', type=int, help='Mini-batch size')
parser.add_argument('--num_workers', type=int, help='Number of workers for DataLoader')
FLAGS = parser.parse_args()

# parse configs
cfg_dict = cu.file2dict(FLAGS.cfg_file)
dataset_cfg = cu.parse_args_update(FLAGS, cfg_dict['dataset_cfg']).copy()
model_cfg = cfg_dict['model_cfg'].copy()
loss_cfg = cfg_dict['loss_cfg'].copy()
training_cfg = cfg_dict['training_cfg'].copy()
loader_cfg = cu.parse_args_update(FLAGS, training_cfg['loader_cfg']).copy()
optimizer_cfg = cu.parse_args_update(FLAGS, training_cfg['optimizer_cfg']).copy()
log_cfg = cu.parse_args_update(FLAGS, cfg_dict['log_cfg']).copy()

# log parse_args
logger = X_Logger(log_cfg['log_dir'])
logger.add_parse_args(FLAGS)

# build dataset and dataloader
train_set = KittiBoxSet(kitti_root=dataset_cfg['kitti_root'], split='train', 
                        transform=box_image2input(**dataset_cfg['img_norm']), 
                        label_transform=box_label2tensor(dataset_cfg['del_labels']))

train_loader = DataLoader(train_set, shuffle=True, **loader_cfg)


# build model
posenet = build_from(models, model_cfg).cuda()

# build loss
dimension_loss = build_loss(loss_cfg['dimension_loss_cfg'].copy()).cuda()
pose_loss = build_loss(loss_cfg['pose_loss_cfg'].copy()).cuda()

# build optimizer
optim_type = getattr(torch.optim, optimizer_cfg.pop('type'))
optimizer = optim_type(posenet.parameters(), **optimizer_cfg)