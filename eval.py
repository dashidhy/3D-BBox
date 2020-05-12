import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from miscs import config_utils as cu, eval_utils as eu, create_logger
from datasets.kitti import KittiBoxSet
import models
from models.builder import build_from
from datasets.kitti import kitti_utils as ku
from bbox3D_estimate_v2 import dimensions_to_corners, solve_3d_bbox_single

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg_file', type=str, required=True, help='Config file path, required flag.')
parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint to be evaluated.')
parser.add_argument('--log_dir', type=str, help='Folder to save experiment records.')
parser.add_argument('--kitti_root', type=str, help='KITTI dataset root.')
parser.add_argument('--batch_size', type=int, help='Mini-batch size')
parser.add_argument('--num_workers', type=int, help='Number of workers for DataLoader')
FLAGS = parser.parse_args()

calib_root = os.path.join(FLAGS.kitti_root, 'training', 'calib')

# create logger
os.makedirs(FLAGS.log_dir, exist_ok=True)
logger = create_logger(os.path.join(FLAGS.log_dir, 'log_eval.txt'))

# parse configs
cfg_dict = cu.file2dict(FLAGS.cfg_file)
dataset_cfg = cu.parse_args_update(FLAGS, cfg_dict['dataset_cfg']).copy()
model_cfg = cfg_dict['model_cfg'].copy()
loss_cfg = cfg_dict['loss_cfg'].copy()
training_cfg = cfg_dict['training_cfg'].copy()
loader_cfg = cu.parse_args_update(FLAGS, training_cfg['loader_cfg']).copy()
loader_cfg['drop_last'] = False

# build dataset and dataloader
train_set = KittiBoxSet(kitti_root=dataset_cfg['kitti_root'], split='train', 
                        transform=ku.box_image2input(**dataset_cfg['img_norm']), 
                        label_transform=ku.box_label2tensor())

train_loader = DataLoader(train_set, shuffle=True, **loader_cfg)

total_train_sample = len(train_loader) * train_loader.batch_size if train_loader.drop_last else len(train_loader.dataset)

val_set = KittiBoxSet(kitti_root=dataset_cfg['kitti_root'], split='val', 
                      transform=ku.box_image2input(**dataset_cfg['img_norm']), 
                      label_transform=ku.box_label2tensor())

val_loader = DataLoader(val_set, shuffle=True, **loader_cfg)

total_val_sample = len(val_loader) * val_loader.batch_size if val_loader.drop_last else len(val_loader.dataset)

# build model, load from checkpoint
posenet = build_from(models, model_cfg)
logger.info('Load checkpoint from: ' + os.path.abspath(FLAGS.ckpt))
ckpt_dict = torch.load(FLAGS.ckpt)
posenet.load_state_dict(ckpt_dict['model'])
posenet = posenet.cuda()
posenet.eval()

# build predictor
dimension_predictor = eu.Dimension_Predictor(loss_cfg['dimension_loss_cfg']['avg_dim']).cuda()
pose_predictor = eu.Pose_Predictor(loss_cfg['pose_loss_cfg']['num_bins']).cuda()

# eval train set
dim_over_50 = 0
dim_over_70 = 0
bin_over_90 = 0
bin_over_95 = 0
loc_less_01 = 0
loc_less_03 = 0
loc_less_05 = 0
logger.info('EVAL TRAIN SET...')
for batch_image, batch_label in tqdm(train_loader):

    # load batch data to gpu
    batch_image_cuda = batch_image.cuda()
    batch_dim_label_cuda = batch_label['dimensions'].cuda()
    batch_theta_l_label_cuda = batch_label['theta_l'].cuda()

    # forward
    with torch.no_grad():
        dim_reg, bin_conf, bin_reg = posenet(batch_image_cuda)
    
    # predict
    dim_pred, dim_pred_score = dimension_predictor.predict_and_eval(dim_reg, batch_dim_label_cuda)
    bin_pred, bin_pred_score = pose_predictor.predict_and_eval(bin_conf, bin_reg, batch_theta_l_label_cuda)

    dim_over_50 += (dim_pred_score > 0.50).sum().item()
    dim_over_70 += (dim_pred_score > 0.70).sum().item()
    bin_over_90 += (bin_pred_score > 0.90).sum().item()
    bin_over_95 += (bin_pred_score > 0.95).sum().item()

    # estimate location
    bbox2D = torch.unbind(batch_label['bbox2D'])
    dim_pred = dim_pred.cpu()
    corners = torch.unbind(dimensions_to_corners(dim_pred))
    theta_l = torch.unbind(bin_pred.cpu())
    location = torch.unbind(batch_label['location'])
    for i in range(len(dim_pred)):
        sample_id = batch_label['sample'][i]
        calib = ku.read_calib(os.path.join(calib_root, sample_id+'.txt'))
        location_pred = solve_3d_bbox_single(bbox2D[i], corners[i], theta_l[i], calib)
        location_error = torch.norm(location_pred - location[i])
        if location_error < 1.0:
            loc_less_01 += 1
        if location_error < 3.0:
            loc_less_03 += 1
        if location_error < 5.0:
            loc_less_05 += 1

dim_over_50 /= total_train_sample
dim_over_70 /= total_train_sample
bin_over_90 /= total_train_sample
bin_over_95 /= total_train_sample
loc_less_01 /= total_train_sample
loc_less_03 /= total_train_sample
loc_less_05 /= total_train_sample

logger.info('A-IoU 3D @ 0.50: %6.4f | A-IoU 3D @ 0.70: %6.4f | OS @ 0.90: %6.4f | OS @ 0.95: %6.4f | LOC @ 1.0: %6.4f | LOC @ 3.0: %6.4f | LOC @ 5.0: %6.4f' \
             % (dim_over_50, dim_over_70, bin_over_90, bin_over_95, loc_less_01, loc_less_03, loc_less_05))


# eval val set
dim_over_50 = 0
dim_over_70 = 0
bin_over_90 = 0
bin_over_95 = 0
loc_less_01 = 0
loc_less_03 = 0
loc_less_05 = 0
logger.info('EVAL Val SET...')
for batch_image, batch_label in tqdm(val_loader):

    # load batch data to gpu
    batch_image_cuda = batch_image.cuda()
    batch_dim_label_cuda = batch_label['dimensions'].cuda()
    batch_theta_l_label_cuda = batch_label['theta_l'].cuda()

    # forward
    with torch.no_grad():
        dim_reg, bin_conf, bin_reg = posenet(batch_image_cuda)
    
    # predict
    dim_pred, dim_pred_score = dimension_predictor.predict_and_eval(dim_reg, batch_dim_label_cuda)
    bin_pred, bin_pred_score = pose_predictor.predict_and_eval(bin_conf, bin_reg, batch_theta_l_label_cuda)

    dim_over_50 += (dim_pred_score > 0.50).sum().item()
    dim_over_70 += (dim_pred_score > 0.70).sum().item()
    bin_over_90 += (bin_pred_score > 0.90).sum().item()
    bin_over_95 += (bin_pred_score > 0.95).sum().item()

    # estimate location
    bbox2D = torch.unbind(batch_label['bbox2D'])
    dim_pred = dim_pred.cpu()
    corners = torch.unbind(dimensions_to_corners(dim_pred))
    theta_l = torch.unbind(bin_pred.cpu())
    location = torch.unbind(batch_label['location'])
    for i in range(len(dim_pred)):
        sample_id = batch_label['sample'][i]
        calib = ku.read_calib(os.path.join(calib_root, sample_id+'.txt'))
        location_pred = solve_3d_bbox_single(bbox2D[i], corners[i], theta_l[i], calib)
        location_error = torch.norm(location_pred - location[i])
        if location_error < 1.0:
            loc_less_01 += 1
        if location_error < 3.0:
            loc_less_03 += 1
        if location_error < 5.0:
            loc_less_05 += 1

dim_over_50 /= total_val_sample
dim_over_70 /= total_val_sample
bin_over_90 /= total_val_sample
bin_over_95 /= total_val_sample
loc_less_01 /= total_val_sample
loc_less_03 /= total_val_sample
loc_less_05 /= total_val_sample

logger.info('A-IoU 3D @ 0.50: %6.4f | A-IoU 3D @ 0.70: %6.4f | OS @ 0.90: %6.4f | OS @ 0.95: %6.4f | LOC @ 1.0: %6.4f | LOC @ 3.0: %6.4f | LOC @ 5.0: %6.4f' \
             % (dim_over_50, dim_over_70, bin_over_90, bin_over_95, loc_less_01, loc_less_03, loc_less_05))