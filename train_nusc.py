import os
import argparse
import torch
from torch.utils.data import DataLoader
from miscs import config_utils as cu, eval_utils as eu, X_Logger
from datasets.nuscenes import NuscBoxSet
import models
from models.builder import build_from, build_loss
from datasets import box_label2tensor, box_image2input

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg_file', type=str, required=True, help='Config file path, required flag.')
parser.add_argument('--log_dir', type=str, help='Folder to save experiment records.')
parser.add_argument('--nusc_root', type=str, help='NuScenes dataset root.')
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

# build logger and backup configs
logger = X_Logger(log_cfg['log_dir'])
logger.add_parse_args(FLAGS)
logger.add_config_file(FLAGS.cfg_file)

# build dataset and dataloader
train_set = NuscBoxSet(nusc_root=dataset_cfg['nusc_root'], split='train', 
                       transform=box_image2input(**dataset_cfg['img_norm']), 
                       label_transform=box_label2tensor(dataset_cfg['del_labels']))

train_loader = DataLoader(train_set, shuffle=True, **loader_cfg)

total_train_sample = len(train_loader) * train_loader.batch_size if train_loader.drop_last else len(train_loader.dataset)

val_set = NuscBoxSet(nusc_root=dataset_cfg['nusc_root'], split='val', 
                     transform=box_image2input(**dataset_cfg['img_norm']), 
                     label_transform=box_label2tensor(dataset_cfg['del_labels']))

val_loader = DataLoader(val_set, shuffle=True, **loader_cfg)

total_val_sample = len(val_loader) * val_loader.batch_size if val_loader.drop_last else len(val_loader.dataset)

# build model
posenet = build_from(models, model_cfg).cuda()

# build loss
dimension_loss = build_loss(loss_cfg['dimension_loss_cfg'].copy()).cuda()
pose_loss = build_loss(loss_cfg['pose_loss_cfg'].copy()).cuda()

# build optimizer
optim_type = getattr(torch.optim, optimizer_cfg.pop('type'))
optimizer = optim_type(posenet.parameters(), **optimizer_cfg)

# build predictor
dimension_predictor = eu.Dimension_Predictor(loss_cfg['dimension_loss_cfg']['avg_dim']).cuda()
pose_predictor = eu.Pose_Predictor(loss_cfg['pose_loss_cfg']['num_bins']).cuda()

# begin training
logger.info('TRAINING BEGINS!!!')
iteration = 0
for epoch in range(training_cfg['total_epoch']):

    logger.info('******** EPOCH %03d ********' % epoch)

    for batch_image, batch_label in train_loader:

        # load batch data to gpu
        batch_image_cuda = batch_image.cuda()
        batch_dim_label_cuda = batch_label['dimensions'].cuda()
        batch_theta_l_label_cuda = batch_label['theta_l'].cuda()

        # forward
        optimizer.zero_grad()
        dim_reg, bin_conf, bin_reg = posenet(batch_image_cuda)

        # loss
        dim_reg_loss = dimension_loss(dim_reg, batch_dim_label_cuda, reduction='batch_mean')
        bin_conf_loss, bin_reg_loss = pose_loss(bin_conf, bin_reg, batch_theta_l_label_cuda, reg_reduction='batch_mean')
        loss = dim_reg_loss * loss_cfg['loss_weights']['dim_reg'] +  \
               bin_conf_loss * loss_cfg['loss_weights']['bin_conf'] + \
               bin_reg_loss * loss_cfg['loss_weights']['bin_reg']
        
        # optimize
        loss.backward()
        optimizer.step()

        # log
        if iteration % log_cfg['log_loss_every'] == 0:
            logger.add_scalar('Loss/dim_reg', dim_reg_loss, iteration)
            logger.add_scalar('Loss/bin_conf', bin_conf_loss, iteration)
            logger.add_scalar('Loss/bin_reg', bin_reg_loss, iteration)
        
        if iteration % log_cfg['show_loss_every'] == 0:
            logger.info('batch %05d | dim_reg: %8.6f | bin_conf: %8.6f | bin_reg: %9.6f' \
                        % (iteration, dim_reg_loss.item(), bin_conf_loss.item(), bin_reg_loss.item()))

        iteration += 1
    
    # checkpoint
    if (epoch + 1) % log_cfg['ckpt_every'] == 0:
        logger.add_checkpoint(epoch + 1, posenet, optimizer)
    
    # eval
    if (epoch + 1) % log_cfg['eval_every'] == 0:
        logger.info('EVAL ...')
        posenet.eval()

        # eval train set
        dim_over_50 = 0
        dim_over_70 = 0
        bin_over_90 = 0
        bin_over_95 = 0
        for batch_image, batch_label in train_loader:
            
            # load batch data to gpu
            batch_image_cuda = batch_image.cuda()
            batch_dim_label_cuda = batch_label['dimensions'].cuda()
            batch_theta_l_label_cuda = batch_label['theta_l'].cuda()

            # forward
            with torch.no_grad():
                dim_reg, bin_conf, bin_reg = posenet(batch_image_cuda)
            
            # predict
            _, dim_pred_score = dimension_predictor.predict_and_eval(dim_reg, batch_dim_label_cuda)
            _, bin_pred_score = pose_predictor.predict_and_eval(bin_conf, bin_reg, batch_theta_l_label_cuda)
            dim_over_50 += (dim_pred_score > 0.50).sum().item()
            dim_over_70 += (dim_pred_score > 0.70).sum().item()
            bin_over_90 += (bin_pred_score > 0.90).sum().item()
            bin_over_95 += (bin_pred_score > 0.95).sum().item()
        
        dim_over_50 = dim_over_50 / total_train_sample
        dim_over_70 = dim_over_70 / total_train_sample
        bin_over_90 = bin_over_90 / total_train_sample
        bin_over_95 = bin_over_95 / total_train_sample

        logger.add_scalar('Eval_TRAIN/Aligned_IoU_3D_0.50', dim_over_50, epoch + 1)
        logger.add_scalar('Eval_TRAIN/Aligned_IoU_3D_0.70', dim_over_70, epoch + 1)
        logger.add_scalar('Eval_TRAIN/OS_0.90', bin_over_90, epoch + 1)
        logger.add_scalar('Eval_TRAIN/OS_0.95', bin_over_95, epoch + 1)

        logger.info('TRAIN SET | A-IoU 3D @ 0.50: %6.4f | A-IoU 3D @ 0.70: %6.4f | OS @ 0.90: %6.4f | OS @ 0.95: %6.4f' \
                    % (dim_over_50, dim_over_70, bin_over_90, bin_over_95))
        
        # eval val set
        dim_over_50 = 0
        dim_over_70 = 0
        bin_over_90 = 0
        bin_over_95 = 0
        for batch_image, batch_label in val_loader:
            
            # load batch data to gpu
            batch_image_cuda = batch_image.cuda()
            batch_dim_label_cuda = batch_label['dimensions'].cuda()
            batch_theta_l_label_cuda = batch_label['theta_l'].cuda()

            # forward
            with torch.no_grad():
                dim_reg, bin_conf, bin_reg = posenet(batch_image_cuda)
            
            # predict
            _, dim_pred_score = dimension_predictor.predict_and_eval(dim_reg, batch_dim_label_cuda)
            _, bin_pred_score = pose_predictor.predict_and_eval(bin_conf, bin_reg, batch_theta_l_label_cuda)
            dim_over_50 += (dim_pred_score > 0.50).sum().item()
            dim_over_70 += (dim_pred_score > 0.70).sum().item()
            bin_over_90 += (bin_pred_score > 0.90).sum().item()
            bin_over_95 += (bin_pred_score > 0.95).sum().item()
        
        dim_over_50 = dim_over_50 / total_val_sample
        dim_over_70 = dim_over_70 / total_val_sample
        bin_over_90 = bin_over_90 / total_val_sample
        bin_over_95 = bin_over_95 / total_val_sample

        logger.add_scalar('Eval_VAL/Aligned_IoU_3D_0.50', dim_over_50, epoch + 1)
        logger.add_scalar('Eval_VAL/Aligned_IoU_3D_0.70', dim_over_70, epoch + 1)
        logger.add_scalar('Eval_VAL/OS_0.90', bin_over_90, epoch + 1)
        logger.add_scalar('Eval_VAL/OS_0.95', bin_over_95, epoch + 1)

        logger.info('VALID SET | A-IoU 3D @ 0.50: %6.4f | A-IoU 3D @ 0.70: %6.4f | OS @ 0.90: %6.4f | OS @ 0.95: %6.4f' \
                    % (dim_over_50, dim_over_70, bin_over_90, bin_over_95))

        # eval val set
        posenet.train()

logger.info('TRAINING ENDS!!!')