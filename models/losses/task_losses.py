import torch
import numpy as np
from . import base_losses

__all__ = [
    'Dimension_Loss', 'Pose_Loss'
]


def build_base_loss(cfg):
    attr = getattr(base_losses, cfg.pop('type'))
    return attr(**cfg)


class Dimension_Loss(object):

    def __init__(self, base_loss_cfg, avg_dim=[1., 1., 1.]):
        self.base_loss = build_base_loss(base_loss_cfg)
        self.avg_dim = torch.tensor(avg_dim)
    
    def __call__(self, value, label, weight=None, reduction='mean'):
        target = label - self.avg_dim
        return self.base_loss(value, target, weight, reduction)


class Pose_Loss(object):

    def __init__(self, base_conf_cfg, base_reg_cfg, num_bins, bin_range):
        self.base_conf_loss = build_base_loss(base_conf_cfg)
        self.base_reg_loss = build_base_loss(base_reg_cfg)
        self.bin_centers = torch.arange(num_bins).float() * (2 * np.pi / num_bins)
        self.bin_cos_half_range = torch.cos(bin_range * 0.5)
    
    def label2targets(self, label):
        reg_target = label.view(-1, 1) - self.bin_centers
        reg_target_cos = torch.cos(reg_target)
        reg_mask = (reg_target_cos > self.bin_cos_half_range).float()
        conf_target = torch.argmax(reg_target_cos, dim=1)
        return conf_target, reg_target, reg_mask
    
    def __call__(self, conf_value, reg_value, label, 
                 conf_weight=None, reg_weight=None,
                 conf_reduction='mean',
                 reg_reduction='mean'):
        conf_target, reg_target, reg_mask = self.label2targets(label)
        reg_final_weight = reg_mask if reg_weight is None else reg_mask * reg_weight
        
        conf_loss = self.base_conf_loss(conf_value, conf_target, conf_weight, conf_reduction)
        reg_loss = self.base_reg_loss(reg_value, reg_target, reg_final_weight, reg_reduction)
        return conf_loss, reg_loss