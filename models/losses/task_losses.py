import torch
from torch import nn
import numpy as np
from . import base_losses

__all__ = [
    'Dimension_Loss', 'Pose_Loss'
]


def build_base_loss(cfg):
    attr = getattr(base_losses, cfg.pop('type'))
    return attr(**cfg)


class Dimension_Loss(nn.Module):

    def __init__(self, base_loss_cfg, avg_dim=(1., 1., 1.), normalize=False):
        super(Dimension_Loss, self).__init__()
        self.base_loss = build_base_loss(base_loss_cfg)
        self.register_buffer('avg_dim', torch.tensor(avg_dim).float())
        self.normalize = normalize
    
    def forward(self, value, label, weight=None, reduction='mean'):
        target = label - self.avg_dim.to(label.device)
        if self.normalize:
            if weight is None:
                weight = 1.0 / label
            else:
                weight *= 1.0 / label
        return self.base_loss(value, target, weight, reduction)


class Pose_Loss(nn.Module):

    def __init__(self, base_conf_cfg, base_reg_cfg, num_bins, bin_range_degree):
        super(Pose_Loss, self).__init__()
        self.base_conf_loss = build_base_loss(base_conf_cfg)
        self.base_reg_loss = build_base_loss(base_reg_cfg)
        bin_centers = torch.arange(num_bins).float() * 2 * np.pi / num_bins
        bin_centers[bin_centers > np.pi] -= 2 * np.pi # to [-pi, pi]
        self.register_buffer('bin_centers', bin_centers)
        self.bin_cos_half_range = np.cos(bin_range_degree * np.pi / 360.0)
    
    def label2targets(self, label):
        reg_target = label.view(-1, 1) - self.bin_centers.to(label.device)
        reg_target[reg_target > np.pi] -= 2 * np.pi # to [-pi, pi]
        reg_target[reg_target < -np.pi] += 2 * np.pi # to [-pi, pi]
        reg_target_cos = torch.cos(reg_target)
        reg_mask = (reg_target_cos > self.bin_cos_half_range).float()
        conf_target = torch.argmax(reg_target_cos, dim=1)
        return conf_target, reg_target, reg_mask
    
    def forward(self, conf_value, reg_value, label, 
                conf_weight=None, reg_weight=None,
                conf_reduction='mean', reg_reduction='mean'):
        conf_target, reg_target, reg_mask = self.label2targets(label)
        reg_final_weight = reg_mask
        if reg_weight is not None: 
            if isinstance(reg_weight, torch.Tensor):
                reg_weight = reg_weight.to(reg_mask.device)
            reg_final_weight *= reg_weight
        
        conf_loss = self.base_conf_loss(conf_value, conf_target, conf_weight, conf_reduction)
        reg_loss = self.base_reg_loss(reg_value, reg_target, reg_final_weight, reg_reduction)
        return conf_loss, reg_loss