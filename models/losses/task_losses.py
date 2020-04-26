import torch
from . import base_losses as bl 


class TaskLoss(object):

    def __init__(self, base_loss_cfg):
        loss_class = getattr(bl, base_loss_cfg.pop('type'))
        self.base_loss = loss_class(**base_loss_cfg)

    def label2target(self, label):
        raise NotImplementedError('label2target method not implemented!')

    def __call__(self, value, label, weight=None, reduction='mean'):
        target = self.label2target(label)
        return self.base_loss(value, target, weight, reduction)


class Dimension_Loss(TaskLoss):

    def __init__(self, base_loss_cfg, avg_dim=[1., 1., 1.]):
        super(Dimension_Loss, self).__init__(base_loss_cfg)
        self.avg_dim = torch.tensor(avg_dim)
    
    def label2target(self, label):
        """
        Input:
            label: Tensor(N, 3), gt dimensions
        Return:
            target: Tensor(N, 3), residual dimensions w.r.t avg dimension
        """
        return label - self.avg_dim


class Bin_Confidence_Loss(TaskLoss):

    def __init__(self, base_loss_cfg, bin_centers):
        """
        bin_centers: Tensor(num_bins)
        """
        super(Bin_Confidence_Loss, self).__init__(base_loss_cfg)
        self.bin_centers = bin_centers
    
    def label2target(self, label):
        """
        Input:
            label: Tensor(N, dtype=float), theta_l
        Return:
            target: Tensor(N, dtype=int), bin classes,
                    theta_l is assigned to the class of
                    nearest bin center
        """
        label_to_center = torch.cos(label.view(-1, 1) - self.bin_centers) # (N, num_bins)
        return torch.argmax(label_to_center, dim=1)


class Bin_Regression_Loss(TaskLoss):

    def __init__(self, base_loss_cfg, bin_centers, bin_range):
        super(Bin_Regression_Loss, self).__init__(base_loss_cfg)
        self.bin_centers = bin_centers
        self.bin_cos_half_range = torch.cos(bin_range * 0.5)
    
    def label2target(self, label):
        """
        Input:
            label: Tensor(N), theta_l
        Return:
            target: Tensor(N, num_bins)
            mask: Tensor(N, num_bins)
        """
        target = label.view(-1, 1) - self.bin_centers
        mask = (torch.cos(target) < self.bin_cos_half_range).float()
        return target, mask

    def __call__(self, value, label, weight=None, reduction='mean'):
        target, mask = self.label2target(label)
        final_weight = mask if weight is None else mask * weight
        return self.base_loss(value, target, final_weight, reduction)