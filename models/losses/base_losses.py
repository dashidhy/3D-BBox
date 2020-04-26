import torch
from torch.nn import functional as F

__all__ = [
    'BaseLoss', 'MSE', 'Smooth_L1', 'CrossEntropy', 'Cosine_Expansion'
]

class BaseLoss(object):

    def __call__(self, value, target, weight=None, reduction='mean'):

        loss = self.loss_func(value, target)

        if weight is not None:
            loss *= weight
        
        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'mean':
            return loss.mean()
        else:
            raise ValueError('Unsupported reduction type \'{}\''.format(reduction))

    def loss_func(self, value, target):
        raise NotImplementedError('Loss function is not inplemented yet!')


class MSE(BaseLoss):

    def __init__(self):
        super(MSE, self).__init__()

    def loss_func(self, value, target):
        return F.mse_loss(value, target, reduction='none')


class Smooth_L1(BaseLoss):

    def __init__(self):
        super(Smooth_L1, self).__init__()

    def loss_func(self, value, target):
        return F.smooth_l1_loss(value, target, reduction='none')


class CrossEntropy(BaseLoss):

    def __init__(self, cls_weight=None, ignore_index=-1):
        super(CrossEntropy, self).__init__()
        self.cls_weight = cls_weight
        self.ignore_index = ignore_index

    def loss_func(self, value, target):
        return F.cross_entropy(value, target,
                               weight=self.cls_weight,
                               ignore_index=self.ignore_index,
                               reduction='none')


class Cosine_Expansion(BaseLoss):

    def __init__(self, normalize=False):
        super(Cosine_Expansion, self).__init__()
        self.normalize = normalize

    def loss_func(self, value, target):

        assert value.size(-1) == 2

        if self.normalize:
            value_norm = torch.norm(value, dim=-1, keepdim=True)
            value /= value_norm

        return -(torch.cos(target) * value[..., 0] + torch.sin(target) * value[..., 1])