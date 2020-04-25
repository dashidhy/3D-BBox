from functools import partial
from torch.nn import functional as F


class BaseLoss(object):

    def __init__(self):
        self.loss_func = None

    def __call__(self, value, target, weight=None, reduction='mean'):
        if self.loss_func is None:
            raise NotImplementedError('Loss function is not inplemented yet!')

        loss = self.loss_func(value, target, reduction='none')

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


class MSE(BaseLoss):

    def __init__(self):
        super(MSE, self).__init__()
        self.loss_func = F.mse_loss


class Smooth_L1(BaseLoss):

    def __init__(self):
        super(Smooth_L1, self).__init__()
        self.loss_func = F.smooth_l1_loss


class CrossEntropy(BaseLoss):

    def __init__(self, cls_weight=None, ignore_index=-100):
        super(CrossEntropy, self).__init__()
        self.loss_func = partial(F.cross_entropy, 
                                 weight=cls_weight, 
                                 ignore_index=ignore_index)