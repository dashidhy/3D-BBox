import numpy as np

class loss_weight_scheduler(object):

    def __init__(self, max_weight, min_weight=None, warmup_iter=None, mode='constant'):
        assert mode in ['constant', 'linear', 'cosine']
        self.mode = mode
        self.max_weight = max_weight
        self.min_weight = self.max_weight * 0.1 if min_weight is None else min_weight
        self.warmup_iter = warmup_iter
        if self.mode == 'constant':
            self.get_loss_weight = self._get_loss_weight_constant
        elif self.mode == 'linear':
            self._linear_step = (self.max_weight - self.min_weight) / self.warmup_iter
            self.get_loss_weight = self._get_loss_weight_linear
        elif self.mode == 'cosine':
            self.get_loss_weight = self._get_loss_weight_cosine
    
    def _get_loss_weight_constant(self, iter):
        return self.max_weight
    
    def _get_loss_weight_linear(self, iter):
        if iter > self.warmup_iter:
            return self.max_weight
        else:
            return self.min_weight + self._linear_step * iter
    
    def _get_loss_weight_cosine(self, iter):
        if iter > self.warmup_iter:
            return self.max_weight
        else:
            return self.min_weight + ((1.0 - np.cos((iter / self.warmup_iter) * np.pi)) * 0.5) * (self.max_weight - self.min_weight)