import os
import torch
from torch.utils.tensorboard import SummaryWriter

__all__ = [
    'X_Logger'
]

class X_Logger(SummaryWriter):

    def __init__(self, log_dir=None, comment='', purge_step=None,
                 max_queue=10, flush_secs=120, filename_suffix=''):
        self.root_dir = log_dir
        log_dir = os.path.join(self.root_dir, 'tensorboard')
        super(X_Logger, self).__init__(log_dir, comment, purge_step, 
                                       max_queue, flush_secs, filename_suffix)
        self.ckpt_dir = os.path.join(self.root_dir, 'checkpoints')
        os.mkdir(self.ckpt_dir)
    
    def add_parse_args(self, parse_args):
        with open(os.path.join(self.root_dir, 'parse_args.txt'), 'w') as f:
            f.write(str(parse_args))
    
    def add_checkpoint(self, num_epoch, model, optimizer, ckpt_name=None):
        save_dict = {'num_epoch': num_epoch,
                     'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        if ckpt_name is None:
            ckpt_name = 'ckpt_%04d.tar'% num_epoch
        torch.save(save_dict, os.path.join(self.ckpt_dir, ckpt_name))