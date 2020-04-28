import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

__all__ = [
    'X_Logger'
]


def create_logger(log_file, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setLevel(log_level if rank == 0 else 'ERROR')
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


class X_Logger(SummaryWriter):

    def __init__(self, log_dir=None, comment='', purge_step=None,
                 max_queue=10, flush_secs=120, filename_suffix=''):
        self.root_dir = log_dir
        log_dir = os.path.join(self.root_dir, 'tensorboard')
        super(X_Logger, self).__init__(log_dir, comment, purge_step, 
                                       max_queue, flush_secs, filename_suffix)
        self.ckpt_dir = os.path.join(self.root_dir, 'checkpoints')
        os.mkdir(self.ckpt_dir)
        self.cmd_logger = create_logger(os.path.join(self.root_dir, 'log_train.txt'))
    
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
    
    def info(self, string):
        self.cmd_logger.info(string)