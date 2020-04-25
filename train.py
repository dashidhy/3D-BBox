import argparse
import torch
from utils import config_utils as cu
import models

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str, help='Config file path.')
FLAGS = parser.parse_args()

cfg_dict = cu.file2dict(FLAGS.cfg_file)

# build model
model_cfg = cfg_dict['model']
BACKBONE = getattr(models.backbones, model_cfg['backbone_type'])
HEAD = getattr(models.heads, model_cfg['head_type'])

posenet = models.PoseNet(backbone=BACKBONE(**model_cfg['backbone_cfg']),
                         head=HEAD(**model_cfg['head_cfg']))


# debug
fake_img_batch = torch.randn([8, 3, 224, 224])
out = posenet(fake_img_batch)

for ele in out:
    print(ele.size())