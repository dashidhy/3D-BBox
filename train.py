import argparse
import torch
from utils import config_utils as cu
from models import PoseNet

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str, help='Config file path.')
FLAGS = parser.parse_args()

cfg_dict = cu.file2dict(FLAGS.cfg_file)

# build model
model_cfg = cfg_dict['model']
BACKBONE = cu.import_from('models.backbones', model_cfg['backbone_type'])
HEAD = cu.import_from('models.heads', model_cfg['head_type'])

posenet = PoseNet(backbone=BACKBONE(**model_cfg['backbone_cfg']),
                  head=HEAD(**model_cfg['head_cfg']))


# debug
fake_img_batch = torch.randn([8, 3, 224, 224])
out = posenet(fake_img_batch)

for ele in out:
    print(ele.size())