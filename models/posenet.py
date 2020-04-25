import torch
from torch import nn
from . import builder as bd

__all__ = [
    'PoseNet'
]


class PoseNet(nn.Module):

    def __init__(self, backbone_cfg, head_cfg):
        super(PoseNet, self).__init__()
        self.backbone = bd.build_backbone(backbone_cfg)
        self.head = bd.build_head(head_cfg)
    
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x