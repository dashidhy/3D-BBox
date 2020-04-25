import torch
from torch import nn

__all__ = [
    'PoseNet'
]


class PoseNet(nn.Module):

    def __init__(self, backbone, head):
        super(PoseNet, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x