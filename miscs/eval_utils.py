import torch
from torch import nn
import numpy as np

__all__ = [
    'Aligned_IoU_3D', 'Orientation_Score'
]


def Aligned_IoU_3D(predict, label):
    label = label.to(predict.device)
    predict = torch.max(predict, torch.tensor(0.0))
    inter = torch.min(predict, label)
    inter_vol = inter[:, 0] * inter[:, 1] * inter[:, 2]
    predict_vol = predict[:, 0] * predict[:, 1] * predict[:, 2]
    label_vol = label[:, 0] * label[:, 1] * label[:, 2]
    return inter_vol / (predict_vol + label_vol - inter_vol)


def Orientation_Score(predict, label):
    return 0.5 * (1.0 + torch.cos(predict - label.to(predict.device)))


class Dimension_Predictor(nn.Module):

    def __init__(self, avg_dim=[1.61057209, 1.47745965, 3.52359498]):
        super(Dimension_Predictor, self).__init__()
        self.register_buffer('avg_dim', torch.tensor(avg_dim).float())
    
    def forward(self, value):
        return value + self.avg_dim.to(value.device)
    
    def predict_and_eval(self, value, label):
        pred = self.forward(value)
        IoU = Aligned_IoU_3D(pred, label)
        return pred, IoU


class Pose_Predictor(nn.Module):

    def __init__(self, num_bins):
        super(Pose_Predictor, self).__init__()
        bin_centers = torch.arange(num_bins).float() * 2 * np.pi / num_bins
        bin_centers[bin_centers > np.pi] -= 2 * np.pi # to [-pi, pi]
        self.register_buffer('bin_centers', bin_centers)
    
    def forward(self, bin_conf_value, bin_reg_value):
        bin_value = bin_reg_value + self.bin_centers.to(bin_reg_value.device)
        bin_value[bin_value > np.pi] -= 2.0 * np.pi
        bin_value[bin_value < -np.pi] += 2.0 * np.pi
        bin_max_conf = torch.argmax(bin_conf_value, dim=1, keepdim=True)
        return torch.gather(bin_value, 1, bin_max_conf).squeeze()
    
    def predict_and_eval(self, bin_conf_value, bin_reg_value, label):
        bin_pred = self.forward(bin_conf_value, bin_reg_value)
        return bin_pred, Orientation_Score(bin_pred, label)